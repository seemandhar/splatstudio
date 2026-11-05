"""Python drivers for the vendored FlashGS kernels and the reference 3DGS / gsplat forwards.

Conventions (SplatStudio): camera = {"viewmat": OpenCV world->camera [4,4], "K": [3,3],
"width", "height"}; params hold log-scales, wxyz quats, logit opacities, sh0 [N,1,3], shN [N,15,3].
"""

import math

import torch

TILE = 16  # example.py renders with render_16x16
ALPHA_PAIR_MIN = 1.0 / 255.0


# --------------------------------------------------------------------------- model inputs
def quat_to_rotmat(quats):
    q = quats / quats.norm(dim=-1, keepdim=True)
    w, x, y, z = q.unbind(-1)
    return torch.stack(
        [
            1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
            2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
            2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
        ],
        dim=-1,
    ).reshape(*q.shape[:-1], 3, 3)


def prepare(params, sh_degree=3):
    """FlashGS model tensors (what upstream `ops.loadPly` returns): xyz, 16x3 SH, sigmoid opacity,
    upper-triangular world covariance Sigma = R S S^T R^T. SH bands above `sh_degree` are zeroed,
    which equals evaluating SH at that degree (the kernel always evaluates degree 3)."""
    with torch.no_grad():
        means = params["means"].detach().float().contiguous()
        shs = torch.cat((params["sh0"], params["shN"]), dim=1).detach().float().clone()
        if shs.shape[1] != 16:
            raise ValueError(f"FlashGS needs 16 SH coefficients per Gaussian, got {shs.shape[1]}")
        shs[:, (sh_degree + 1) ** 2 :] = 0.0
        opacity = params["opacities"].detach().float().sigmoid().reshape(-1).contiguous()
        R = quat_to_rotmat(params["quats"].detach().float())
        s2 = params["scales"].detach().float().exp().square()
        cov = (R * s2[:, None, :]) @ R.transpose(1, 2)
        cov6 = torch.stack(
            [cov[:, 0, 0], cov[:, 0, 1], cov[:, 0, 2], cov[:, 1, 1], cov[:, 1, 2], cov[:, 2, 2]],
            dim=-1,
        ).contiguous()
        return {"position": means, "shs": shs.reshape(-1, 48).contiguous(),
                "opacity": opacity, "cov3d": cov6}


def camera_terms(camera):
    """Camera centre, camera->world rotation (row-major, as 3DGS cameras.json 'rotation'),
    focal lengths. FlashGS (and 3DGS) project with the principal point at the image centre."""
    K = camera["K"].detach().float().cpu()
    W, H = int(camera["width"]), int(camera["height"])
    if abs(float(K[0, 2]) - W / 2) > 1e-3 or abs(float(K[1, 2]) - H / 2) > 1e-3:
        raise ValueError(
            f"FlashGS assumes a centred principal point; got cx={float(K[0, 2])}, "
            f"cy={float(K[1, 2])} for {W}x{H}"
        )
    V = camera["viewmat"].detach().float().cpu()
    Rw2c, t = V[:3, :3], V[:3, 3]
    position = -(Rw2c.T @ t)
    return position.contiguous(), Rw2c.T.contiguous(), float(K[0, 0]), float(K[1, 1]), W, H


def scaled_camera(camera, width):
    """Same pose and field of view at another resolution (paper's 1080p / 4K renders)."""
    W, H = int(camera["width"]), int(camera["height"])
    s = width / W
    h = round(H * s)
    K = camera["K"].clone().float()
    K[0, 0] *= s
    K[1, 1] *= h / H
    K[0, 2], K[1, 2] = width / 2, h / 2
    return {**camera, "K": K, "width": int(width), "height": h}


# --------------------------------------------------------------------------- AABB bound
def bbox_tile_counts(prep, camera, tile=TILE):
    """Pure-torch replica of FlashGS's coarse rectangle (getRect over the opacity-aware tight
    box, before the exact ellipse test): an upper bound on key/value pairs it can emit."""
    pos_c, rot_c2w, fx, fy, W, H = camera_terms(camera)
    dev = prep["position"].device
    Rw2c = rot_c2w.T.to(dev)
    t = -(Rw2c @ pos_c.to(dev))
    p = prep["position"] @ Rw2c.T + t
    z = p[:, 2]
    op = prep["opacity"]
    ok = (z > 0.2) & (255.0 * op >= 1.0)
    zc = z.clamp_min(0.2)
    tanx, tany = W / (2 * fx), H / (2 * fy)
    tx = (p[:, 0] / zc).clamp(-1.3 * tanx, 1.3 * tanx) * zc
    ty = (p[:, 1] / zc).clamp(-1.3 * tany, 1.3 * tany) * zc
    J = torch.zeros(len(p), 2, 3, device=dev)
    J[:, 0, 0], J[:, 0, 2] = fx / zc, -fx * tx / zc**2
    J[:, 1, 1], J[:, 1, 2] = fy / zc, -fy * ty / zc**2
    c = prep["cov3d"]
    S = torch.stack([c[:, 0], c[:, 1], c[:, 2], c[:, 1], c[:, 3], c[:, 4],
                     c[:, 2], c[:, 4], c[:, 5]], -1).reshape(-1, 3, 3)
    T = J @ Rw2c
    cov2 = T @ S @ T.transpose(1, 2)
    a, cc = cov2[:, 0, 0] + 0.3, cov2[:, 1, 1] + 0.3
    power = math.log(2) * 8 + torch.log(op.clamp_min(1e-12))
    wdt = (1.414214 * (a * power).clamp_min(0).sqrt() + 1).floor()
    hgt = (1.414214 * (cc * power).clamp_min(0).sqrt() + 1).floor()
    px = (fx * p[:, 0] / zc + W / 2) - 0.5
    py = (fy * p[:, 1] / zc + H / 2) - 0.5
    gx, gy = (W + tile - 1) // tile, (H + tile - 1) // tile
    x0 = ((px - wdt) / tile).trunc().clamp(0, gx)
    y0 = ((py - hgt) / tile).trunc().clamp(0, gy)
    x1 = (((px + wdt) / tile).trunc() + 1).clamp(0, gx)
    y1 = (((py + hgt) / tile).trunc() + 1).clamp(0, gy)
    n = ((x1 - x0) * (y1 - y0)).where(ok, torch.zeros_like(x0))
    return n


# --------------------------------------------------------------------------- FlashGS
class FlashGSRasterizer:
    """Mirror of upstream example.py's `Rasterizer` with buffers that grow on demand.

    Before every frame the key/value capacity is checked against the coarse-rectangle bound
    (the kernel writes without bounds checks). The colour buffer has one spare tile row
    because the kernel's empty-tile branch writes the full 16x16 tile without an image-bounds
    check."""

    def __init__(self, device="cuda", tile=TILE):
        from . import kernels

        if tile != 16:
            raise ValueError("only the 16x16 kernel used by upstream example.py is wired")
        self.ext = kernels.flashgs()
        self.device = torch.device(device)
        self.tile = tile
        self.capacity = 0
        self.points = 0
        self.tiles = 0
        self.last = {}

    def _ensure(self, num_points, num_pairs, num_tiles):
        dev = self.device
        if num_pairs > self.capacity:
            cap = int(max(num_pairs * 1.25, 1 << 20))
            self.keys_unsorted = torch.zeros(cap, device=dev, dtype=torch.int64)
            self.values_unsorted = torch.zeros(cap, device=dev, dtype=torch.int32)
            self.keys_sorted = torch.zeros(cap, device=dev, dtype=torch.int64)
            self.values_sorted = torch.zeros(cap, device=dev, dtype=torch.int32)
            size = self.ext.ops.get_sort_buffer_size(cap)
            self.sort_space = torch.zeros(size, device=dev, dtype=torch.int8)
            self.capacity = cap
        if num_points > self.points:
            self.points_xy = torch.zeros((num_points, 2), device=dev, dtype=torch.float32)
            self.rgb_depth = torch.zeros((num_points, 4), device=dev, dtype=torch.float32)
            self.conic_opacity = torch.zeros((num_points, 4), device=dev, dtype=torch.float32)
            self.points = num_points
        if num_tiles > self.tiles:
            self.ranges = torch.zeros((num_tiles, 2), device=dev, dtype=torch.int32)
            self.tiles = num_tiles
        if not hasattr(self, "curr_offset"):
            self.curr_offset = torch.zeros(1, device=dev, dtype=torch.int32)

    def reserve(self, prep, camera):
        W, H = int(camera["width"]), int(camera["height"])
        gx, gy = (W + self.tile - 1) // self.tile, (H + self.tile - 1) // self.tile
        bound = int(bbox_tile_counts(prep, camera, self.tile).sum().item())
        # +1% / +4096: float differences between this replica and the kernel at rect borders.
        self._ensure(len(prep["opacity"]), int(bound * 1.01) + 4096, gx * gy)
        return bound

    def forward(self, prep, camera, background, reserve=True):
        """uint8 [H,W,3] image exactly as the FlashGS kernel writes it."""
        if reserve:
            self.reserve(prep, camera)
        pos, rot, fx, fy, W, H = camera_terms(camera)
        ops = self.ext.ops
        t = self.tile
        gx, gy = (W + t - 1) // t, (H + t - 1) // t
        self.curr_offset.fill_(0)
        ops.preprocess(prep["position"], prep["shs"], prep["opacity"], prep["cov3d"],
                       W, H, t, t, pos, rot, fx, fy, 100.0, 0.01,
                       self.points_xy, self.rgb_depth, self.conic_opacity,
                       self.keys_unsorted, self.values_unsorted, self.curr_offset)
        num_rendered = int(self.curr_offset.cpu()[0])
        if num_rendered >= self.capacity:
            raise RuntimeError(f"FlashGS wrote {num_rendered} pairs into {self.capacity} slots")
        ops.sort_gaussian(num_rendered, W, H, t, t, self.sort_space,
                          self.keys_unsorted, self.values_unsorted,
                          self.keys_sorted, self.values_sorted)
        out = torch.zeros(((gy + 1) * t) * W * 3, device=self.device, dtype=torch.uint8)
        bg = background.detach().float().cpu().contiguous()
        if num_rendered > 0:
            ops.render_16x16(num_rendered, W, H, self.points_xy, self.rgb_depth,
                             self.conic_opacity, self.keys_sorted, self.values_sorted,
                             self.ranges, bg, out)
        else:  # nothing visible: the kernel is not launched on an empty list
            out.view(-1, 3)[:] = quantize_trunc(bg).to(self.device)
        self.last = {"num_rendered": num_rendered, "grid": (gx, gy)}
        return out[: H * W * 3].view(H, W, 3)

    def edge_wrap_tiles(self, width):
        """Right-column tiles that were empty in the last frame. For W % 16 != 0 their
        unbounded writes land on the first pixels of the next row (an upstream bug)."""
        gx, gy = self.last["grid"]
        if width % self.tile == 0 or self.last["num_rendered"] == 0:
            return 0
        r = self.ranges[: gx * gy].view(gy, gx, 2)[:, -1]
        return int((r[:, 0] == r[:, 1]).sum().item())

    def pairs(self):
        """(tile_id, gaussian_id) of the last frame's sorted key/value pairs."""
        n = self.last["num_rendered"]
        return (self.keys_sorted[:n] >> 32).long(), self.values_sorted[:n].long()


# --------------------------------------------------------------------------- references
def projection_3dgs(W, H, fx, fy, znear=0.01, zfar=100.0):
    """3DGS utils/graphics_utils.getProjectionMatrix (centred principal point)."""
    tanx, tany = W / (2 * fx), H / (2 * fy)
    top, right = tany * znear, tanx * znear
    P = torch.zeros(4, 4)
    P[0, 0] = 2 * znear / (2 * right)
    P[1, 1] = 2 * znear / (2 * top)
    P[3, 2] = 1.0
    P[2, 2] = zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P


class Reference3DGS:
    """Original 3DGS CUDA forward (graphdeco diff-gaussian-rasterization, unmodified)."""

    def __init__(self):
        from . import kernels

        self.ext = kernels.ref3dgs()
        self.last = {}

    def inputs(self, params, sh_degree=3):
        with torch.no_grad():
            return {
                "means": params["means"].detach().float().contiguous(),
                "opacity": params["opacities"].detach().float().sigmoid().reshape(-1, 1),
                "scales": params["scales"].detach().float().exp().contiguous(),
                "rotations": (params["quats"] / params["quats"].norm(dim=-1, keepdim=True))
                .detach().float().contiguous(),
                "sh": torch.cat((params["sh0"], params["shN"]), 1).detach().float().contiguous(),
                "degree": sh_degree,
            }

    def forward(self, inp, camera, background):
        pos, _rot, fx, fy, W, H = camera_terms(camera)
        dev = inp["means"].device
        Vt = camera["viewmat"].detach().float().to(dev).T.contiguous()
        full = (Vt @ projection_3dgs(W, H, fx, fy).to(dev).T).contiguous()
        empty = torch.empty(0, device=dev)
        num, color, *_ = self.ext.rasterize_gaussians(
            background.float().to(dev), inp["means"], empty, inp["opacity"], inp["scales"],
            inp["rotations"], 1.0, empty, Vt, full, W / (2 * fx), H / (2 * fy), H, W,
            inp["sh"], inp["degree"], pos.to(dev), False, False)
        self.last = {"num_rendered": int(num)}
        return color.permute(1, 2, 0)


def gsplat_forward(params, camera, sh_degree=3, white=True):
    """gsplat rasterization forward with the SplatStudio renderer settings (RGB only)."""
    from gsplat import rasterization

    colors = torch.cat((params["sh0"], params["shN"]), dim=1)
    render, alpha, info = rasterization(
        means=params["means"], quats=params["quats"], scales=params["scales"].exp(),
        opacities=params["opacities"].sigmoid(), colors=colors,
        viewmats=camera["viewmat"][None], Ks=camera["K"][None],
        width=int(camera["width"]), height=int(camera["height"]),
        sh_degree=sh_degree, packed=False, render_mode="RGB",
    )
    rgb = render[0] + (1 - alpha[0]) if white else render[0]
    return rgb, info


def quantize_trunc(rgb):
    """FlashGS output encoding (render.cu `encode`): uint8(clamp(x*255, 0, 255)), truncating."""
    return (rgb * 255.0).clamp(0, 255).to(torch.uint8)


def psnr(a, b):
    mse = (a.float() - b.float()).square().mean().clamp_min(1e-12)
    return float(-10 * torch.log10(mse))
