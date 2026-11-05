"""AGS-Mesh (Splatfacto variant) renderer = DN-Splatter renderer (dn_model.py:get_outputs, official
commit 97588b4; copied from reproductions/dn-splatter ss_dnsplatter.renderer) plus the surface
normals of the rendered depth used by AGS-Mesh's ANR term (paper Eq. 6, dn_model.py:582-604):

* ``surface_normal`` (GL camera frame, [-1, 1], zero 1-px border): back-project the rendered depth
  with pixel centres +0.5 (camera_utils.get_means3d_backproj, c2w = I), central-difference cross
  product (normal_utils.pcd_to_normal), then diag(1, -1, -1). The official code back-projects
  ``depth_im.detach()``, so ANR carries no gradient; ``surface_grad: true`` keeps the graph
  (the paper's 2DGS-style formulation) and is not the default.

* RGB + expected depth in one gsplat pass (``RGB+ED``, classic, absgrad, near 0.01, tile 16).
* Random background while training, the viser grey (0.1490, 0.1647, 0.2157) at evaluation
  (populate_modules with background_color="random"); rgb clamped to [0, 1].
* Depth: pixels with zero accumulation get the detached maximum depth.
* Normals (paper Eq. 4, 7): n_i = R(q_i) onehot(argmin s_i), flipped to face the camera every
  step, rotated into the camera frame and alpha-composited in a second pass that uses the
  *detached* 2D means but differentiable conics/opacities, like the official legacy
  ``rasterize_gaussians(xys.detach(), ...)`` call. The legacy rasterizer composites over a white
  background (gsplat 1.0 cuda_legacy/_wrapper.py:339), then the map is normalised per pixel and
  encoded as (n + 1) / 2.

Frame convention: official normals live in the Nerfstudio (OpenGL) camera frame and are compared
with Omnidata pngs converted by diag(1, -1, -1). Elementwise |D a - D b| = |a - b| and the TV
term is sign invariant, so rendering in the OpenCV frame (R_w2c n) and comparing with the raw
png values gives exactly the official loss values and gradients.
"""

import torch
import torch.nn.functional as F
from gsplat import rasterization, rasterize_to_pixels
from gsplat.utils import normalized_quat_to_rotmat

from . import sidecar

CAPABILITIES = {
    "render.rgb",
    "render.depth.expected_z",
    "render.normal.camera",
    "gradient.screen.absolute",
    "prior.sensor_depth",
    "prior.mono_normal",
    "render.normal.surface",
    "prior.depth_confidence",
}
DEFAULTS = {
    "random_background": True,
    "eval_background": [0.1490, 0.1647, 0.2157],
    "near_plane": 0.01,
    "normal_eps": 1e-12,
    "surface_grad": False,
}

GL = (1.0, -1.0, -1.0)  # diag OpenCV <-> Nerfstudio/OpenGL camera frame


def gaussian_normals(params, viewmat):
    """World normals of every Gaussian, flipped towards the camera centre (dn_model.py:476-492)."""
    quats = F.normalize(params["quats"], dim=-1)
    onehot = F.one_hot(torch.argmin(params["scales"], dim=-1), num_classes=3).to(quats.dtype)
    rots = normalized_quat_to_rotmat(quats)
    normals = F.normalize(torch.bmm(rots, onehot[:, :, None]).squeeze(-1), dim=1)
    center = torch.linalg.inv(viewmat.detach())[:3, 3]
    viewdirs = F.normalize(center[None] - params["means"].detach(), dim=-1)
    flip = (normals * viewdirs).sum(-1) < 0
    return torch.where(flip[:, None], -normals, normals)


def surface_normals(depth, K):
    """normal_from_depth_image(depth, fx, fy, cx, cy, c2w=I) @ diag(1,-1,-1): [H,W,1] -> [H,W,3]."""
    H, W = depth.shape[:2]
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    v, u = torch.meshgrid(torch.arange(H, device=depth.device, dtype=depth.dtype),
                          torch.arange(W, device=depth.device, dtype=depth.dtype), indexing="ij")
    d = depth[..., 0]
    xyz = torch.stack(((u + 0.5 - cx) * d / fx, (v + 0.5 - cy) * d / fy, d), -1)
    left_to_right = xyz[1:-1, 2:] - xyz[1:-1, :-2]
    bottom_to_top = xyz[:-2, 1:-1] - xyz[2:, 1:-1]
    n = F.normalize(torch.cross(left_to_right, bottom_to_top, dim=-1), p=2, dim=-1)
    n = F.pad(n.permute(2, 0, 1), (1, 1, 1, 1), mode="constant").permute(1, 2, 0)
    return n * torch.tensor(GL, device=n.device, dtype=n.dtype)


class Renderer:
    def __init__(self, spec):
        unknown = set(spec.config) - set(DEFAULTS)
        if unknown:
            raise ValueError(f"Unknown AGS-Mesh renderer options: {sorted(unknown)}")
        self.cfg = {**DEFAULTS, **spec.config}
        self.capabilities = set(CAPABILITIES)
        self._table = False

    def table(self):
        if self._table is False:
            self._table = sidecar.table()
        return self._table

    def render(self, params, camera, sh_degree=3):
        dev = params["means"].device
        training = torch.is_grad_enabled() and params["means"].requires_grad
        viewmat, K = camera["viewmat"], camera["K"]
        W, H = int(camera["width"]), int(camera["height"])
        colors = torch.cat((params["sh0"], params["shN"]), dim=1)
        render, alpha, info = rasterization(
            means=params["means"],
            quats=params["quats"] / params["quats"].norm(dim=-1, keepdim=True),
            scales=params["scales"].exp(),
            opacities=params["opacities"].sigmoid(),
            colors=colors,
            viewmats=viewmat[None],
            Ks=K[None],
            width=W,
            height=H,
            tile_size=16,
            packed=False,
            near_plane=float(self.cfg["near_plane"]),
            far_plane=1e10,
            render_mode="RGB+ED",
            sh_degree=sh_degree,
            sparse_grad=False,
            absgrad=True,
            rasterize_mode="classic",
        )
        if training and self.cfg["random_background"]:
            bg = torch.rand(3, device=dev)
        else:
            bg = torch.tensor(self.cfg["eval_background"], dtype=torch.float32, device=dev)
        alpha = alpha[0]
        rgb = torch.clamp(render[0, ..., :3] + (1 - alpha) * bg, 0.0, 1.0)
        depth = render[0, ..., 3:4]
        depth = torch.where(alpha > 0, depth, depth.detach().max())
        out = {"rgb": rgb, "depth": depth, "alpha": alpha, "info": info, "camera": camera}
        if not training:
            return out
        normals = gaussian_normals(params, viewmat)
        normals_cam = normals @ viewmat[:3, :3].T  # R_w2c n (OpenCV camera frame)
        nimg, nalpha = rasterize_to_pixels(
            info["means2d"].detach(),
            info["conics"],
            normals_cam[None],
            info["opacities"],
            W,
            H,
            info["tile_size"],
            info["isect_offsets"],
            info["flatten_ids"],
        )
        # official white normal background in the OpenGL frame = (1, -1, -1) in OpenCV (DN-Splatter review fix)
        bg_n = torch.tensor((1.0, -1.0, -1.0), dtype=nimg.dtype, device=nimg.device)
        nimg = nimg[0] + (1 - nalpha[0]) * bg_n
        nimg = nimg / nimg.norm(dim=-1, keepdim=True).clamp_min(float(self.cfg["normal_eps"]))
        out["normal"] = (nimg + 1) / 2
        d_in = depth if self.cfg["surface_grad"] else depth.detach()
        out["surface_normal"] = surface_normals(d_in, K)
        tab = self.table()
        idx = tab.index(viewmat) if tab is not None else None
        out["dn_view"] = tab.view(idx, dev) if idx is not None else None
        return out


def create_renderer(spec):
    return Renderer(spec)
