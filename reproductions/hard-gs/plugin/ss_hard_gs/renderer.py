"""HGS renderer: Scaffold-GS anchors -> view-dependent neural Gaussians -> gsplat.

Native Gaussian population = Scaffold-GS *anchors*:
  means        [N,3]     anchor position (lr 0)
  hgs_feat     [N,32]    anchor feature
  hgs_offset   [N,10,3]  offsets (in units of exp(scaling[:, :3]))
  hgs_scaling  [N,6]     log scaling
  hgs_blob     [N,8]     the three decoder MLPs, repacked after every training step (ss-eval
                         transport; pattern of reproductions/hac/plugin/ss_hac/renderer.py)
quats / opacities / sh0 / shN are carried but unused (no gradient).

Decode follows Scaffold-GS gaussian_renderer/__init__.py:generate_neural_gaussians with
appearance_dim 0 and add_*_dist False. The anchor prefilter is the Scaffold fork's
visible_filter (re-implemented in the HAC plugin; copied here).
"""

import torch
import torch.nn.functional as F
from gsplat import rasterization

from . import model

BLOB = "hgs_blob"
BLOB_WIDTH = 8
MAGIC = 4343.0
BLOB_HEADER = 8


# ------------------------------------------------------------------------- blob packing
def mlp_layout(feat_dim, n_offsets):
    ref = model.scaffold_mlps(feat_dim, n_offsets)
    return [(f"{k}.{n}", tuple(t.shape)) for k in model.RENDER_MLPS
            for n, t in ref[k].state_dict().items()]


def blob_floats(feat_dim, n_offsets):
    return BLOB_HEADER + sum(int(torch.Size(s).numel()) for _, s in mlp_layout(feat_dim, n_offsets))


@torch.no_grad()
def pack_blob(mlps, n_rows, feat_dim, n_offsets, device):
    """Flatten the three MLPs into an [n_rows, BLOB_WIDTH] tensor (header + weights)."""
    state = {f"{k}.{n}": t for k in model.RENDER_MLPS for n, t in mlps[k].state_dict().items()}
    body = torch.cat([state[name].detach().float().reshape(-1).to(device)
                      for name, _ in mlp_layout(feat_dim, n_offsets)])
    head = torch.tensor([MAGIC, 1.0, BLOB_HEADER + body.numel(), feat_dim, n_offsets, 0, 0, 0],
                        device=device)
    head[5] = body.double().sum().float()
    flat = torch.cat([head, body])
    if flat.numel() > n_rows * BLOB_WIDTH:
        raise ValueError(f"{n_rows} anchors cannot hold {flat.numel()} decoder floats")
    out = torch.zeros(n_rows * BLOB_WIDTH, device=device)
    out[: flat.numel()] = flat
    return out.view(n_rows, BLOB_WIDTH)


def unpack_blob(blob, feat_dim, n_offsets):
    flat = blob.reshape(-1)
    if flat.numel() < BLOB_HEADER or float(flat[0]) != MAGIC:
        raise ValueError("No Scaffold-GS decoder weights in hgs_blob (state was never trained)")
    if int(flat[3]) != feat_dim or int(flat[4]) != n_offsets:
        raise ValueError("hgs_blob was packed for a different feat_dim / n_offsets")
    mlps = model.scaffold_mlps(feat_dim, n_offsets).to(blob.device)
    state, i = {}, BLOB_HEADER
    for name, shape in mlp_layout(feat_dim, n_offsets):
        n = int(torch.Size(shape).numel())
        state[name] = flat[i : i + n].view(shape).clone()
        i += n
    if i != int(flat[2]):
        raise ValueError("hgs_blob length disagrees with the decoder layout")
    if not torch.isclose(flat[BLOB_HEADER:i].double().sum().float(), flat[5], rtol=1e-4,
                         atol=1e-3):
        raise ValueError("hgs_blob checksum mismatch (anchor rows reordered or corrupted)")
    for k in model.RENDER_MLPS:
        mlps[k].load_state_dict({n.split(".", 1)[1]: v for n, v in state.items()
                                 if n.split(".", 1)[0] == k})
    return mlps.eval()


# --------------------------------------------------------------------- neural Gaussians
def camera_center(viewmat):
    return -viewmat[:3, :3].T @ viewmat[:3, 3]


def visible_anchors(anchors, scales, camera, near=0.2, block=16):
    """prefilter_voxel -> Scaffold fork's visible_filter (filter_preprocessCUDA): keep an anchor
    iff z_view > 0.2 and its own Gaussian (scales = exp(scaling[:, :3]), identity rotation) has a
    non-empty 16x16-tile rectangle (EWA covariance, 1.3 tan(fov) clamp, +0.3 low-pass, 3 sigma)."""
    viewmat, K = camera["viewmat"], camera["K"]
    W, H = int(camera["width"]), int(camera["height"])
    R, t = viewmat[:3, :3], viewmat[:3, 3]
    p = anchors @ R.T + t
    z = p[:, 2]
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    tan_x, tan_y = W / (2 * fx), H / (2 * fy)
    zs = torch.where(z > near, z, torch.ones_like(z))
    tx = torch.clamp(p[:, 0] / zs, -1.3 * tan_x, 1.3 * tan_x) * zs
    ty = torch.clamp(p[:, 1] / zs, -1.3 * tan_y, 1.3 * tan_y) * zs
    n = len(anchors)
    J = torch.zeros(n, 2, 3, device=anchors.device, dtype=anchors.dtype)
    J[:, 0, 0] = fx / zs
    J[:, 0, 2] = -fx * tx / zs**2
    J[:, 1, 1] = fy / zs
    J[:, 1, 2] = -fy * ty / zs**2
    T = J @ R
    cov = (T * scales.square()[:, None, :]) @ T.transpose(1, 2)
    a = cov[:, 0, 0] + 0.3
    b = cov[:, 0, 1]
    c = cov[:, 1, 1] + 0.3
    det = a * c - b * b
    mid = 0.5 * (a + c)
    disc = torch.sqrt(torch.clamp(mid * mid - det, min=0.1))
    radius = torch.ceil(3.0 * torch.sqrt(torch.maximum(mid + disc, mid - disc)))
    u = fx * p[:, 0] / zs + cx - 0.5
    v = fy * p[:, 1] / zs + cy - 0.5
    gx, gy = (W + block - 1) // block, (H + block - 1) // block

    def tile(val, hi):
        return torch.clamp(torch.trunc(val / block), min=0, max=hi)

    xmin, xmax = tile(u - radius, gx), tile(u + radius + block - 1, gx)
    ymin, ymax = tile(v - radius, gy), tile(v + radius + block - 1, gy)
    return (z > near) & (det != 0) & ((xmax - xmin) * (ymax - ymin) != 0)


def neural_gaussians(mlps, anchor, feat, offsets, scaling, center, n_offsets):
    """generate_neural_gaussians (appearance_dim 0, no distance inputs, no feature bank)."""
    ob_view = anchor - center
    ob_view = ob_view / ob_view.norm(dim=1, keepdim=True)
    x = torch.cat([feat, ob_view], dim=1)
    neural_opacity = mlps["mlp_opacity"](x).reshape(-1, 1)
    sel = (neural_opacity > 0.0).view(-1)
    color = mlps["mlp_color"](x).reshape(-1, 3)
    scale_rot = mlps["mlp_cov"](x).reshape(-1, 7)
    rep = torch.cat([scaling, anchor], dim=-1).repeat_interleave(n_offsets, dim=0)
    allv = torch.cat([rep, color, scale_rot, offsets.reshape(-1, 3)], dim=-1)[sel]
    scaling_rep, anchor_rep, color, scale_rot, off = allv.split([6, 3, 3, 7, 3], dim=-1)
    scales = scaling_rep[:, 3:] * torch.sigmoid(scale_rot[:, :3])
    quats = F.normalize(scale_rot[:, 3:7], dim=-1)
    xyz = anchor_rep + off * scaling_rep[:, :3]
    return xyz, color, neural_opacity[sel][:, 0], scales, quats, neural_opacity, sel


class HGSRenderer:
    capabilities = {"render.rgb", "render.depth.expected_z", "gradient.screen.signed"}

    def __init__(self, spec):
        c = spec.config
        self.feat_dim = int(c.get("feat_dim", 32))
        self.n_offsets = int(c.get("n_offsets", 10))
        self.bank = None  # explicit bank (probes); otherwise model.ACTIVE
        self._eval_mlps = None
        self._eval_key = None

    def mlps(self, params):
        bank = self.bank if self.bank is not None else model.ACTIVE["bank"]
        if bank is not None:
            return bank.mlps
        blob = params[BLOB]
        key = (blob.data_ptr(), tuple(blob.shape), float(blob.reshape(-1)[5]))
        if self._eval_key != key:
            self._eval_mlps = unpack_blob(blob.detach(), self.feat_dim, self.n_offsets)
            self._eval_key = key
        return self._eval_mlps

    def render(self, params, camera, sh_degree=None):
        mlps = self.mlps(params)
        anchors = params["means"]
        with torch.no_grad():
            vis = visible_anchors(anchors, params["hgs_scaling"][:, :3].exp(), camera)
        g = neural_gaussians(mlps, anchors[vis], params["hgs_feat"][vis],
                             params["hgs_offset"][vis], params["hgs_scaling"][vis].exp(),
                             camera_center(camera["viewmat"]), self.n_offsets)
        xyz, color, opacity, scales, quats = g[:5]
        render, alpha, info = rasterization(
            means=xyz, quats=quats, scales=scales, opacities=opacity, colors=color,
            viewmats=camera["viewmat"][None], Ks=camera["K"][None], width=camera["width"],
            height=camera["height"], near_plane=0.2, sh_degree=None, packed=False,
            render_mode="RGB+ED",
        )
        info["hgs"] = {"visible": vis, "selection": g[6], "neural_opacity": g[5],
                       "scaling": scales}
        # Runtime convention: white background (official Scaffold-GS: black).
        return {"rgb": render[0, ..., :3] + (1 - alpha[0]), "depth": render[0, ..., 3:4],
                "alpha": alpha[0], "info": info}


def create_renderer(spec):
    return HGSRenderer(spec)
