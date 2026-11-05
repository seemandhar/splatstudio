"""Octree-GS renderer: LOD anchor selection -> Scaffold visible_filter -> neural Gaussians (view dependent MLPs)
-> gsplat.

Native population = anchor population:
  means            [N,3]     anchor position (lr 0)
  ogs_feat         [N,32]    anchor feature
  ogs_offset       [N,10,3]  offsets (units of scaling[:, :3])
  ogs_scaling      [N,6]     log scaling
  ogs_level        [N,1]     octree level L_j (no gradient)
  ogs_extra_level  [N,1]     LOD bias Delta L_j (no gradient; updated by the density)
  ogs_blob         [N,8]     MLP weights + octree scalars, re-packed by the density after every step
The base fields quats / opacities / scales / sh0 / shN are unused (no gradient; Adam skips them).

Training renders (grad enabled, live bank) follow train.py: set_anchor_mask(iteration) with the progressive level
limit, prefilter_voxel, generate_neural_gaussians, render. Renders without gradient (validation, ss-eval in a fresh
process, probes) decode MLPs and scalars from `ogs_blob` and use all K levels (the official eval mode).

Blob packing, the Scaffold `visible_filter` port, the neural-Gaussian decode and the rasterisation call are adapted
from the HAC / ContextGS native ports (reproductions/hac/plugin/ss_hac/renderer.py,
reproductions/contextgs/plugin/ss_contextgs/renderer.py); Octree-GS's decode input has no distance term.
"""

import torch
import torch.nn.functional as F
from gsplat import rasterization

from . import model, octree

BLOB = "ogs_blob"
BLOB_WIDTH = 8
MAGIC = 7575.0
BLOB_HEADER = 16
# header: 0 magic, 1 version, 2 length, 3 feat_dim, 4 n_offsets, 5 checksum, 6 step, 7 levels, 8 fork,
#         9 voxel_size, 10 standard_dist, 11 init_pos, 12 init_level


# ------------------------------------------------------------------------- blob packing
def mlp_layout(feat_dim, n_offsets):
    ref = model.scaffold_mlps(feat_dim, n_offsets)
    return [(f"{k}.{n}", tuple(t.shape)) for k in model.RENDER_MLPS for n, t in ref[k].state_dict().items()]


def blob_floats(feat_dim, n_offsets):
    return BLOB_HEADER + sum(int(torch.Size(s).numel()) for _, s in mlp_layout(feat_dim, n_offsets))


@torch.no_grad()
def pack_blob_flat(mlps, oct_, feat_dim, n_offsets, step, device):
    head = torch.zeros(BLOB_HEADER, device=device)
    head[:13] = torch.tensor([MAGIC, 1.0, 0.0, feat_dim, n_offsets, 0.0, float(step), oct_["levels"],
                              oct_["fork"], oct_["voxel_size"], oct_["standard_dist"], oct_["init_pos"],
                              oct_["init_level"]], device=device)
    state = {f"{k}.{n}": t for k in model.RENDER_MLPS for n, t in mlps[k].state_dict().items()}
    flat = [head] + [state[name].detach().float().reshape(-1).to(device)
                     for name, _ in mlp_layout(feat_dim, n_offsets)]
    flat = torch.cat(flat)
    flat[2] = float(flat.numel())
    flat[5] = flat[BLOB_HEADER:].double().sum().float() + flat[6:13].double().sum().float()
    return flat


@torch.no_grad()
def write_blob(blob, flat):
    """In-place write (keeps the Parameter object and its optimizer entry)."""
    n = flat.numel()
    if n > blob.numel():
        raise ValueError(f"{blob.shape[0]} anchors cannot hold the {n}-float Octree-GS blob")
    v = blob.data.view(-1)
    v[:n] = flat
    v[n:] = 0


def unpack_blob(blob, feat_dim, n_offsets):
    flat = blob.reshape(-1)
    if flat.numel() < BLOB_HEADER or float(flat[0]) != MAGIC:
        raise ValueError("No Octree-GS decoder weights in the state (ogs_blob is empty): the model was never "
                         "packed by the ogs-density component")
    if int(flat[3]) != feat_dim or int(flat[4]) != n_offsets:
        raise ValueError("ogs_blob was packed for a different feat_dim / n_offsets")
    mlps = model.scaffold_mlps(feat_dim, n_offsets).to(blob.device)
    state, i = {}, BLOB_HEADER
    for name, shape in mlp_layout(feat_dim, n_offsets):
        n = int(torch.Size(shape).numel())
        state[name] = flat[i : i + n].view(shape).clone()
        i += n
    if i != int(flat[2]) or int(flat[2]) > flat.numel():
        raise ValueError("ogs_blob length disagrees with the decoder layout")
    check = flat[BLOB_HEADER:i].double().sum().float() + flat[6:13].double().sum().float()
    if not torch.isclose(check, flat[5], rtol=1e-4, atol=1e-3):
        raise ValueError("ogs_blob checksum mismatch (anchor rows reordered or corrupted)")
    for k in model.RENDER_MLPS:
        mlps[k].load_state_dict({n.split(".", 1)[1]: v for n, v in state.items() if n.split(".", 1)[0] == k})
    h = flat[:13].tolist()
    oct_ = {"levels": int(h[7]), "fork": int(h[8]), "voxel_size": flat[9].clone(),
            "standard_dist": float(h[10]), "init_pos": float(h[11]), "init_level": int(h[12])}
    return mlps.eval(), oct_


# --------------------------------------------------------------------- neural Gaussians
def camera_center(viewmat):
    return -viewmat[:3, :3].T @ viewmat[:3, 3]


def visible_anchors(anchors, scales, camera, near=0.2, block=16):
    """prefilter_voxel -> Scaffold fork's rasterizer.visible_filter (filter_preprocessCUDA): keep an anchor iff
    z_view > 0.2 and its own Gaussian (scales = scaling[:, :3], identity rotation, scale_modifier 1) has a
    non-empty 16x16-tile rectangle: EWA 2D covariance with the 1.3 tan(fov) clamp and +0.3 low-pass, det != 0,
    radius ceil(3 sqrt(lambda_max)), getRect clamped to the tile grid. (Copied from the ContextGS port.)"""
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
    """generate_neural_gaussians (script settings): input [f, view dir]; opacity Tanh kept where > 0; colour
    Sigmoid; scale = scaling[:,3:] * sigmoid(cov[:3]); rot = normalize(cov[3:7]); xyz = anchor + offset *
    scaling[:,:3]. Returns (xyz, color, opacity, scales, quats, neural_opacity [M*k,1], selection mask)."""
    ob_view = anchor - center
    ob_dist = ob_view.norm(dim=1, keepdim=True)
    ob_view = ob_view / ob_dist
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


def lod_mask(params, oct_, center, max_level):
    return octree.anchor_mask(params["means"], params["ogs_level"], params["ogs_extra_level"], center,
                              oct_["voxel_size"], oct_["standard_dist"], oct_["fork"], max_level)


class OctreeGSRenderer:
    capabilities = {"render.rgb", "render.depth.expected_z", "gradient.screen.signed"}

    def __init__(self, spec):
        c = spec.config
        self.feat_dim = int(c.get("feat_dim", 32))
        self.n_offsets = int(c.get("n_offsets", 10))
        self.opacity_cap = c.get("opacity_cap", 0.999)
        self.bank = None  # explicit bank (probes); otherwise model.ACTIVE
        self._eval = None
        self._eval_key = None

    def active_bank(self):
        bank = self.bank if self.bank is not None else model.ACTIVE["bank"]
        return bank if bank is not None and bank.ready else None

    def decoded(self, params):
        blob = params[BLOB]
        key = (blob.data_ptr(), tuple(blob.shape), float(blob.reshape(-1)[5]), float(blob.reshape(-1)[6]))
        if self._eval_key != key:
            self._eval = unpack_blob(blob.detach(), self.feat_dim, self.n_offsets)
            self._eval_key = key
        return self._eval

    def render(self, params, camera, sh_degree=None):
        bank = self.active_bank()
        if bank is not None and torch.is_grad_enabled():
            mlps, oct_ = bank.mlps, bank.octree()
            max_level = octree.coarse_index(bank.coarse_intervals, bank.step, oct_["init_level"],
                                            oct_["levels"], bank.c["progressive"], training=True) - 1
            training = True
        else:
            mlps, oct_ = self.decoded(params)
            max_level = oct_["levels"] - 1
            training = False
        center = camera_center(camera["viewmat"])
        with torch.no_grad():
            lod = lod_mask(params, oct_, center, max_level)
            vis = lod.clone()
            idx = torch.where(lod)[0]
            vis[idx] = visible_anchors(params["means"][idx], params["ogs_scaling"][idx, :3].exp(), camera)
        g = neural_gaussians(mlps, params["means"][vis], params["ogs_feat"][vis], params["ogs_offset"][vis],
                             params["ogs_scaling"][vis].exp(), center, self.n_offsets)
        out = self._rasterize(g[:5], camera)
        out["info"]["ogs"] = {"visible": vis, "lod": lod, "selection": g[6], "neural_opacity": g[5],
                              "scaling": g[3], "max_level": max_level, "training": training}
        return out

    def _rasterize(self, gaussians, camera):
        xyz, color, opacity, scales, quats = gaussians
        if self.opacity_cap is not None:
            # gsplat's backward skips Gaussians with opacity*G > 0.999 (TODO-runtime.md); the Inria fork clamps alpha
            # to 0.99 in the forward pass and always back-propagates. Straight-through cap keeps the gradient.
            cap = float(self.opacity_cap)
            opacity = opacity - (opacity - cap).clamp(min=0).detach()
        render, alpha, info = rasterization(
            means=xyz, quats=quats, scales=scales, opacities=opacity, colors=color,
            viewmats=camera["viewmat"][None], Ks=camera["K"][None], width=camera["width"],
            height=camera["height"], near_plane=0.2, sh_degree=None, packed=False, render_mode="RGB+ED",
        )
        # Runtime convention: RGB over a white background (official Octree-GS: black).
        return {"rgb": render[0, ..., :3] + (1 - alpha[0]), "depth": render[0, ..., 3:4], "alpha": alpha[0],
                "info": info}


def create_renderer(spec):
    return OctreeGSRenderer(spec)

