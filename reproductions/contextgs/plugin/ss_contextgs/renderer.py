"""ContextGS renderer: anchors -> neural Gaussians (Scaffold-GS MLPs, view dependent) -> gsplat.

The native Gaussian population is the *anchor* population:
  means        [N,3]     anchor position (lr 0, quantized to 16 bit inside the bounds)
  cgs_feat     [N,50]    anchor feature f
  cgs_offset   [N,10,3]  offsets O (in units of scaling[:, :3])
  cgs_scaling  [N,6]     log scaling (exp in training; decoded linear values after baking)
  cgs_mask     [N,10,1]  offset-mask logits (sigmoid>0.01 STE; binary after baking)
  cgs_hyper    [N,12]    hyperprior latent z (Eq. 9)
  cgs_blob     [N,8]     render-MLP weights packed at the end of training (see `pack_blob`)
The base fields quats / opacities / sh0 / shN are unused (no gradient; Adam skips them).

Training renders use the live ContextBank (model.ACTIVE) and follow gaussian_renderer/__init__.py:
generate_neural_gaussians of the official code (fixed-Q noise in (3000, 10000], bound update at 10000, anchor-level
context + hyperprior on *all* anchors for step > 10000, then the visible anchors are decoded). After the final-step
bake, and in a fresh process (ss-eval), the renderer decodes the MLP weights from `cgs_blob`.

Blob packing, the Scaffold `visible_filter` port, the neural-Gaussian decode and the rasterization are copied from
the HAC native port (reproductions/hac/plugin/ss_hac/renderer.py); ContextGS uses the same Scaffold-GS backbone.
"""

import torch
import torch.nn.functional as F
from gsplat import rasterization

from . import levels, model

BLOB = "cgs_blob"
BLOB_WIDTH = 8
MAGIC = 4242.0
BLOB_HEADER = 8

FIELDS = ("cgs_feat", "cgs_offset", "cgs_scaling", "cgs_mask", "cgs_hyper", BLOB)


# ------------------------------------------------------------------------- blob packing
def mlp_layout(feat_dim, n_offsets):
    ref = model.scaffold_mlps(feat_dim, n_offsets)
    return [(f"{k}.{n}", tuple(t.shape)) for k in model.RENDER_MLPS
            for n, t in ref[k].state_dict().items()]


def blob_floats(feat_dim, n_offsets):
    return BLOB_HEADER + sum(int(torch.Size(s).numel()) for _, s in mlp_layout(feat_dim, n_offsets))


@torch.no_grad()
def pack_blob(mlps, n_rows, feat_dim, n_offsets, device):
    """Flatten the three render MLPs into an [n_rows, BLOB_WIDTH] tensor (header + weights)."""
    flat = [torch.tensor([MAGIC, 1.0, 0.0, feat_dim, n_offsets, 0, 0, 0], device=device)]
    state = {f"{k}.{n}": t for k in model.RENDER_MLPS for n, t in mlps[k].state_dict().items()}
    for name, shape in mlp_layout(feat_dim, n_offsets):
        flat.append(state[name].detach().float().reshape(-1).to(device))
    flat = torch.cat(flat)
    flat[2] = float(flat.numel())
    flat[5] = flat[BLOB_HEADER:].double().sum().float()  # checksum
    if flat.numel() > n_rows * BLOB_WIDTH:
        raise ValueError(f"{n_rows} anchors cannot hold {flat.numel()} decoder floats")
    out = torch.zeros(n_rows * BLOB_WIDTH, device=device)
    out[: flat.numel()] = flat
    return out.view(n_rows, BLOB_WIDTH)


def unpack_blob(blob, feat_dim, n_offsets):
    flat = blob.reshape(-1)
    if flat.numel() < BLOB_HEADER or float(flat[0]) != MAGIC:
        raise ValueError("No ContextGS decoder weights in the state (cgs_blob is empty): the model was "
                         "not baked at the end of training")
    if int(flat[3]) != feat_dim or int(flat[4]) != n_offsets:
        raise ValueError("cgs_blob was packed for a different feat_dim / n_offsets")
    mlps = model.scaffold_mlps(feat_dim, n_offsets).to(blob.device)
    state, i = {}, BLOB_HEADER
    for name, shape in mlp_layout(feat_dim, n_offsets):
        n = int(torch.Size(shape).numel())
        state[name] = flat[i : i + n].view(shape).clone()
        i += n
    if i != int(flat[2]) or int(flat[2]) > flat.numel():
        raise ValueError("cgs_blob length disagrees with the decoder layout")
    if not torch.isclose(flat[BLOB_HEADER:i].double().sum().float(), flat[5], rtol=1e-4,
                         atol=1e-3):
        raise ValueError("cgs_blob checksum mismatch (anchor rows reordered or corrupted)")
    for k in model.RENDER_MLPS:
        mlps[k].load_state_dict({n.split(".", 1)[1]: v for n, v in state.items()
                                 if n.split(".", 1)[0] == k})
    return mlps.eval()


# --------------------------------------------------------------------- neural Gaussians
def camera_center(viewmat):
    return -viewmat[:3, :3].T @ viewmat[:3, 3]


def visible_anchors(anchors, scales, camera, near=0.2, block=16):
    """prefilter_voxel -> Scaffold fork's rasterizer.visible_filter (filter_preprocessCUDA):
    keep an anchor iff z_view > 0.2 and its own Gaussian (scales = scaling[:, :3], identity
    rotation, scale_modifier 1) has a non-empty 16x16-tile rectangle: EWA 2D covariance with
    the 1.3 tan(fov) clamp and +0.3 low-pass, det != 0, radius ceil(3 sqrt(lambda_max)),
    getRect clamped to the tile grid."""
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
    T = J @ R  # [n,2,3]
    cov = (T * scales.square()[:, None, :]) @ T.transpose(1, 2)
    a = cov[:, 0, 0] + 0.3
    b = cov[:, 0, 1]
    c = cov[:, 1, 1] + 0.3
    det = a * c - b * b
    mid = 0.5 * (a + c)
    disc = torch.sqrt(torch.clamp(mid * mid - det, min=0.1))
    radius = torch.ceil(3.0 * torch.sqrt(torch.maximum(mid + disc, mid - disc)))
    # ndc2Pix(fx x/z ...): pixel = fx x/z + cx - 0.5 (3DGS pixel-centre convention)
    u = fx * p[:, 0] / zs + cx - 0.5
    v = fy * p[:, 1] / zs + cy - 0.5
    gx, gy = (W + block - 1) // block, (H + block - 1) // block

    def tile(val, hi):
        return torch.clamp(torch.trunc(val / block), min=0, max=hi)

    xmin, xmax = tile(u - radius, gx), tile(u + radius + block - 1, gx)
    ymin, ymax = tile(v - radius, gy), tile(v + radius + block - 1, gy)
    return (z > near) & (det != 0) & ((xmax - xmin) * (ymax - ymin) != 0)


def neural_gaussians(mlps, anchor, feat, offsets, scaling, masks, center, n_offsets):
    """View-dependent decode of generate_neural_gaussians (use_feat_bank = False)."""
    ob_view = anchor - center
    ob_dist = ob_view.norm(dim=1, keepdim=True)
    ob_view = ob_view / ob_dist
    x = torch.cat([feat, ob_view, ob_dist], dim=1)
    neural_opacity = mlps["mlp_opacity"](x).reshape(-1, 1) * masks.reshape(-1, 1)
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


class ContextGSRenderer:
    capabilities = {"render.rgb", "render.depth.expected_z", "gradient.screen.signed"}

    def __init__(self, spec):
        c = spec.config
        self.feat_dim = int(c.get("feat_dim", 50))
        self.n_offsets = int(c.get("n_offsets", 10))
        self.bank = None  # explicit bank (probes); otherwise model.ACTIVE
        self._eval_mlps = None
        self._eval_key = None

    def active_bank(self):
        bank = self.bank if self.bank is not None else model.ACTIVE["bank"]
        return None if bank is None or bank.baked else bank

    def decoded_mlps(self, params):
        blob = params[BLOB]
        key = (blob.data_ptr(), tuple(blob.shape))
        if self._eval_key != key:
            self._eval_mlps = unpack_blob(blob.detach(), self.feat_dim, self.n_offsets)
            self._eval_key = key
        return self._eval_mlps

    def render(self, params, camera, sh_degree=None):
        bank = self.active_bank()
        if bank is not None and torch.is_grad_enabled():
            return self._render_train(bank, params, camera)
        return self._render_decoded(params, camera, bank)

    # -- decoded model (after bake, ss-eval) -------------------------------------------------
    def _render_decoded(self, params, camera, bank):
        if bank is not None:
            raise RuntimeError("ContextGS renders without gradient only for the baked/decoded model")
        mlps = self.decoded_mlps(params)
        vis = visible_anchors(params["means"], params["cgs_scaling"][:, :3], camera)
        g = neural_gaussians(mlps, params["means"][vis], params["cgs_feat"][vis],
                             params["cgs_offset"][vis], params["cgs_scaling"][vis],
                             params["cgs_mask"][vis], camera_center(camera["viewmat"]),
                             self.n_offsets)
        out = self._rasterize(g[:5], camera)
        out["info"]["cgs"] = {"visible": vis, "selection": g[6], "neural_opacity": g[5],
                              "scaling": g[3], "bits": None}
        return out

    # -- training ----------------------------------------------------------------------------
    def _render_train(self, bank, params, camera):
        step = bank.step
        q0 = bank.c["q0"]
        anchors_all = bank.quantize_anchor(params["means"])[0]
        with torch.no_grad():
            vis = visible_anchors(anchors_all, params["cgs_scaling"][:, :3].exp(), camera)
        bits = None
        if step == bank.c["entropy_start"]:
            bank.update_anchor_bound(params["means"])
        if step > bank.c["entropy_start"]:
            scaling_all = params["cgs_scaling"].exp()
            masks_all = ste_mask(params["cgs_mask"])
            mask_anchor_all = masks_all.detach().sum(dim=1)[:, 0] > 0
            x_means = {"feat": params["cgs_feat"].mean(), "scaling": scaling_all.mean(),
                       "offsets": params["cgs_offset"].mean()}
            feat, scaling, offsets, bits = levels.multi_scale(
                bank, anchors_all, params["cgs_hyper"], params["cgs_feat"],
                params["cgs_offset"], scaling_all, masks_all, mask_anchor_all, training=True,
                predict_bpp=True, x_means=x_means, sample=bank.c["rate_sample"],
                use_hyper=bank.c["use_hyper"])
            bits.pop("levels")
            bits.pop("order")
            anchor, feat, offsets = anchors_all[vis], feat[vis], offsets[vis]
            scaling, masks = scaling[vis], masks_all[vis]
        else:
            anchor = anchors_all[vis]
            feat = params["cgs_feat"][vis]
            offsets = params["cgs_offset"][vis]
            scaling = params["cgs_scaling"][vis].exp()
            masks = ste_mask(params["cgs_mask"][vis])
            if bank.c["noise_start"] < step <= bank.c["entropy_start"]:
                feat = feat + torch.empty_like(feat).uniform_(-0.5, 0.5) * q0["feat"]
                scaling = scaling + torch.empty_like(scaling).uniform_(-0.5, 0.5) * q0["scaling"]
                offsets = offsets + torch.empty_like(offsets).uniform_(-0.5, 0.5) * q0["offsets"]
        g = neural_gaussians(bank.mlps, anchor, feat, offsets, scaling, masks,
                             camera_center(camera["viewmat"]), self.n_offsets)
        out = self._rasterize(g[:5], camera)
        out["info"]["cgs"] = {"visible": vis, "selection": g[6], "neural_opacity": g[5],
                              "scaling": g[3], "bits": bits}
        return out

    def _rasterize(self, gaussians, camera):
        xyz, color, opacity, scales, quats = gaussians
        render, alpha, info = rasterization(
            means=xyz, quats=quats, scales=scales, opacities=opacity, colors=color,
            viewmats=camera["viewmat"][None], Ks=camera["K"][None], width=camera["width"],
            height=camera["height"], near_plane=0.2, sh_degree=None, packed=False,
            render_mode="RGB+ED",
        )
        # Runtime convention: RGB over a white background (official ContextGS: black for mip360).
        return {"rgb": render[0, ..., :3] + (1 - alpha[0]), "depth": render[0, ..., 3:4],
                "alpha": alpha[0], "info": info}


def ste_mask(logits):
    """get_mask: sigmoid > 0.01 with a straight-through gradient."""
    sig = torch.sigmoid(logits)
    return ((sig > 0.01).float() - sig).detach() + sig


def create_renderer(spec):
    return ContextGSRenderer(spec)
