"""Compact-3DGS renderer: volume mask + R-VQ geometry + neural view-dependent colour -> gsplat.

Training (live `CompactBank` in model.ACTIVE, not baked) follows the official
gaussian_renderer/__init__.py `render(..., itr, rvq_iter)`:
  M = sg(1[sigmoid(m) > eps] - sigmoid(m)) + sigmoid(m)            (Eq. 1, eps = 0.01)
  s = RVQ_s(exp(scales)) if step > rvq_iter else exp(scales);  s_hat = M s      (Eq. 2, 4-5)
  q = RVQ_r(normalize(quats)) if step > rvq_iter else normalize(quats)
  o_hat = M sigmoid(opacities)
  sh0 = MLP(cat[HashGrid(contract(means.detach())), SH_tcnn(dir)])  (Eq. 7-8), SH degree 0
Decoded model (after the final-step bake, and in a fresh process such as ss-eval) follows the
official test path `render(itr=-1)` after `precompute()`: the stored attributes are rendered as
they are, the hash features are the baked per-Gaussian `c3dgs_feature`, and the MLP weights come
from `c3dgs_blob` (the runtime checkpoints every field, so the global weights reach ss-eval; the
blob pattern is copied from the HAC port, reproductions/hac/plugin/ss_hac/renderer.py).
"""

import torch
from gsplat import rasterization

from . import model

MASK = "c3dgs_mask"
FEATURE = "c3dgs_feature"
BLOB = "c3dgs_blob"
BLOB_WIDTH = 8
BLOB_HEADER = 8
MAGIC = 4343.0


# ------------------------------------------------------------------------- blob packing
@torch.no_grad()
def pack_blob(weights, n_rows, n_in, sh_degree, quirk, device):
    """[n_rows, 8]: header (magic, version, length, n_in, n_layers, checksum, sh degree, quirk) +
    the flat MLP weights."""
    flat = torch.cat([w.detach().float().reshape(-1).to(device) for w in weights])
    header = torch.tensor([MAGIC, 1.0, float(BLOB_HEADER + flat.numel()), float(n_in),
                           float(len(weights)), float(flat.double().sum()), float(sh_degree),
                           float(bool(quirk))], device=device)
    data = torch.cat([header, flat])
    if data.numel() > n_rows * BLOB_WIDTH:
        raise ValueError(f"{n_rows} Gaussians cannot hold {data.numel()} MLP floats")
    out = torch.zeros(n_rows * BLOB_WIDTH, device=device)
    out[: data.numel()] = data
    return out.view(n_rows, BLOB_WIDTH)


def unpack_blob(blob):
    flat = blob.detach().reshape(-1)
    if flat.numel() < BLOB_HEADER or float(flat[0]) != MAGIC:
        raise ValueError("No Compact-3DGS MLP weights in the state (c3dgs_blob is empty): the "
                         "model was not encoded/baked at the end of training")
    length, n_in, n_layers = int(flat[2]), int(flat[3]), int(flat[4])
    shapes = model.mlp_shapes(n_in, 64, n_layers - 1)
    if BLOB_HEADER + sum(a * b for a, b in shapes) != length or length > flat.numel():
        raise ValueError("c3dgs_blob length disagrees with the MLP layout")
    if not torch.isclose(flat[BLOB_HEADER:length].double().sum(), flat[5].double(), rtol=1e-5,
                         atol=1e-3):
        raise ValueError("c3dgs_blob checksum mismatch (rows reordered or corrupted)")
    weights, i = [], BLOB_HEADER
    for a, b in shapes:
        weights.append(flat[i : i + a * b].view(a, b).clone())
        i += a * b
    return weights, int(flat[6]), bool(flat[7] > 0.5)


def straight_through_mask(mask_logits, threshold):
    """Eq. 1: binary forward value, sigmoid gradient."""
    s = torch.sigmoid(mask_logits)
    return ((s > threshold).float() - s).detach() + s


class CompactRenderer:
    capabilities = {"render.rgb", "render.depth.expected_z", "gradient.screen.signed"}

    def __init__(self, spec):
        c = dict(spec.config)
        unknown = set(c) - {"mask_threshold"}
        if unknown:
            raise ValueError(f"Unknown compact-renderer options: {sorted(unknown)}")
        self.mask_threshold = float(c.get("mask_threshold", 0.01))
        self.bank = None  # explicit bank (probes); otherwise model.ACTIVE
        self._eval = None
        self._eval_key = None

    def active_bank(self):
        bank = self.bank if self.bank is not None else model.ACTIVE["bank"]
        return None if bank is None or bank.baked else bank

    def render(self, params, camera, sh_degree=None):
        bank = self.active_bank()
        if bank is None:
            return self.render_decoded(params, camera)
        return self.render_train(bank, params, camera, torch.is_grad_enabled())

    # -- training ---------------------------------------------------------------------------
    def render_train(self, bank, params, camera, train=True):
        mask = straight_through_mask(params[MASK], self.mask_threshold)
        scales = params["scales"].exp()
        quats = torch.nn.functional.normalize(params["quats"], dim=-1)
        rvq = bank.step > bank.rvq_iter
        if rvq:
            scales = bank.vq_scale(scales, train)[0]
            quats = bank.vq_rot(quats, train)[0]
        scales = scales * mask[:, None]
        opacities = torch.sigmoid(params["opacities"]) * mask
        sh0 = bank.color(params["means"], model.camera_center(camera["viewmat"]))
        out = self._rasterize(params["means"], quats, scales, opacities, sh0, camera)
        out["info"]["compact3dgs"] = {"mask": mask, "rvq": rvq}
        return out

    # -- decoded model (after bake, ss-eval) -------------------------------------------------
    def decoded_mlp(self, params):
        blob = params[BLOB]
        key = (blob.data_ptr(), tuple(blob.shape))
        if self._eval_key != key:
            self._eval = unpack_blob(blob)
            self._eval_key = key
        return self._eval

    def render_decoded(self, params, camera):
        weights, degree, quirk = self.decoded_mlp(params)
        center = model.camera_center(camera["viewmat"])
        sh0 = model.color_sh0(weights, params[FEATURE], params["means"], center, degree, quirk)
        return self._rasterize(params["means"], params["quats"], params["scales"].exp(),
                               torch.sigmoid(params["opacities"]), sh0, camera)

    def _rasterize(self, means, quats, scales, opacities, sh0, camera):
        render, alpha, info = rasterization(
            means=means, quats=quats, scales=scales, opacities=opacities, colors=sh0[:, None, :],
            viewmats=camera["viewmat"][None], Ks=camera["K"][None], width=camera["width"],
            height=camera["height"], sh_degree=0, packed=False, render_mode="RGB+ED",
        )
        # Runtime convention: RGB over a white background (official: black).
        return {"rgb": render[0, ..., :3] + (1 - alpha[0]), "depth": render[0, ..., 3:4],
                "alpha": alpha[0], "info": info}


def create_renderer(spec):
    return CompactRenderer(spec)
