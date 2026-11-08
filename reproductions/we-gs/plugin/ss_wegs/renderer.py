"""WE-GS renderer on gsplat (paper §4, Fig. 3).

Per render: (l_k, M_k) = Enc(I_k) from ``camera["image"]`` (training and evaluation frames both
carry their image: the paper encodes test images feed-forward); Delta sh = G_phi(sh, mu, l_k)
(Eq. 9); rasterise sh + Delta sh (Eq. 10) with gsplat's SH evaluation (3DGS clamp(SH + 0.5, 0))
over a black background (released code ``itw_white_background=False``).

Plain ``ss-eval`` has no bank: the renderer raises instead of silently rendering base SH. Use
``python -m ss_wegs.evaluate --run RUN``.
"""

import torch
import torch.nn.functional as F
from gsplat import rasterization

from . import bank as bank_mod

CAPABILITIES = ("render.rgb", "gradient.screen.signed")
DEFAULTS = {"background": 0.0}


def sh_coeffs(params):
    return torch.cat((params["sh0"], params["shN"]), dim=1)


def full_mask(mask, height, width):
    """R(M_k): bilinear resize of [1, 1, H', W'] to [H, W, 1]."""
    m = F.interpolate(mask, size=(int(height), int(width)), mode="bilinear", align_corners=False)
    return m[0].permute(1, 2, 0)


class Renderer:
    def __init__(self, spec):
        unknown = set(spec.config) - set(DEFAULTS)
        if unknown:
            raise ValueError(f"Unknown we-gs renderer options: {sorted(unknown)}")
        self.cfg = {**DEFAULTS, **spec.config}
        self.capabilities = set(CAPABILITIES)

    def conditioned_sh(self, params, camera, bank, embedding=None):
        if "image" not in camera and embedding is None:
            raise ValueError("WE-GS encodes the view's own image: camera['image'] is required")
        mask = None
        if embedding is None:
            embedding, mask = bank.encode(camera["image"])
        sh = sh_coeffs(params)
        delta = bank.residual(sh, params["means"], embedding)
        return sh + delta, delta, embedding, mask

    def render(self, params, camera, sh_degree=3, embedding=None):
        bank = bank_mod.ACTIVE
        if bank is None:
            raise RuntimeError("we-gs needs its learned bank: evaluate with "
                               "`python -m ss_wegs.evaluate --run RUN` (plain ss-eval cannot)")
        sh, delta, lk, mask = self.conditioned_sh(params, camera, bank, embedding)
        render, alpha, info = rasterization(
            means=params["means"], quats=params["quats"], scales=params["scales"].exp(),
            opacities=torch.sigmoid(params["opacities"]), colors=sh,
            viewmats=camera["viewmat"][None], Ks=camera["K"][None],
            width=int(camera["width"]), height=int(camera["height"]), sh_degree=int(sh_degree),
            packed=False, render_mode="RGB", absgrad=False, rasterize_mode="classic")
        rgb = render[0] + float(self.cfg["background"]) * (1 - alpha[0])
        out = {"rgb": rgb, "alpha": alpha[0], "info": info, "sh_residual": delta,
               "embedding": lk, "opacity": torch.sigmoid(params["opacities"])}
        if mask is not None:
            out["static_mask"] = mask
            out["static_mask_full"] = full_mask(mask, camera["height"], camera["width"])
        return out


def create_renderer(spec):
    return Renderer(spec)
