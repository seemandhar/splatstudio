"""BBSplat regularisers (official train.py).

texture_reg (paper §3.3, Eq. 8-9; code version): for visible billboards (impact > 0),
  w = sigma - clamp(impact, 0, sigma)
  L = lambda_rgb * mean_vis(mean(T^RGB) * w) + lambda_alpha * mean_vis(|mean(T^alpha - A0)| * w)
with sigma = max_impact_threshold (code 100; paper text 500) and A0 = 0.5 G (initial alpha).
opacity_reg ("For MCMC sampler"): opacity_reg * mean(T^alpha) over all billboards and texels.
geometry: 2DGS normal consistency (after 7000) and distortion (after 3000), official unnormalised
form; weights 0 for Mip-NeRF 360 / Tanks&Temples in scripts/train_all.sh.
"""

import torch

from . import texture as tex

TEXTURE_DEFAULTS = {"max_impact_threshold": 100.0, "lambda_texture_value": 1e-4,
                    "lambda_alpha_value": 1e-4}
OPACITY_DEFAULTS = {"opacity_reg": 0.01}
GEOMETRY_DEFAULTS = {"lambda_normal": 0.0, "lambda_dist": 0.0, "normal_from": 7000,
                     "dist_from": 3000}


def _check(spec, defaults, label):
    unknown = set(spec.config) - set(defaults)
    if unknown:
        raise ValueError(f"Unknown BBSplat {label} options: {sorted(unknown)}")
    return {**defaults, **spec.config}


def texture_reg(alpha_logits, color_logits, impact, sigma, lam_rgb, lam_alpha):
    """train.py textures_reg (Eq. 9, code form)."""
    vis = impact > 0
    if not bool(vis.any()):
        return (alpha_logits.sum() + color_logits.sum()) * 0.0
    w = sigma - torch.clamp(impact[vis].detach(), 0, sigma)
    color = color_logits[vis].sigmoid()
    alpha = alpha_logits[vis].sigmoid()
    a0 = tex.initial_alpha(alpha.device)
    reg = (color.mean(dim=[1, 2, 3]) * w).mean() * lam_rgb
    return reg + torch.abs((alpha - a0).mean(dim=[1, 2]) * w).mean() * lam_alpha


def create_texture_reg(spec):
    c = _check(spec, TEXTURE_DEFAULTS, "texture-reg")

    def loss(params, outputs, target):
        return texture_reg(params[tex.ALPHA], params[tex.COLOR], outputs["info"]["impact"],
                           float(c["max_impact_threshold"]), float(c["lambda_texture_value"]),
                           float(c["lambda_alpha_value"]))

    return loss


def create_opacity_reg(spec):
    c = _check(spec, OPACITY_DEFAULTS, "opacity-reg")

    def loss(params, outputs, target):
        return float(c["opacity_reg"]) * params[tex.ALPHA].sigmoid().mean()

    return loss


def create_geometry(spec):
    c = _check(spec, GEOMETRY_DEFAULTS, "geometry")

    def loss(params, outputs, target):
        out = outputs["bbsplat"]
        step = outputs["step"]
        ln = float(c["lambda_normal"]) if step > c["normal_from"] else 0.0
        ld = float(c["lambda_dist"]) if step > c["dist_from"] else 0.0
        total = out["alpha"].sum() * 0.0
        if ln:
            err = 1 - (out["rend_normal"] * out["surf_normal"]).sum(dim=0)
            total = total + ln * err.mean()
        if ld:
            total = total + ld * out["rend_dist"].mean()
        return total

    return loss
