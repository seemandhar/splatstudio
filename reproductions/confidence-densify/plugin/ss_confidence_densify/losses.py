"""Losses: confidence-weighted photometric loss (paper Eq. confidence, CoMe) and the 3DGS-MCMC
opacity/scale regulariser (paper Eq. reg)."""

import torch
import torch.nn.functional as F

TARGET_KEY = "cad_target"
SSIM_ERROR_KEY = "cad_ssim_error"

CONF_DEFAULTS = {"ssim_weight": 0.2, "beta_conf": 0.075, "conf_min": 1e-3, "conf_max": 5.0,
                 "weight_photometric": True}
REG_DEFAULTS = {"opacity_reg": 0.01, "scale_reg": 0.01, "stop": 25000}


def ssim_map(image, target):
    """Per-pixel SSIM [H,W,3]: 11x11 Gaussian window (sigma 1.5), same padding (3DGS / CoMe
    `_ssim(size_average=False)`)."""
    coords = torch.arange(11, device=image.device, dtype=image.dtype) - 5
    g = torch.exp(-(coords**2) / (2 * 1.5**2))
    g = g / g.sum()
    window = (g[:, None] * g[None, :])[None, None].expand(3, 1, 11, 11).contiguous()
    x, y = (t.permute(2, 0, 1)[None] for t in (image, target))

    def conv(t):
        return F.conv2d(t, window, padding=5, groups=3)

    mx, my = conv(x), conv(y)
    vx, vy, cov = conv(x * x) - mx * mx, conv(y * y) - my * my, conv(x * y) - mx * my
    c1, c2 = 0.01**2, 0.03**2
    s = ((2 * mx * my + c1) * (2 * cov + c2)) / ((mx * mx + my * my + c1) * (vx + vy + c2))
    return s[0].permute(1, 2, 0)


def per_pixel_rgb_loss(rgb, target, ssim_weight, smap=None):
    """L_rgb(p) per pixel and channel (CoMe appearance_network.py:104)."""
    smap = ssim_map(rgb, target) if smap is None else smap
    return (1 - ssim_weight) * (rgb - target).abs() + ssim_weight * (1 - smap)


def confidence_loss_value(rgb, target, confidence, beta, ssim_weight, cmin, cmax, smap=None,
                          weight_photometric=True):
    """Eq. confidence: mean_p( L_rgb(p) C(p) - beta log C(p) ), C clamped to [cmin, cmax].

    weight_photometric=True (CoMe): this term replaces L_rgb, so C also weights the geometry and
    colour gradients. False (iteration 1, post-hoc reading of the paper: the confidence is only
    an error signal for sampling): geometry/colour get the plain L_rgb of 3DGS-MCMC, and the
    confidence term sees a detached L_rgb(p), so it trains only gamma with the same gradient.
    """
    c = confidence.clamp(cmin, cmax)
    pp = per_pixel_rgb_loss(rgb, target, ssim_weight, smap)
    if weight_photometric:
        return (pp * c).mean() - beta * torch.log(c).mean()
    return pp.mean() + (pp.detach() * c).mean() - beta * torch.log(c).mean()


def create_confidence_loss(spec):
    cfg = {**CONF_DEFAULTS, **spec.config}
    unknown = set(cfg) - set(CONF_DEFAULTS)
    if unknown:
        raise ValueError(f"Unknown confidence-loss options: {sorted(unknown)}")
    w, beta = float(cfg["ssim_weight"]), float(cfg["beta_conf"])
    cmin, cmax = float(cfg["conf_min"]), float(cfg["conf_max"])
    weighted = bool(cfg["weight_photometric"])

    def loss(params, outputs, target):
        rgb = outputs["rgb"]
        smap = ssim_map(rgb, target)
        # Hand the density component what the runtime does not give it: the target image and
        # the per-pixel structural error 1 - SSIM(p) (channel mean) for the SSIM term.
        outputs[TARGET_KEY] = target.detach()
        outputs[SSIM_ERROR_KEY] = (1 - smap.detach().mean(-1)).clamp_min(0)
        return confidence_loss_value(rgb, target, outputs["confidence"], beta, w, cmin, cmax, smap,
                                     weighted)

    return loss


def create_reg(spec):
    """3DGS-MCMC regulariser lambda (mean|o| + mean|s|) (gsplat simple_trainer / MCMC code),
    active until densification ends (paper Sec. 4: kept until iteration 25k)."""
    cfg = {**REG_DEFAULTS, **spec.config}
    unknown = set(cfg) - set(REG_DEFAULTS)
    if unknown:
        raise ValueError(f"Unknown regulariser options: {sorted(unknown)}")
    lo, ls, stop = float(cfg["opacity_reg"]), float(cfg["scale_reg"]), int(cfg["stop"])

    def loss(params, outputs, target):
        active = float(outputs["step"] < stop) if "step" in outputs else 1.0
        value = lo * params["opacities"].sigmoid().abs().mean() + ls * params[
            "scales"
        ].exp().abs().mean()
        return active * value

    return loss
