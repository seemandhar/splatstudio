"""DNGaussian photometric-pass losses (official train_llff.py:146-162).

photometric (Eq. 10): L1 + lambda * (1 - SSIM), lambda = 0.2, L1 weight 1.0; SSIM is 3DGS's
(11x11 Gaussian window, sigma 1.5, same padding). ``ssim_same`` is copied from the
DropGaussian plugin (reproductions/dropgaussian/plugin/ss_dropgaussian/losses.py).

regularization (code only): shape_pena * mean(max_s / min_s) + scale_pena * mean(max_s^2)
+ opa_pena * (1 - mean(o[o > 0.2]^2) + mean((1 - o[o < 0.2])^2)) over all Gaussians, with
shape 0.002, scale 0.001 (default, run_llff.sh leaves it), opacity 0.001. When a selection is
empty the official value is NaN (its logger skips NaN); that term is constant w.r.t. the
parameters, so it is replaced by 0 here: identical gradients, finite loss.
"""

import math

import torch
import torch.nn.functional as F

PHOTO_DEFAULTS = {"l1_weight": 1.0, "ssim_weight": 0.2}
REG_DEFAULTS = {"shape_pena": 0.002, "scale_pena": 0.001, "opa_pena": 0.001, "opa_split": 0.2}


def gaussian_window(size=11, sigma=1.5, channels=3, device="cpu"):
    g = torch.tensor(
        [math.exp(-((x - size // 2) ** 2) / (2 * sigma**2)) for x in range(size)], device=device
    )
    g = g / g.sum()
    w = g[:, None] @ g[None, :]
    return w.expand(channels, 1, size, size).contiguous()


def ssim_same(img1, img2, size=11):
    """3DGS ssim() on HWC images (conv2d padding = size // 2, mean over the map)."""
    x, y = (t.permute(2, 0, 1)[None] for t in (img1, img2))
    c = x.shape[1]
    w = gaussian_window(size, 1.5, c, x.device).to(x.dtype)
    pad = size // 2
    mu1 = F.conv2d(x, w, padding=pad, groups=c)
    mu2 = F.conv2d(y, w, padding=pad, groups=c)
    mu1_sq, mu2_sq, mu12 = mu1.pow(2), mu2.pow(2), mu1 * mu2
    s1 = F.conv2d(x * x, w, padding=pad, groups=c) - mu1_sq
    s2 = F.conv2d(y * y, w, padding=pad, groups=c) - mu2_sq
    s12 = F.conv2d(x * y, w, padding=pad, groups=c) - mu12
    c1, c2 = 0.01**2, 0.03**2
    m = ((2 * mu12 + c1) * (2 * s12 + c2)) / ((mu1_sq + mu2_sq + c1) * (s1 + s2 + c2))
    return m.mean()


def create_photometric(spec):
    unknown = set(spec.config) - set(PHOTO_DEFAULTS)
    if unknown:
        raise ValueError(f"Unknown DNGaussian photometric options: {sorted(unknown)}")
    cfg = {**PHOTO_DEFAULTS, **spec.config}
    l1_w, ssim_w = float(cfg["l1_weight"]), float(cfg["ssim_weight"])

    def loss(params, outputs, target):
        rgb = outputs["rgb"]
        return l1_w * (rgb - target).abs().mean() + ssim_w * (1.0 - ssim_same(rgb, target))

    return loss


def _mean_or_zero(x, like):
    return x.mean() if x.numel() else like.sum() * 0.0


def penalties(scales_log, opacity, split=0.2):
    s = scales_log.exp()
    smax, smin = s.max(dim=1).values, s.min(dim=1).values
    shape = (smax / smin).mean()
    scale = smax.square().mean()
    hi, lo = opacity[opacity > split], opacity[opacity < split]
    opa = 1 - _mean_or_zero(hi.square(), opacity) + _mean_or_zero((1 - lo).square(), opacity)
    return shape, scale, opa


def create_regularization(spec):
    unknown = set(spec.config) - set(REG_DEFAULTS)
    if unknown:
        raise ValueError(f"Unknown DNGaussian regularization options: {sorted(unknown)}")
    cfg = {**REG_DEFAULTS, **spec.config}

    def loss(params, outputs, target):
        shape, scale, opa = penalties(params["scales"], outputs["opacity"], float(cfg["opa_split"]))
        return (float(cfg["shape_pena"]) * shape + float(cfg["scale_pena"]) * scale
                + float(cfg["opa_pena"]) * opa)

    return loss
