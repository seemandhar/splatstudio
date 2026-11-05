"""Official DropGaussian loss (paper Eq. 7, source/code/train.py:79-81):
L = L1 + lambda * (1 - SSIM), lambda = 0.2 — the L1 weight is 1.0 (3DGS uses 1 - lambda).
SSIM is 3DGS's (utils/loss_utils.py): 11x11 Gaussian window, sigma 1.5, same padding.
"""

import math

import torch
import torch.nn.functional as F

DEFAULTS = {"l1_weight": 1.0, "ssim_weight": 0.2}


def gaussian_window(size=11, sigma=1.5, channels=3, device="cpu"):
    g = torch.tensor(
        [math.exp(-((x - size // 2) ** 2) / (2 * sigma**2)) for x in range(size)], device=device
    )
    g = g / g.sum()
    w = g[:, None] @ g[None, :]
    return w.expand(channels, 1, size, size).contiguous()


def ssim_same(img1, img2, size=11):
    """3DGS ssim() on HWC images in [0,1] (conv2d padding = size // 2, mean over the map)."""
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
    unknown = set(spec.config) - set(DEFAULTS)
    if unknown:
        raise ValueError(f"Unknown DropGaussian loss options: {sorted(unknown)}")
    cfg = {**DEFAULTS, **spec.config}
    l1_w, ssim_w = float(cfg["l1_weight"]), float(cfg["ssim_weight"])

    def loss(params, outputs, target):
        rgb = outputs["rgb"]
        return l1_w * (rgb - target).abs().mean() + ssim_w * (1.0 - ssim_same(rgb, target))

    return loss
