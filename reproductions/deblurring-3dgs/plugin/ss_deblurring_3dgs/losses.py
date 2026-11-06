"""Deblurring 3DGS objective: unchanged 3DGS loss on the blurred training render (train.py):
(1 - lambda_dssim) L1 + lambda_dssim (1 - SSIM), lambda_dssim = 0.2, with the Inria SSIM
(utils/loss_utils.ssim: 11x11 Gaussian window sigma 1.5, zero 'same' padding, mean)."""

import torch
import torch.nn.functional as F

DEFAULTS = {"lambda_dssim": 0.2}


def ssim_same(image, target, window_size=11, sigma=1.5):
    """HWC inputs; utils/loss_utils._ssim with padding = window_size // 2."""
    coords = torch.arange(window_size, device=image.device, dtype=image.dtype) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    window = (g[:, None] * g[None, :])[None, None].expand(3, 1, window_size, window_size)
    x, y = (t.permute(2, 0, 1)[None] for t in (image, target))
    pad = window_size // 2

    def conv(t):
        return F.conv2d(t, window, padding=pad, groups=3)

    mx, my = conv(x), conv(y)
    vx, vy, cov = conv(x * x) - mx * mx, conv(y * y) - my * my, conv(x * y) - mx * my
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    return (((2 * mx * my + c1) * (2 * cov + c2))
            / ((mx * mx + my * my + c1) * (vx + vy + c2))).mean()


def create_photometric(spec):
    unknown = set(spec.config) - set(DEFAULTS)
    if unknown:
        raise ValueError(f"Unknown deblur3dgs-photometric options: {sorted(unknown)}")
    lam = float({**DEFAULTS, **spec.config}["lambda_dssim"])

    def loss(params, outputs, target):
        rgb = outputs["rgb"]
        return (1.0 - lam) * (rgb - target).abs().mean() + lam * (1.0 - ssim_same(rgb, target))

    return loss
