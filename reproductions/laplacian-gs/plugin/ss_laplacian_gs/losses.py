"""Reconstructed-image supervision (Sec. 3.3, Alg. 1 line 19, Tab. 6 default).

L_rec^l = (1 - 0.2) * L1(I_hat^l, I^l) + 0.2 * (1 - SSIM(I_hat^l, I^l)), where I_hat^l is the
composed image of the active stage (renderer output) and I^l = Down^l(GT) is the Gaussian-
pyramid level of the full-resolution training image the runtime passes as `target`.
D-SSIM uses fused_ssim's convention (11x11 Gaussian window, sigma 1.5, zero "same" padding).
"""

import torch
import torch.nn.functional as F
from splatstudio.losses import ssim as ssim_valid

from . import pyramid as P


def ssim_same(image, target):
    coords = torch.arange(11, device=image.device, dtype=image.dtype) - 5
    gauss = torch.exp(-(coords**2) / (2 * 1.5**2))
    gauss = gauss / gauss.sum()
    window = (gauss[:, None] * gauss[None, :])[None, None].expand(3, 1, 11, 11).contiguous()
    x, y = (t.permute(2, 0, 1)[None] for t in (image, target))

    def conv(t):
        return F.conv2d(t, window, padding=5, groups=3)

    mx, my = conv(x), conv(y)
    vx, vy, cov = conv(x * x) - mx * mx, conv(y * y) - my * my, conv(x * y) - mx * my
    c1, c2 = 0.01**2, 0.03**2
    return (((2 * mx * my + c1) * (2 * cov + c2)) / ((mx * mx + my * my + c1) * (vx + vy + c2))).mean()


SSIM = {"same": ssim_same, "valid": ssim_valid}


def create_loss(spec):
    unknown = set(spec.config) - {"ssim_weight", "ssim_padding"}
    if unknown:
        raise ValueError(f"Unknown lap-photometric options: {sorted(unknown)}")
    weight = float(spec.config.get("ssim_weight", 0.2))
    if not 0 <= weight <= 1:
        raise ValueError("ssim_weight must be in [0, 1]")
    padding = spec.config.get("ssim_padding", "same")
    if padding not in SSIM:
        raise ValueError("ssim_padding must be 'same' or 'valid'")
    ssim = SSIM[padding]

    def loss(params, outputs, target):
        if "lap_level" not in outputs:
            raise ValueError("lap-photometric needs the renderer's lap_level output")
        rgb = outputs["rgb"]
        gt = P.target_at_level(target, int(outputs["lap_level"]))
        if gt.shape != rgb.shape:
            raise RuntimeError(f"pyramid target {tuple(gt.shape)} != render {tuple(rgb.shape)}")
        return (1 - weight) * F.l1_loss(rgb, gt) + weight * (1 - ssim(rgb, gt))

    return loss
