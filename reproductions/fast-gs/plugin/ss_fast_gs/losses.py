"""3DGS photometric loss with the official FastGS D-SSIM (fused_ssim: 11x11 Gaussian window,
sigma 1.5, zero "same" padding, mean over all pixels); Eq. 7, lambda = 0.2. The same function
gives E_photo of Eq. 5 for the pruning score. (Same as the SplatStudio DashGaussian port's
`ssim_same`.) Evaluation SSIM stays the runtime's.
"""

import torch
import torch.nn.functional as F


def ssim_same(image, target):
    """HWC inputs; 3DGS `_ssim` with padding = window_size // 2 (zero padding), mean."""
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


def photometric(image, target, weight=0.2):
    """(1 - lambda) L1 + lambda (1 - SSIM) on HWC images (Eq. 5 / Eq. 7)."""
    return (1 - weight) * (image - target).abs().mean() + weight * (1 - ssim_same(image, target))


def create_photometric(spec):
    unknown = set(spec.config) - {"ssim_weight"}
    if unknown:
        raise ValueError(f"Unknown fastgs-photometric options: {sorted(unknown)}")
    weight = float(spec.config.get("ssim_weight", 0.2))
    if not 0 <= weight <= 1:
        raise ValueError("ssim_weight must be in [0, 1]")

    def loss(params, outputs, target):
        return photometric(outputs["rgb"], target, weight)

    return loss
