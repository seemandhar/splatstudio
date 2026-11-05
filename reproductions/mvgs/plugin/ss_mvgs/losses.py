"""MVGS multi-view regulated loss (Sec. 3.1, Eq. 4): the 3DGS photometric loss summed (not
averaged) over the M views of the step, `total_loss += loss` in train.py:86-112.

D-SSIM uses the official 3DGS training SSIM (utils/loss_utils.py: 11x11 Gaussian window,
sigma 1.5, zero padding 5, mean over all pixels); `ssim_padding: valid` selects the runtime's
valid-convolution SSIM instead.
"""

import torch
import torch.nn.functional as F
from splatstudio.losses import ssim as ssim_valid


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


def view_loss(rgb, target, weight, ssim):
    return (1 - weight) * F.l1_loss(rgb, target) + weight * (1 - ssim(rgb, target))


def create_photometric(spec):
    unknown = set(spec.config) - {"ssim_weight", "ssim_padding"}
    if unknown:
        raise ValueError(f"Unknown mvgs photometric options: {sorted(unknown)}")
    weight = float(spec.config.get("ssim_weight", 0.2))
    if not 0 <= weight <= 1:
        raise ValueError("ssim_weight must be in [0, 1]")
    padding = spec.config.get("ssim_padding", "same")
    if padding not in SSIM:
        raise ValueError("ssim_padding must be 'same' or 'valid'")
    ssim = SSIM[padding]

    def loss(params, outputs, target):
        views = outputs.get("mv_views")
        if views is None:  # no multi-view schedule active (probes / single-view renders)
            return view_loss(outputs["rgb"], target, weight, ssim)
        return sum(view_loss(v["rgb"], v["target"], weight, ssim) for v in views)

    return loss
