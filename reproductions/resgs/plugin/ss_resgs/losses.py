"""3DGS photometric loss against the image-pyramid target (official train.py:loss_fn).

The runtime passes the full-resolution training image; the target is reduced with the exact
cv2.pyrDown port `pyramid_level` times (official camera images are pyrDown'ed float images).
D-SSIM uses the official 3DGS `utils/loss_utils.ssim` (11x11 Gaussian window, sigma 1.5, zero
padding 5) by default; "valid" selects the runtime's valid-convolution SSIM.
"""

import torch
import torch.nn.functional as F
from splatstudio.losses import ssim as ssim_valid

from .pyramid import pyramid_target


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


def create_photometric(spec):
    unknown = set(spec.config) - {"ssim_weight", "ssim_padding"}
    if unknown:
        raise ValueError(f"Unknown resgs photometric options: {sorted(unknown)}")
    weight = float(spec.config.get("ssim_weight", 0.2))
    if not 0 <= weight <= 1:
        raise ValueError("ssim_weight must be in [0, 1]")
    padding = spec.config.get("ssim_padding", "same")
    if padding not in SSIM:
        raise ValueError("ssim_padding must be 'same' or 'valid'")
    ssim = SSIM[padding]

    def loss(params, outputs, target):
        rgb = outputs["rgb"]
        gt = pyramid_target(target, outputs.get("pyramid_level", 0))
        if gt.shape != rgb.shape:
            raise RuntimeError(f"Rendered size {tuple(rgb.shape)} != pyramid target {tuple(gt.shape)}")
        return (1 - weight) * F.l1_loss(rgb, gt) + weight * (1 - ssim(rgb, gt))

    return loss
