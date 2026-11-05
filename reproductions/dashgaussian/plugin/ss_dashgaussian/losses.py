"""3DGS photometric loss against a target resampled to the rendered resolution.

The runtime always passes the full-resolution training image as `target`; when the renderer
returns a reduced-resolution image, the target is downsampled with PyTorch's antialiased
bilinear interpolation (official train_dash.py:114, supp. Sec. 7.1).

D-SSIM padding (training loss only; evaluation SSIM is always the runtime's valid SSIM):
- "same" (iteration 1, default): 11x11 Gaussian window, sigma 1.5, zero padding 5, mean over all
  pixels, as 3DGS utils/loss_utils.ssim and fused_ssim used by the official DashGaussian code.
- "valid" (iteration 0): splatstudio.losses.ssim (valid convolution, drops a 5-px border).
"""

import torch
import torch.nn.functional as F
from splatstudio.losses import ssim as ssim_valid


def ssim_same(image, target):
    """HWC inputs; 3DGS `_ssim` with padding = window_size // 2 (zero padding), mean reduction."""
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


def downsample_target(target, height, width):
    """HWC target -> (height, width), antialiased bilinear (identity if the size matches)."""
    if target.shape[0] == height and target.shape[1] == width:
        return target
    x = target.permute(2, 0, 1)[None]
    y = F.interpolate(x, size=(height, width), mode="bilinear", align_corners=False,
                      antialias=True)
    return y[0].permute(1, 2, 0)


def create_photometric(spec):
    unknown = set(spec.config) - {"ssim_weight", "ssim_padding"}
    if unknown:
        raise ValueError(f"Unknown dash-photometric options: {sorted(unknown)}")
    weight = float(spec.config.get("ssim_weight", 0.2))
    if not 0 <= weight <= 1:
        raise ValueError("ssim_weight must be in [0, 1]")
    padding = spec.config.get("ssim_padding", "same")
    if padding not in SSIM:
        raise ValueError("ssim_padding must be 'same' or 'valid'")
    ssim = SSIM[padding]

    def loss(params, outputs, target):
        rgb = outputs["rgb"]
        gt = downsample_target(target, rgb.shape[0], rgb.shape[1])
        return (1 - weight) * F.l1_loss(rgb, gt) + weight * (1 - ssim(rgb, gt))

    return loss
