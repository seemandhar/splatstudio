"""MS-GS photometric loss (official train.py:209-215).

loss = ((1 - w) L1 + w (1 - SSIM)) * (1 if the render is 1x else 0.1), with the 3DGS same-padded
SSIM (utils/loss_utils.py: 11x11 Gaussian window, sigma 1.5, zero padding 5, mean), which also
works on the 12 x 8 px 128x renders. The runtime passes the 1x training image; for a
downsampled training render the target is resized to the render size with antialiased bicubic
interpolation (the official code PIL-bicubic-resizes the full-res JPEG; fidelity gap, measured
in evidence/gt_resample.json).
"""

import torch
import torch.nn.functional as F


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


def resize_target(target, height, width):
    """HWC target -> (height, width) by antialiased bicubic (identity when the size matches)."""
    if target.shape[0] == height and target.shape[1] == width:
        return target
    # CPU antialiased bicubic: PyTorch's CUDA antialias kernel fails beyond ~16x (shared memory)
    # and deviated from PIL at some sizes in evidence/gt_resample.json; the CPU path matches PIL
    # bicubic-from-full-res to >= 53 dB.
    x = target.detach().permute(2, 0, 1)[None].float().cpu()
    y = F.interpolate(x, size=(height, width), mode="bicubic", align_corners=False,
                      antialias=True)
    return y[0].permute(1, 2, 0).clamp(0, 1).to(target.device)


def create_photometric(spec):
    unknown = set(spec.config) - {"ssim_weight", "low_res_weight"}
    if unknown:
        raise ValueError(f"Unknown ms-photometric options: {sorted(unknown)}")
    weight = float(spec.config.get("ssim_weight", 0.2))
    low = float(spec.config.get("low_res_weight", 0.1))

    def loss(params, outputs, target):
        rgb = outputs["rgb"]
        gt = resize_target(target, rgb.shape[0], rgb.shape[1])
        value = (1 - weight) * F.l1_loss(rgb, gt) + weight * (1 - ssim_same(rgb, gt))
        return value * (1.0 if int(outputs.get("ms_level", 0)) == 0 else low)

    return loss
