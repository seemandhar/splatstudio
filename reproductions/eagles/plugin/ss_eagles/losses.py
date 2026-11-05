"""3DGS photometric loss on the progressively downsampled target (EAGLES §3.2)."""

import torch.nn.functional as F
from splatstudio.losses import ssim


def downsample_target(target, s):
    """Official utils/image_utils.downsample_image on an HWC image."""
    if s >= 1.0:
        return target
    chw = target.permute(2, 0, 1)[None]
    out = F.interpolate(chw, scale_factor=s, mode="bilinear", align_corners=False)
    return out[0].permute(1, 2, 0)


def create_photometric(spec):
    weight = float(spec.config.get("ssim_weight", 0.2))
    if not 0 <= weight <= 1:
        raise ValueError("ssim_weight must be in [0, 1]")

    def loss(params, outputs, target):
        rgb = outputs["rgb"]
        target = downsample_target(target, float(outputs.get("eagles_scale", 1.0)))
        if rgb.shape != target.shape:
            raise ValueError(f"render {tuple(rgb.shape)} != downsampled target {tuple(target.shape)}")
        return (1 - weight) * F.l1_loss(rgb, target) + weight * (1 - ssim(rgb, target))

    return loss
