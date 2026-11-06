"""Photometric loss for the coarse-to-fine variants: (1 - 0.2) L1 + 0.2 D-SSIM (paper Sec. 3.5)
against the target bilinearly resized to the rendered size (train_anchor.py:
``F.interpolate(gt, size=cur_size, mode='bilinear', align_corners=False)``). At full resolution
it equals SplatStudio's base photometric loss exactly."""

import torch.nn.functional as F
from splatstudio.losses import ssim


def resize_target(target, height, width):
    if target.shape[0] == height and target.shape[1] == width:
        return target
    t = target.permute(2, 0, 1)[None]
    return F.interpolate(t, size=(height, width), mode="bilinear",
                         align_corners=False)[0].permute(1, 2, 0)


def create_photometric(spec):
    weight = float(spec.config.get("ssim_weight", 0.2))

    def loss(params, outputs, target):
        rgb = outputs["rgb"]
        tgt = resize_target(target, rgb.shape[0], rgb.shape[1])
        return (1 - weight) * F.l1_loss(rgb, tgt) + weight * (1 - ssim(rgb, tgt))

    return loss
