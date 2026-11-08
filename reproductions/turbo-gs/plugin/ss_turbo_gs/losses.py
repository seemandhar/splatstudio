"""3DGS photometric loss (L1 + 0.2 D-SSIM, unchanged) on the dilated pixel subset.

The runtime passes the full training image; with dilated rendering only the sampled pixels
image[o_y::d, o_x::d] are supervised (Sec. 4.1). The detached loss value is exported in
`outputs["training_loss"]` for the convergence-aware budget scheduler (Sec. 4.2), which fits a
power law to the training loss.
"""

import torch.nn.functional as F
from splatstudio.losses import ssim

from .renderer import dilated_target


def create_photometric(spec):
    unknown = set(spec.config) - {"ssim_weight"}
    if unknown:
        raise ValueError(f"Unknown turbo photometric options: {sorted(unknown)}")
    weight = float(spec.config.get("ssim_weight", 0.2))
    if not 0 <= weight <= 1:
        raise ValueError("ssim_weight must be in [0, 1]")

    def loss(params, outputs, target):
        rgb = outputs["rgb"]
        d, ox, oy = outputs.get("dilation", (1, 0, 0))
        gt = dilated_target(target, d, ox, oy, rgb.shape[0], rgb.shape[1])
        if gt.shape != rgb.shape:
            raise ValueError(f"target subset {tuple(gt.shape)} != render {tuple(rgb.shape)}")
        value = (1 - weight) * F.l1_loss(rgb, gt) + weight * (1 - ssim(rgb, gt))
        outputs["training_loss"] = value.detach()
        return value

    return loss
