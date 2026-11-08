"""3DGS photometric loss against the frequency-modulated training image I_n (Eq. 4-5).

The loss form is unchanged (L1 + 0.2 D-SSIM, the runtime's SSIM so that the comparison with the
shared gsplat baseline isolates the modulation); only the target is mean-blurred with the
scheduled kernel. Evaluation is untouched (original images)."""

import torch.nn.functional as F
from splatstudio.losses import ssim

from .filtering import PADDING_MODES, check_schedule, kernel_at, mean_blur

DEFAULTS = {
    "ssim_weight": 0.2,
    # Fig. 3 step curve: 15 -> 9 -> 5 -> 3, original images after 12000 (Sec. 4).
    "kernel_sizes": [15, 9, 5, 3],
    "boundaries": [1000, 4000, 8000, 12000],
    "padding_mode": "reflect",
}


def photometric(rgb, target, weight):
    return (1 - weight) * F.l1_loss(rgb, target) + weight * (1 - ssim(rgb, target))


def create_photometric(spec):
    unknown = set(spec.config) - set(DEFAULTS)
    if unknown:
        raise ValueError(f"Unknown opti3dgs photometric options: {sorted(unknown)}")
    cfg = {**DEFAULTS, **spec.config}
    weight = float(cfg["ssim_weight"])
    if not 0 <= weight <= 1:
        raise ValueError("ssim_weight must be in [0, 1]")
    sizes, bounds = check_schedule(cfg["kernel_sizes"], cfg["boundaries"])
    mode = cfg["padding_mode"]
    if mode not in PADDING_MODES:
        raise ValueError(f"padding_mode must be one of {PADDING_MODES}")

    def loss(params, outputs, target):
        if "step" not in outputs:
            raise KeyError("opti3dgs photometric loss needs outputs['step'] for its schedule")
        k = kernel_at(outputs["step"], sizes, bounds)
        return photometric(outputs["rgb"], mean_blur(target, k, mode), weight)

    return loss
