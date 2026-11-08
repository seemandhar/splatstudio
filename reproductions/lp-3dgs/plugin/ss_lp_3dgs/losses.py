"""LP-3DGS mask sparsity loss (Eq. 9-10 as implemented in the official code)."""

from .core import FIELD, in_mask_window, mask_loss

DEFAULTS = {"weight": 5e-4, "mask_start": 19500, "mask_iters": 500}


def create_mask_loss(spec):
    unknown = set(spec.config) - set(DEFAULTS)
    if unknown:
        raise ValueError(f"Unknown LP-3DGS mask-loss options: {sorted(unknown)}")
    cfg = {**DEFAULTS, **spec.config}
    weight, start, iters = float(cfg["weight"]), int(cfg["mask_start"]), int(cfg["mask_iters"])

    def loss(params, outputs, target):
        mask = params[FIELD]
        if in_mask_window(outputs["step"], start, iters):
            return mask_loss(mask, weight)
        # Attached zero through the image, NOT through the mask: the mask must receive no
        # gradient outside the window so its Adam state stays empty (official optimiser is
        # created fresh at mask_start).
        return outputs["rgb"].sum() * 0.0

    return loss
