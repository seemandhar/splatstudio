"""MaskGaussian squared mask loss (Eq. Lm / L, train.py).

L_m = (mean_i M_i)^2 over all N Gaussians, M = the straight-through Gumbel sample the renderer
used this step (outputs["info"]["mask"]); weight lambda(step) = lambda if from <= step <= until
else 0 (train.py simple_scheduler, both ends inclusive).
"""

import torch

DEFAULTS = {"lambda_mask": 0.1, "mask_from": 19000, "mask_until": 20000}  # Ours-alpha


def lambda_at(step, cfg):
    s = int(step)
    return float(cfg["lambda_mask"]) if cfg["mask_from"] <= s <= cfg["mask_until"] else 0.0


def mask_loss(mask, weight):
    return weight * torch.mean(mask) ** 2


def create_mask_loss(spec):
    unknown = set(spec.config) - set(DEFAULTS)
    if unknown:
        raise ValueError(f"Unknown MaskGaussian mask-loss options: {sorted(unknown)}")
    cfg = {**DEFAULTS, **spec.config}
    if float(cfg["lambda_mask"]) < 0 or cfg["mask_from"] > cfg["mask_until"]:
        raise ValueError("lambda_mask must be >= 0 and mask_from <= mask_until")

    def loss(params, outputs, target):
        mask = outputs["info"].get("mask")
        if mask is None:
            raise RuntimeError("mask loss needs outputs['info']['mask'] (MaskGaussian renderer)")
        return mask_loss(mask, lambda_at(outputs["step"], cfg))

    return loss
