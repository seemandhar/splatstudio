"""Compact-3DGS mask loss (paper Eq. 3; official train.py `opt.lambda_mask * mean(sigmoid(_mask))`)."""

import torch

from .renderer import MASK


def create_mask_loss(spec):
    unknown = set(spec.config) - {"weight"}
    if unknown:
        raise ValueError(f"Unknown mask-loss options: {sorted(unknown)}")
    weight = float(spec.config.get("weight", 5e-4))

    def loss(params, outputs, target):
        return weight * torch.mean(torch.sigmoid(params[MASK]))

    return loss
