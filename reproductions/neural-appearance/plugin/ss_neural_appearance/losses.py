"""Losses of the official FasterGSVDALoss beyond the runtime photometric term (0.8 L1 + 0.2 D-SSIM):
LAMBDA_OPACITY_REGULARIZATION * mean(sigmoid(o)) + LAMBDA_SCALE_REGULARIZATION * mean(exp(s))
(MCMC, 0.01 each) + PPISP regularization (weight 1.0). During the PPISP controller distillation
(step > train_steps) the official code sets both Gaussian regularizers to 0 and the PPISP
parameters are frozen, so this term is a graph-attached zero.
"""

import torch

from . import bank as bank_mod

DEFAULTS = {"opacity_weight": 0.01, "scale_weight": 0.01, "ppisp_weight": 1.0,
            "train_steps": 30000}


def create_regularizers(spec):
    unknown = set(spec.config) - set(DEFAULTS)
    if unknown:
        raise ValueError(f"Unknown regularizer options: {sorted(unknown)}")
    cfg = {**DEFAULTS, **spec.config}

    def loss(params, outputs, target):
        if outputs["step"] > cfg["train_steps"]:
            return outputs["rgb"].sum() * 0.0
        value = (cfg["opacity_weight"] * torch.sigmoid(params["opacities"]).mean()
                 + cfg["scale_weight"] * torch.exp(params["scales"]).mean())
        bank = outputs["info"].get("neural_appearance") or bank_mod.ACTIVE
        if bank is not None and cfg["ppisp_weight"] > 0:
            reg = bank.ppisp_regularization()
            if reg is not None:
                value = value + cfg["ppisp_weight"] * reg
        return value

    return loss
