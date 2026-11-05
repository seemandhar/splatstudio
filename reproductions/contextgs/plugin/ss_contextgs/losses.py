"""ContextGS loss terms besides the runtime photometric loss (official train.py:225-232):

  L = (1 - 0.2) L1 + 0.2 D-SSIM                      (runtime `photometric`, unchanged)
    + 0.01 * mean(prod(scale of neural Gaussians))   (Scaffold-GS volume regulariser, part of L_scaffold)
    + lambda_e * bit_per_param                        (Eq. 10-11, step > 10000)
    + lambda_m * mean(sigmoid(mask logits))          (masking loss L_m, step > 10000)
with bit_per_param = (sum of hyperprior + feature + scaling + offset bits of the sampled anchors) /
(number of feature + scaling + offset parameters of the sampled anchors) * masked-anchor rate
(the HAC normalisation of App. A; computed in levels.multi_scale).

The scaling regulariser is copied from the HAC native port (identical in both official codes).
"""

import torch


def create_scaling_reg(spec):
    weight = float(spec.config.get("weight", 0.01))

    def loss(params, outputs, target):
        scaling = outputs["info"]["cgs"]["scaling"]
        if scaling.numel() == 0:
            return params["cgs_scaling"].sum() * 0.0
        return weight * scaling.prod(dim=1).mean()

    return loss


def create_rate(spec):
    lmbda = float(spec.config.get("lmbda", 0.004))
    mask_weight = float(spec.config.get("mask_weight", 5e-4))

    def loss(params, outputs, target):
        bits = outputs["info"]["cgs"].get("bits")
        if bits is None:
            return params["cgs_mask"].sum() * 0.0
        return lmbda * bits["bit_per_param"] + mask_weight * torch.sigmoid(params["cgs_mask"]).mean()

    return loss
