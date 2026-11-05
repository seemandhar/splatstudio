"""HAC loss terms besides the runtime photometric loss (train.py):

  L = (1 - 0.2) L1 + 0.2 D-SSIM                      (runtime `photometric`, unchanged)
    + 0.01 * mean(prod(scale of neural Gaussians))   (Scaffold-GS volume regulariser)
    + lambda_e * (bit_per_param + L_hash / denom)    (Eq. 7, step > 10000)
    + lambda_m * mean(sigmoid(mask logits))          (offset-mask rate, step > 10000)
with denom = N_anchor * (feat_dim + 6 + 3 K).
"""

import torch


def create_scaling_reg(spec):
    weight = float(spec.config.get("weight", 0.01))

    def loss(params, outputs, target):
        scaling = outputs["info"]["hac"]["scaling"]
        if scaling.numel() == 0:
            return params["hac_scaling"].sum() * 0.0
        return weight * scaling.prod(dim=1).mean()

    return loss


def create_rate(spec):
    lmbda = float(spec.config.get("lmbda", 0.004))
    mask_weight = float(spec.config.get("mask_weight", 5e-4))
    feat_dim = int(spec.config.get("feat_dim", 50))
    n_offsets = int(spec.config.get("n_offsets", 10))

    def loss(params, outputs, target):
        bits = outputs["info"]["hac"].get("bits")
        if bits is None:
            return params["hac_mask"].sum() * 0.0
        denom = len(params["means"]) * (feat_dim + 6 + 3 * n_offsets)
        rate = lmbda * (bits["bit_per_param"] + bits["hash_bits"] / denom)
        return rate + mask_weight * torch.sigmoid(params["hac_mask"]).mean()

    return loss
