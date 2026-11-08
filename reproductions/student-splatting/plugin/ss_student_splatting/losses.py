"""SSS regularisers (paper Sec. 3.4 loss, from 3DGS-MCMC; official train.py:131-132):
  L += opacity_reg * mean_i |tanh(o_i)| + scale_reg * mean_ij |exp(s_ij)|   at every step
(paper writes sums over |o| and sqrt of the covariance eigenvalues = activated scales; the code
uses means of the signed opacity magnitude and of the scales)."""

import torch

OPTIONS = {"opacity_reg": 0.01, "scale_reg": 0.01}


def regularizer(params, opacity_reg, scale_reg):
    return (opacity_reg * torch.tanh(params["opacities"]).abs().mean()
            + scale_reg * torch.exp(params["scales"]).abs().mean())


def create_regularizer(spec):
    cfg = {**OPTIONS, **spec.config}
    unknown = set(cfg) - set(OPTIONS)
    if unknown:
        raise ValueError(f"Unknown regularizer options: {sorted(unknown)}")

    def loss(params, outputs, target):
        return regularizer(params, float(cfg["opacity_reg"]), float(cfg["scale_reg"]))

    return loss
