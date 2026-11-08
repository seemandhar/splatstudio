"""DBS regularisers (paper Sec. 4.4 loss; official train.py):
  L += opacity_reg * mean_i |o_i| + scale_reg * mean_i,j |s_ij|   for start < step < stop
(paper writes sums over primitives and sqrt(eig(Sigma)) = activated scales; the code uses means).
"""

import torch

OPTIONS = {"opacity_reg": 0.01, "scale_reg": 0.01, "start": 500, "stop": 25_000}


def regularizer(params, step, opacity_reg, scale_reg, start, stop):
    value = (opacity_reg * torch.sigmoid(params["opacities"]).abs().mean()
             + scale_reg * torch.exp(params["scales"]).abs().mean())
    # multiply (not skip) outside the window so the loss stays attached to the graph
    return value * float(start < step < stop)


def create_regularizer(spec):
    cfg = {**OPTIONS, **spec.config}
    unknown = set(cfg) - set(OPTIONS)
    if unknown:
        raise ValueError(f"Unknown regularizer options: {sorted(unknown)}")

    def loss(params, outputs, target):
        return regularizer(params, outputs["step"], float(cfg["opacity_reg"]),
                           float(cfg["scale_reg"]), int(cfg["start"]), int(cfg["stop"]))

    return loss
