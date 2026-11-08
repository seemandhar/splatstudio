"""Effective-rank regularisation (paper Eq. 1-4; official train.py / utils/get_stats.py).

Code form used for every reported run (train.sh):
    erank_loss = erank_lambda * clamp(-log(erank - 1 + 1e-7), 0).mean()
    thin_loss  = thin_lambda * sort(scales, descending)[:, 2].mean()
active for erank_from_iter <= step <= erank_end_iter. The paper writes a sum over Gaussians
and eps = 1e-5; the code (which produced the numbers) uses the mean and eps = 1e-7.
"""

import torch

DEFAULTS = {
    "erank_lambda": 0.01,   # paper Sec. 4.1, train.sh --erank_lambda 0.01
    "thin_lambda": 1.0,     # train.sh --thin_lambda 1 (paper: "+ s_3", weight 1)
    "eps": 1e-7,            # train.py (paper Eq. 4: 1e-5)
    "start": 7000,          # paper Sec. 3.2, train.sh --erank_from_iter 7000
    "end": 30000,           # code default erank_end_iter = iterations
}


def effective_rank(scales):
    """get_effective_rank: q_i = s_i^2 / sum s^2, erank = exp(-sum q log q) (Eq. 1-3)."""
    d = scales * scales
    q = d / d.sum(dim=1, keepdim=True)
    return torch.exp(-(q * torch.log(q)).sum(dim=1))


def erank_terms(scales, cfg):
    erank = effective_rank(scales)
    barrier = cfg["erank_lambda"] * torch.clamp(-torch.log(erank - 1 + cfg["eps"]), min=0).mean()
    s_sorted, _ = torch.sort(scales, dim=1, descending=True)
    thin = cfg["thin_lambda"] * s_sorted[:, 2].mean()
    return barrier, thin


def create_erank_loss(spec):
    unknown = set(spec.config) - set(DEFAULTS)
    if unknown:
        raise ValueError(f"Unknown erank options: {sorted(unknown)}")
    cfg = {**DEFAULTS, **spec.config}

    def loss(params, outputs, target):
        step = int(outputs["step"])
        scales = params["scales"]
        if not cfg["start"] <= step <= cfg["end"]:
            return scales.sum() * 0.0
        barrier, thin = erank_terms(scales.exp(), cfg)
        return barrier + thin

    return loss
