"""GaussianSpa "optimizing" step penalty (Eq. 11; official `OptimizingSpa.append_spa_loss`)."""

import torch

from . import spa

PENALTY_DEFAULTS = {
    "delta": 5e-4,  # rho_lr (arguments/__init__.py); adjust_rho is never called
    "spa_start": 15200,
    "spa_stop": 25200,
    "spa_interval": 50,
    "stop_inclusive": True,  # train_opacity.py: it <= stop; train_imp_score.py: it < stop
}


def create_penalty(spec):
    unknown = set(spec.config) - set(PENALTY_DEFAULTS)
    if unknown:
        raise ValueError(f"Unknown spa-penalty options: {sorted(unknown)}")
    cfg = {**PENALTY_DEFAULTS, **spec.config}
    delta = float(cfg["delta"])
    start, stop, every = int(cfg["spa_start"]), int(cfg["spa_stop"]), int(cfg["spa_interval"])
    inclusive = bool(cfg["stop_inclusive"])

    def loss(params, outputs, target):
        logit = params["opacities"]
        if not spa.penalty_active(outputs["step"], start, stop, every, inclusive):
            return logit.sum() * 0.0  # inactive: zero, graph-attached
        a = torch.sigmoid(logit)
        return spa.penalty(a, params["spa_z"], params["spa_u"], delta)

    return loss
