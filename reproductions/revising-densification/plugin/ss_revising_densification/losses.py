"""Auxiliary loss of Revising Densification (arXiv 2404.06109).

`revdens-aux` does two things on every step:
1. Sec. 3.5 residual-transmittance regulariser: weight * mean_u (1 - alpha(u)).
2. Sec. 3.2 auxiliary per-pixel error E_pi(u) (gradient detached), stored in
   outputs["revdens_error"] for the density component, which attributes it to primitives with
   the alpha-compositing weights (kernels.error_stats). The density's after_backward hook does
   not receive the target image, so the loss (which does) hands the map over through the
   outputs dict of the same step.
"""

from . import core

ERROR_KEY = "revdens_error"
LOSS_DEFAULTS = {"transmittance_weight": 0.1, "error": "ssim", "error_until": 27000}


def create_aux(spec):
    unknown = set(spec.config) - set(LOSS_DEFAULTS)
    if unknown:
        raise ValueError(f"Unknown revdens-aux options: {sorted(unknown)}")
    c = {**LOSS_DEFAULTS, **spec.config}
    weight = float(c["transmittance_weight"])
    kind = c["error"]
    until = int(c["error_until"])
    if weight < 0:
        raise ValueError("transmittance_weight must be >= 0")
    if kind not in ("ssim", "l1"):
        raise ValueError("error must be 'ssim' or 'l1'")

    def loss(params, outputs, target):
        if outputs.get("step", 0) < until:
            outputs[ERROR_KEY] = core.error_map(outputs["rgb"], target, kind)
        # Sec. 3.5: residual transmittance T_K(u) = 1 - accumulated alpha, pushed to zero
        return weight * (1.0 - outputs["alpha"]).mean()

    return loss
