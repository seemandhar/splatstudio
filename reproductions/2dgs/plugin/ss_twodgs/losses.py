"""2DGS regularisers (paper §5 "Training"; official train.py:77-87).

  normal_error = 1 - sum_c(rend_normal * surf_normal)      (rend_normal = sum_i w_i n_i, world,
                                                             unnormalised; surf_normal =
                                                             depth_to_normal(surf_depth)*alpha.detach())
  L_n = lambda_normal * mean(normal_error)  for step > 7000      (lambda_normal 0.05)
  L_d = lambda_dist   * mean(rend_dist)     for step > 3000      (code default 0.0; paper 100
                                                                   unbounded / 1000 bounded)
"""

DIST_DEFAULTS = {"lambda_dist": 0.0, "start": 3000}
NORMAL_DEFAULTS = {"lambda_normal": 0.05, "start": 7000}


def _cfg(spec, defaults, label):
    unknown = set(spec.config) - set(defaults)
    if unknown:
        raise ValueError(f"Unknown {label} options: {sorted(unknown)}")
    return {**defaults, **spec.config}


def normal_error(normal, surface_normal):
    """Per-pixel official normal error [H,W] (both maps HWC)."""
    return 1 - (normal * surface_normal).sum(dim=-1)


def create_normal(spec):
    c = _cfg(spec, NORMAL_DEFAULTS, "2DGS normal-loss")
    lam, start = float(c["lambda_normal"]), int(c["start"])

    def loss(params, outputs, target):
        weight = lam if outputs["step"] > start else 0.0  # train.py: iteration > 7000
        return weight * normal_error(outputs["normal"], outputs["surface_normal"]).mean()

    return loss


def create_distortion(spec):
    c = _cfg(spec, DIST_DEFAULTS, "2DGS distortion-loss")
    lam, start = float(c["lambda_dist"]), int(c["start"])

    def loss(params, outputs, target):
        weight = lam if outputs["step"] > start else 0.0  # train.py: iteration > 3000
        return weight * outputs["distortion"].mean()

    return loss
