"""Reduced-3DGS regularisers (official train.py / rasterizer backward).

* opacity L1 (paper §4.1 "L1 sparsity term on opacity"; train.py:
  `Lalpha_regul = gaussians.get_opacity[visibility_filter].abs().mean()`, weight
  `lambda_alpha_regul` = 0.001 in the `full_final` configuration).
* SH sparsity (paper §4.2 "sparsity loss on SH coefficients"). The official code adds it inside
  the CUDA backward: dL/dsh_k += lambda_sh / (V * 15 * 3) * sign(sh_k) for every visible
  Gaussian (radii > 0) and every coefficient of its active bands (`rasterizer_impl.cu:566`,
  `backward.cu:computeColorFromSH`), lambda_sh = 0.1. That is exactly the gradient of
  lambda_sh / (45 V) * sum_{visible} sum_{active k, rgb} |sh_k|, implemented here.
"""


from .renderer import DEGREE_FIELD, coefficient_mask


def visible_mask(outputs):
    radii = outputs["info"]["radii"]
    radii = radii[0]
    return (radii > 0).all(dim=-1) if radii.dim() == 2 else radii > 0


def active_degree(step, interval=1000, max_degree=3):
    """SplatStudio runtime SH schedule (runtime.py: degree = min((step - 1) // 1000, 3))."""
    return min((int(step) - 1) // interval, max_degree)


class OpacityL1:
    def __init__(self, spec):
        unknown = set(spec.config) - {"weight"}
        if unknown:
            raise ValueError(f"Unknown opacity-l1 options: {sorted(unknown)}")
        self.weight = float(spec.config.get("weight", 0.001))

    def __call__(self, params, outputs, target):
        vis = visible_mask(outputs)
        opacity = params["opacities"].sigmoid()
        if not bool(vis.any()):
            return opacity.sum() * 0.0
        return self.weight * opacity[vis].abs().mean()


class SHSparsity:
    def __init__(self, spec):
        unknown = set(spec.config) - {"weight", "degree_interval"}
        if unknown:
            raise ValueError(f"Unknown sh-sparsity options: {sorted(unknown)}")
        self.weight = float(spec.config.get("weight", 0.1))
        self.interval = int(spec.config.get("degree_interval", 1000))

    def __call__(self, params, outputs, target):
        vis = visible_mask(outputs)
        shn = params["shN"]
        n_vis = int(vis.sum())
        if n_vis == 0:
            return shn.sum() * 0.0
        active = active_degree(outputs["step"], self.interval)
        mask = coefficient_mask(params[DEGREE_FIELD][vis], active)
        return self.weight / (n_vis * 45.0) * (shn[vis].abs() * mask).sum()


def create_opacity_l1(spec):
    return OpacityL1(spec)


def create_sh_sparsity(spec):
    return SHSparsity(spec)
