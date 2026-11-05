"""SOG smoothness ("neighbor") regularization on the 2D attribute grids (paper §3.2).

Official: scene/gaussian_model.py:neighborloss_2d and train.py:121-140, config
`neighbor_loss` of config/ours_q_sh_local_test.yaml (activated: false, normalize: false).
"""

import math

import kornia
import torch.nn.functional as F

# Official yaml weights opacity 1, rotation 10 (others 0); normalised by their sum they are the
# supp. Table values 0.09 / 0.91. Keys are SplatStudio field names (opacity -> opacities,
# rotation -> quats, xyz -> means, features_dc -> sh0, features_rest -> shN, scaling -> scales).
DEFAULTS = {
    "lambda_neighbor": 1.0,
    "kernel_size": 5,
    "sigma": 3.0,
    "loss_fn": "huber",
    "weights": {"means": 0.0, "sh0": 0.0, "shN": 0.0, "opacities": 1.0, "scales": 0.0,
                "quats": 10.0},
}


def grid_side(n):
    side = math.isqrt(n)
    if side * side != n:
        raise ValueError(f"{n} Gaussians do not fill a square grid (density must prune to square)")
    return side


def attr_flat(t):
    """Official get_attr_flat: un-activated attribute flattened to [N, C]."""
    return t.reshape(len(t), -1)


def as_grid_img(t):
    """Official as_grid_img: [N, C] -> [S, S, C] in parameter (row-major) order."""
    side = grid_side(len(t))
    return t.reshape(side, side, -1)


def neighborloss_2d(tensor, kernel_size, sigma, loss_fn):
    """Loss between the grid and its detached separable Gaussian blur (official neighborloss_2d)."""
    img = as_grid_img(attr_flat(tensor)).permute(2, 0, 1).unsqueeze(0)
    blurred_x = kornia.filters.gaussian_blur2d(
        img.detach(), kernel_size=(1, kernel_size), sigma=(sigma, sigma), border_type="circular")
    blurred_xy = kornia.filters.gaussian_blur2d(
        blurred_x, kernel_size=(kernel_size, 1), sigma=(sigma, sigma), border_type="reflect")
    if loss_fn == "mse":
        return F.mse_loss(blurred_xy, img)
    if loss_fn == "huber":
        return F.huber_loss(blurred_xy, img)
    raise ValueError(f"Unknown loss function {loss_fn}")


def neighbor_terms(params, cfg):
    """Per-attribute weighted terms lambda * w_a / sum(w) * L_a (train.py:127-134)."""
    weights = cfg["weights"]
    weight_sum = sum(weights.values())
    return {
        name: cfg["lambda_neighbor"] * w / weight_sum
        * neighborloss_2d(params[name], cfg["kernel_size"], cfg["sigma"], cfg["loss_fn"])
        for name, w in weights.items() if w > 0
    }


def create_neighbor_loss(spec):
    unknown = set(spec.config) - set(DEFAULTS)
    if unknown:
        raise ValueError(f"Unknown sog-neighbor options: {sorted(unknown)}")
    cfg = {**DEFAULTS, **spec.config}
    unknown_fields = set(cfg["weights"]) - set(DEFAULTS["weights"])
    if unknown_fields:
        raise ValueError(f"Unknown neighbor-loss attributes: {sorted(unknown_fields)}")
    if cfg["lambda_neighbor"] <= 0 or not any(w > 0 for w in cfg["weights"].values()):
        raise ValueError("sog-neighbor needs lambda_neighbor > 0 and a positive weight")

    def loss(params, outputs, target):
        return sum(neighbor_terms(params, cfg).values())

    return loss
