"""Square-grid pruning and PLAS sorting of all Gaussian attributes (paper §3.1).

Official: scene/gaussian_model.py:prune_all_but_these_indices / prune_to_square_shape /
normalize / sort_into_grid (sorting config of config/ours_q_sh_local_test.yaml). The sorter is
PLAS (github.com/fraunhoferhhi/PLAS, Apache-2.0, installed package `plas`).
"""

import math

import torch
import torch.nn.functional as F
from gsplat.strategy import ops
from plas import sort_with_plas

from .renderer import inverse_log_transform

# Official attribute order of `sorting.weights` (xyz, features_dc, features_rest, opacity,
# scaling, rotation) mapped to SplatStudio field names.
ORDER = ("means", "sh0", "shN", "opacities", "scales", "quats")
MIN_GRID_SIDE = 16  # PLAS min_block_size; smaller grids cannot be sorted


def activated(params, name):
    """Official get_activated_attr_flat: values as the renderer sees them, [N, C]."""
    p = params[name].detach()
    if name == "means":
        v = inverse_log_transform(p)
    elif name == "opacities":
        v = torch.sigmoid(p)[:, None]
    elif name == "scales":
        v = torch.exp(p)
    elif name == "quats":
        v = F.normalize(p, dim=-1)
    else:  # sh0 / shN: get_features_dc / get_features_rest return the raw coefficients
        v = p
    return v.reshape(len(v), -1)


def raw(params, name):
    p = params[name].detach()
    return p.reshape(len(p), -1)


def standardize(t):
    """Official GaussianModel.normalize: global (not per-channel) mean / std."""
    t = t - t.mean()
    if t.std() > 0:
        t = t / t.std()
    return t


@torch.no_grad()
def permute(params, optimizers, state, indices):
    """Reorder params, Adam moments and per-Gaussian strategy state by `indices`."""
    n = len(params["means"])

    def param_fn(name, p):
        return torch.nn.Parameter(p[indices], requires_grad=p.requires_grad)

    def optimizer_fn(key, v):
        return v[indices]

    ops._update_param_with_optimizer(param_fn, optimizer_fn, params, optimizers)
    for k, v in list(state.items()):
        if isinstance(v, torch.Tensor) and v.dim() > 0 and len(v) == n:
            state[k] = v[indices]


@torch.no_grad()
def prune_to_square(params, optimizers, state):
    """Keep the S*S = floor(sqrt N)^2 most opaque Gaussians, in their original order."""
    n = len(params["means"])
    side = math.isqrt(n)
    removed = n - side * side
    if removed:
        alpha = torch.sigmoid(params["opacities"].detach().flatten())
        keep = torch.topk(alpha, k=side * side).indices
        mask = torch.ones(n, dtype=torch.bool, device=alpha.device)
        mask[keep] = False
        ops.remove(params, optimizers, state, mask)
    return side, removed


def sort_features(params, weights, normalize=True, use_activated=True):
    getter = activated if use_activated else raw
    norm = standardize if normalize else (lambda x: x)
    feats = [norm(getter(params, name)) * weights[name] for name in ORDER if weights.get(name, 0) > 0]
    if not feats:
        raise ValueError("sorting needs at least one positive weight")
    return torch.cat(feats, dim=1)


@torch.no_grad()
def sort_indices(features, side, improvement_break, shuffle=True):
    """Permutation that places Gaussian sorted_indices[k] at grid cell k (row-major)."""
    if side < MIN_GRID_SIDE:
        raise ValueError(f"grid side {side} < {MIN_GRID_SIDE}: PLAS cannot sort it")
    n = side * side
    if len(features) != n:
        raise ValueError("feature count does not match the square grid")
    shuffled = None
    if shuffle:
        shuffled = torch.randperm(n, device=features.device)
        features = features[shuffled]
    grid = features.reshape(side, side, -1).permute(2, 0, 1)
    _, idx = sort_with_plas(grid.contiguous(), improvement_break=improvement_break, verbose=False)
    idx = idx.squeeze().flatten().long()
    if shuffle:
        idx = shuffled[idx]
    return idx


@torch.no_grad()
def sort_into_grid(params, optimizers, state, cfg):
    side = math.isqrt(len(params["means"]))
    feats = sort_features(params, cfg["sort_weights"], cfg["sort_normalize"], cfg["sort_activated"])
    idx = sort_indices(feats, side, cfg["improvement_break"], cfg["sort_shuffle"])
    permute(params, optimizers, state, idx)
    return idx


def neighbour_distance(features, side):
    """Mean L2 distance between horizontally / vertically adjacent grid cells (sort quality)."""
    g = features.reshape(side, side, -1)
    dx = (g[:, 1:] - g[:, :-1]).norm(dim=-1).mean()
    dy = (g[1:] - g[:-1]).norm(dim=-1).mean()
    return float((dx + dy) / 2)
