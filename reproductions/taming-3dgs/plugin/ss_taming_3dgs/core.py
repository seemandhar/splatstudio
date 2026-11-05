"""Pure functions of Taming 3DGS (CPU-testable). References are to the official code
https://github.com/humansensinglab/taming-3dgs (fd0f7d9) and arXiv 2406.15643."""

import numpy as np
import torch
import torch.nn.functional as F

# Official full_eval.py `big_budgets` ("Ours (#3DGS)": final count = 3DGS's count), keyed by the
# scene's initial SfM point count in our native datasets (unique per scene; the native point
# clouds hold exactly the COLMAP points3D).
SCENE_SFM_COUNTS = {
    "bicycle": 54275,
    "flowers": 38347,
    "garden": 138766,
    "stump": 32049,
    "treehill": 52363,
    "room": 112627,
    "counter": 155767,
    "kitchen": 241367,
    "bonsai": 206613,
    "truck": 136029,
    "train": 182686,
    "drjohnson": 80861,
    "playroom": 37005,
}
BIG_BUDGETS = {
    "bicycle": 5987095,
    "flowers": 3618411,
    "garden": 5728191,
    "stump": 4867429,
    "treehill": 3770257,
    "room": 1548960,
    "counter": 1190919,
    "kitchen": 1803735,
    "bonsai": 1252367,
    "truck": 2584171,
    "train": 1085480,
    "playroom": 2326100,
    "drjohnson": 3273600,
}
# Official full_eval.py `budget_multipliers` ("Ours", paper Sec. 5.2).
BUDGET_MULTIPLIERS = {
    "bicycle": 15, "flowers": 15, "garden": 15, "stump": 15, "treehill": 15,
    "room": 2, "counter": 2, "kitchen": 2, "bonsai": 2,
    "truck": 2, "train": 2, "playroom": 5, "drjohnson": 5,
}


def table_by_count(values):
    """{scene: value} -> {str(initial SfM count): value} (JSON-serializable config)."""
    return {str(SCENE_SFM_COUNTS[s]): v for s, v in values.items()}


def resolve_budget(initial_count, mode, budget, table):
    """Final Gaussian count B. `budget` (explicit) wins; else look up the scene by its count."""
    value = budget if budget is not None else table.get(str(int(initial_count)))
    if value is None:
        raise ValueError(
            f"No Taming budget for a scene with {initial_count} initial points; set the density "
            "config 'budget' explicitly"
        )
    if mode == "multiplier":
        return int(initial_count * float(value))
    if mode == "final_count":
        return int(value)
    raise ValueError("mode must be 'multiplier' or 'final_count'")


def count_array(start_count, budget, densify_from, densify_until, interval):
    """Eq. 2 as implemented in utils/taming_utils.py:get_count_array (parabola peaking at B)."""
    num_steps = (densify_until - densify_from) // interval
    slope_lower_bound = (budget - start_count) / num_steps
    k = 2 * slope_lower_bound
    a = (budget - start_count - k * num_steps) / (num_steps * num_steps)
    return [int(1 * a * (x**2) + (k * x) + start_count) for x in range(num_steps)]


def normalize(weight, values):
    """Official `normalize`: NaN -> 0; positive entries scaled by weight / median(positive)."""
    values = torch.nan_to_num(values.float(), nan=0.0)
    valid = values > 0
    out = torch.zeros_like(values)
    if bool(valid.any()):
        v = values[valid]
        out[valid] = weight * (v / torch.median(v))
    return out


def split_budgets(target, current, n_clone, n_split):
    """Official densify_with_score budget arithmetic -> (clone_budget, split_budget)."""
    if n_clone + n_split == 0:
        return 0, 0
    budget = min(target, n_clone + n_split + current)
    clone = ((budget - current) * n_clone) // (n_clone + n_split)
    split = ((budget - current) * n_split) // (n_clone + n_split)
    return max(clone, 0), max(split, 0)


def sample_without_replacement(weights, n):
    """torch.multinomial(weights, n, replacement=False), exactly as the official code calls it.

    When fewer than n weights are positive, torch (CPU and CUDA) does not raise but fills the
    remainder with zero-weight indices, so the official densification always adds the full
    budget (possibly cloning zero-score or even non-candidate Gaussians); this is reproduced.
    Only the all-zero case (which would raise in torch) returns no samples.
    """
    n = min(int(n), weights.numel())
    if n <= 0 or not bool((weights > 0).any()):
        return torch.zeros(0, dtype=torch.long, device=weights.device)
    return torch.multinomial(weights.float(), n, replacement=False)


def minmax(x):
    lo, hi = x.min(), x.max()
    return (x - lo) / (hi - lo) if float(hi - lo) > 0 else torch.zeros_like(x)


def ssim_same(image, target):
    """fused-ssim / 3DGS `ssim`: 11x11 Gaussian (sigma 1.5), zero 'same' padding, mean. HWC."""
    coords = torch.arange(11, device=image.device, dtype=image.dtype) - 5
    g = torch.exp(-(coords**2) / (2 * 1.5**2))
    g /= g.sum()
    window = (g[:, None] * g[None, :])[None, None].expand(3, 1, 11, 11)
    x, y = (t.permute(2, 0, 1)[None] for t in (image, target))

    def conv(t):
        return F.conv2d(t, window, padding=5, groups=3)

    mx, my = conv(x), conv(y)
    vx, vy, cov = conv(x * x) - mx * mx, conv(y * y) - my * my, conv(x * y) - mx * my
    c1, c2 = 0.01**2, 0.03**2
    return (((2 * mx * my + c1) * (2 * cov + c2)) / ((mx * mx + my * my + c1) * (vx + vy + c2))).mean()


def photometric(image, target, lambda_dssim=0.2):
    """utils/taming_utils.py:compute_photometric_loss (per-view weight P_i)."""
    l1 = (image - target).abs().mean()
    return (1.0 - lambda_dssim) * l1 + lambda_dssim * (1.0 - ssim_same(image, target))


def edge_map(image_uint8):
    """utils/taming_utils.py:get_edges: PIL FIND_EDGES on the 8-bit grayscale image -> [H,W]
    uint8 (min-max normalization happens at use, as in train.py)."""
    from PIL import Image, ImageFilter

    arr = image_uint8.cpu().numpy() if isinstance(image_uint8, torch.Tensor) else image_uint8
    edges = Image.fromarray(np.ascontiguousarray(arr)).convert("L").filter(ImageFilter.FIND_EDGES)
    return torch.from_numpy(np.asarray(edges).copy())


def saliency(render, target, edges_norm, mse_importance=50.0, edge_importance=50.0):
    """Eq. 3 as in utils/taming_utils.py:get_loss_map (ROI = all pixels). HWC render/target."""
    l1 = (render - target).abs().mean(dim=-1).detach()
    return mse_importance * minmax(l1) + edge_importance * edges_norm


def gaussian_terms(weights, grads, opacities, scale_prod, depths, radii):
    """Per-Gaussian part of F (compute_gaussian_score: g_importance)."""
    return (
        normalize(weights["grad_importance"], grads)
        + normalize(weights["opac_importance"], opacities)
        + normalize(weights["dept_importance"], depths)
        + normalize(weights["radii_importance"], radii)
        + normalize(weights["scale_importance"], scale_prod)
    )


def pixel_terms(weights, stats):
    """Per-pixel part of F (compute_gaussian_score: p_importance)."""
    return (
        normalize(weights["dist_importance"], stats["dist"])
        + normalize(weights["loss_importance"], stats["saliency"])
        + normalize(weights["count_importance"], stats["count"])
        + normalize(weights["blend_importance"], stats["blend"])
    )


SCORE_WEIGHTS = {
    # train.py `score_coefficients` (code values; paper text: grad 50, no radius term).
    "view_importance": 50,
    "edge_importance": 50,
    "mse_importance": 50,
    "grad_importance": 25,
    "dist_importance": 50,
    "opac_importance": 100,
    "dept_importance": 5,
    "loss_importance": 10,
    "radii_importance": 10,
    "scale_importance": 25,
    "count_importance": 0.1,
    "blend_importance": 50,
}
