"""Pure functions of the BOGausS port (CPU-testable). Section names refer to arXiv 2504.01844.

There is no official BOGausS code; the choices for unspecified details are listed in
reproductions/bogauss/paper.md section 7 and in the plugin README.
"""

import functools
import math

import torch

# "Ours (Big)" targets the vanilla 3DGS count of each scene. The paper's #G column equals the
# Taming 3DGS "big" budgets (T&T mean 1.83M, Deep Blending mean 2.80M), i.e. the per-scene
# counts of the official Taming full_eval.py `big_budgets`
# (https://github.com/humansensinglab/taming-3dgs, fd0f7d9), keyed here by the initial SfM point
# count of our native scenes (unique per scene; same table as the SplatStudio Taming port).
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


def budget_table(fraction=1.0):
    """{str(initial SfM count): budget}; `fraction` gives the paper's lighter variants."""
    return {str(SCENE_SFM_COUNTS[s]): int(round(v * fraction)) for s, v in BIG_BUDGETS.items()}


def resolve_budget(initial_count, budget, table):
    value = budget if budget is not None else table.get(str(int(initial_count)))
    if value is None:
        raise ValueError(
            f"No BOGausS budget for a scene with {initial_count} initial points; set the density "
            "config 'budget' explicitly"
        )
    return int(value)


def count_target(step, start_count, budget, densify_from, densify_until, interval):
    """Target count after the densification at `step`: Taming's parabolic growth curve
    (utils/taming_utils.py:get_count_array) through (0, S), peaking at (n, B) with
    n = (until - from) / interval and x = (step - from) / interval, so the last call reaches B."""
    n = (densify_until - densify_from) // interval
    x = (step - densify_from) / interval
    k = 2.0 * (budget - start_count) / n
    a = (budget - start_count - k * n) / (n * n)
    return int(min(a * x * x + k * x + start_count, budget)) if x < n else int(budget)


def densify_steps(densify_from, densify_until, interval):
    """Steps at which densification runs: from+interval, ..., until (inclusive)."""
    return list(range(densify_from + interval, densify_until + 1, interval))


@torch.no_grad()
def precision_delta(means, cameras, alpha=2.0, sigma_uv=1.0, margin=1.3, near=0.2):
    """Delta_i = alpha * tr(Sigma_i^-1)^-0.5 (section "Precision matrix derivation").

    Per camera, 2D localization noise sigma_uv (pixels) gives a rank-2 lateral precision
    f^2 / (z^2 sigma_uv^2) on each image axis (pseudo-inverse of the paper's per-camera term),
    so tr(Sigma^-1) = sum_c (fx^2 + fy^2) / (z_c^2 sigma_uv^2) over the cameras whose frustum
    (3DGS-style 1.3x margin, z > near) contains the centre. Gaussians in no frustum get the
    largest finite Delta. Returns (delta [N], number of cameras seeing each Gaussian [N]).
    """
    prec = torch.zeros(len(means), device=means.device, dtype=torch.float32)
    seen = torch.zeros(len(means), device=means.device, dtype=torch.int32)
    for cam in cameras:
        vm = cam["viewmat"].to(means.device, torch.float32)
        K = cam["K"].to(means.device, torch.float32)
        pc = means @ vm[:3, :3].T + vm[:3, 3]
        z = pc[:, 2]
        fx, fy = K[0, 0], K[1, 1]
        tx = margin * 0.5 * cam["width"] / fx
        ty = margin * 0.5 * cam["height"] / fy
        zs = z.clamp_min(near)
        inside = (z > near) & ((pc[:, 0] / zs).abs() < tx) & ((pc[:, 1] / zs).abs() < ty)
        prec += torch.where(inside, (fx * fx + fy * fy) / (zs * zs), 0.0) / (sigma_uv**2)
        seen += inside.int()
    delta = torch.full_like(prec, float("inf"))
    ok = prec > 0
    delta[ok] = alpha * prec[ok].rsqrt()
    if bool(ok.any()):
        delta[~ok] = delta[ok].max()
    else:
        delta[:] = 1.0
    return delta, seen


def quat_to_rotmat(q):
    """(N,4) wxyz -> (N,3,3); columns are the principal axes."""
    q = torch.nn.functional.normalize(q, dim=-1)
    w, x, y, z = q.unbind(-1)
    return torch.stack(
        [
            1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
            2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
            2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
        ],
        dim=-1,
    ).reshape(-1, 3, 3)


def children_alpha(x, offset, ratio, o_child):
    """Alpha profile of two children at +-offset (sizes `ratio`, opacity o_child) composited."""
    g1 = o_child * torch.exp(-0.5 * ((x - offset) / ratio) ** 2)
    g2 = o_child * torch.exp(-0.5 * ((x + offset) / ratio) ** 2)
    return 1.0 - (1.0 - g1) * (1.0 - g2)


@functools.cache
def split_adaptation_table(offset=0.3, n_opacity=64):
    """Offline-learned SplitAdaptation (section "Density-preserving Gaussian splitting").

    The paper fits (sigma_child, o_child) = SplitAdaptation(sigma_mother, o_mother) offline so
    that the two children reproduce the mother's density; the function is not given. Here: for
    a unit mother (sigma = 1) of opacity o, children at +-offset along the split axis, grid
    search of (ratio = sigma_child / sigma_mother along that axis, o_child) minimizing the L2
    error between the mother's alpha profile o exp(-x^2/2) and the alpha-composited children,
    viewed perpendicular to the split axis (the other two axes are unchanged). By scale
    invariance, sigma_child = ratio(o) * sigma_mother. Returns (opacity grid, ratio, o_child).
    """
    x = torch.linspace(-4.5, 4.5, 181, dtype=torch.float64)
    ratios = torch.linspace(0.6, 1.05, 91, dtype=torch.float64)
    os_ = torch.linspace(0.005, 0.995, n_opacity, dtype=torch.float64)
    best_r, best_o = [], []
    for o in os_.tolist():
        target = o * torch.exp(-0.5 * x * x)
        oc = torch.linspace(max(o * 0.3, 1e-4), min(o, 0.999), 120, dtype=torch.float64)
        alpha = children_alpha(x[None, None], offset, ratios[:, None, None], oc[None, :, None])
        err = (alpha - target).square().sum(-1)
        i = int(err.argmin())
        best_r.append(float(ratios[i // len(oc)]))
        best_o.append(float(oc[i % len(oc)]))
    return os_.float(), torch.tensor(best_r).float(), torch.tensor(best_o).float()


def split_adaptation(opacity, offset=0.3):
    """Interpolate the offline table: mother opacity [M] -> (size ratio [M], child opacity [M])."""
    grid, ratio, ochild = (t.to(opacity.device) for t in split_adaptation_table(float(offset)))
    o = opacity.clamp(float(grid[0]), float(grid[-1]))
    pos = (o - grid[0]) / (grid[1] - grid[0])
    lo = pos.floor().long().clamp(0, len(grid) - 2)
    w = pos - lo.float()
    r = ratio[lo] * (1 - w) + ratio[lo + 1] * w
    oc = ochild[lo] * (1 - w) + ochild[lo + 1] * w
    return r, oc


def snr_priority_increment(se):
    """One camera's contribution SE_{c,i} / sum_j SE_{c,j} to P_SNR(i)."""
    total = se.sum()
    return se / total if float(total) > 0 else torch.zeros_like(se)


def effective_opacity(blend, count):
    """P_i^(c) = sum_p w_i(p) / S_i^(c), with S = contributing pixels; 0 where unseen."""
    return torch.where(count > 0, blend / count.clamp_min(1.0), torch.zeros_like(blend))


def prune_selection(pmax, threshold, fraction):
    """Indices to prune: P_prune < threshold, lowest first, at most floor(fraction * N)."""
    cand = torch.nonzero(pmax < threshold).flatten()
    cap = int(math.floor(fraction * len(pmax)))
    if len(cand) > cap:
        order = torch.argsort(pmax[cand])[:cap]
        cand = cand[order]
    return cand


def lr_calibration(delta, sigma, extent, ref_means=1.6e-4, ref_log_scales=0.005):
    """Dimensionless step sizes for Delta-scaled updates (paper gives none; paper.md 7.1):
    the median initial step equals 3DGS's (means lr 1.6e-4 * extent, log-scale lr 0.005)."""
    med = float(delta.median())
    eta_means = ref_means * extent / med
    eta_scales = ref_log_scales * float((sigma / delta[:, None]).median())
    return eta_means, eta_scales
