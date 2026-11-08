"""Pure functions of EFA-GS (arXiv 2508.02493) matching the official code.

Official reference: github.com/jcwang-gh/EFA-GS commit 57f330f, `Mip-Splatting/train.py`
(`calculate_training_percent_powered`, the splitting_lb override) and
`Mip-Splatting/scene/gaussian_model.py` (`densify_lff`, `densify_and_prune_lff`,
`compute_3D_interval`, `normalize_interval`). Equation references are to the paper.
"""

import math

import torch


def anneal_fraction(step, start=500, end=15000, power=1.0, lower=0.0):
    """Official `calculate_training_percent_powered` (App. C strategy 2, annealing).

    di = 1 - (step - start) / (end - start);  returns di**power * (1 - lower) + lower.
    The enlarging factor is c = exp(ln c(i) * this), so log c decays from ln c(i) at `start`
    to ln c_last at `end` (lower = ln c_last / ln c_max). The paper writes c = exp(k x^n);
    the code (followed here) decays.
    """
    di = 1.0 - (step - start) / (end - start)
    if not 0.0 < di <= 1.0:
        raise ValueError(f"step {step} outside the densification window ({start}, {end})")
    return math.exp(math.log(di) * power) * (1.0 - lower) + lower


def anneal_lower_bound(c_last, c_max):
    """train.py: percent_lb = log(c_last) / log(c_max) (1 when c_max == 1)."""
    return math.log(c_last) / math.log(c_max) if c_max != 1.0 else 1.0


def splitting_lower_bound(step, start=500, end=15000):
    """train.py overrides --splitting_lb every densification: 1 - (step - start)/(end - start)."""
    return 1.0 - (step - start) / (end - start)


def is_lfcf_step(step, interval_times=2, every=100):
    """`islff = iteration % (interval_times * densification_interval) == 0` (App. C Str. 1)."""
    return step % (interval_times * every) == 0


def normalize_log_minmax(interval):
    """Official `normalize_interval(interval, 'log', 'minmax')` -> theta in [0, 1]."""
    log_i = torch.log(interval)
    lo, hi = log_i.min(), log_i.max()
    return (log_i - lo) / (hi - lo)


def log_enlarging_factor(theta, c_min, c_max, fraction, depth_strategy=True):
    """Depth-based strategy (§4.3): log c per Gaussian, annealed.

    Code: c = theta * c_min + (1 - theta) * c_max with theta = normalised log sampling interval,
    so deep Gaussians (large interval, theta -> 1) get c_min ("deeper Gaussians ... lower enlarging
    factors"; the paper's printed formula has c_min/c_max swapped). `depth_strategy=False` is our
    ablation (uniform c_max), not an official switch.
    """
    if depth_strategy:
        c = theta * c_min + (1 - theta) * c_max
    else:
        c = torch.full_like(theta, c_max)
    return torch.log(c) * fraction


def enlarge_delta(log_scales, log_c, diffscale=True):
    """Δ log-scale for enlarged Gaussians (ExpandGaussian; scale-based strategy, §4.3).

    diffscale: per-axis factors (c, c^-1/3, c^-2/3) for the (smallest, middle, largest) axis,
    product 1 (volume preserving). Otherwise c on every axis. log_c: [K, 1].
    """
    if not diffscale:
        return log_c.expand_as(log_scales).clone()
    coef = torch.ones_like(log_scales)
    order = torch.sort(log_scales, dim=1, descending=False)[1]
    coef.scatter_(1, order[:, 1:2], -1.0 / 3)
    coef.scatter_(1, order[:, 2:3], -2.0 / 3)
    return log_c * coef


def shrink_delta(log_scales, log_c, split_multiplier=2.0, diffscale=True):
    """Δ log-scale for shrunk (split-candidate) Gaussians (ShrinkGaussian).

    -0.5 (log c + ln split_multiplier) times (1, 0.5, 0) for the (largest, middle, smallest) axis
    with diffscale, else on every axis.
    """
    base = -0.5 * (log_c + math.log(split_multiplier))
    if not diffscale:
        return base.expand_as(log_scales).clone()
    coef = torch.ones_like(log_scales)
    order = torch.sort(log_scales, dim=1, descending=True)[1]
    coef.scatter_(1, order[:, 1:2], 0.5)
    coef.scatter_(1, order[:, 2:3], 0.0)
    return base * coef


def quantile_linear(values, q):
    """torch.quantile(values, q) (linear interpolation) without its 2^24-element limit."""
    s = torch.sort(values.reshape(-1))[0]
    pos = q * (s.numel() - 1)
    lo = int(math.floor(pos))
    hi = min(lo + 1, s.numel() - 1)
    return s[lo] + (s[hi] - s[lo]) * (pos - lo)


def abs_threshold(grads, grads_abs, grad_threshold):
    """Mip-Splatting (GOF densification) quantile threshold for absolute gradients.

    ratio = mean(g >= tau); Q = quantile(g_abs, 1 - ratio): the absolute criterion selects the
    same fraction of Gaussians as the signed one.
    """
    ratio = (grads >= grad_threshold).float().mean()
    return quantile_linear(grads_abs, 1.0 - float(ratio))


def lfcf_masks(grads, grads_abs, grad_threshold, q_abs, prev_selected, prev_grads,
               prev_grads_abs, tolerance=1e-5):
    """LFCF decision (Alg. 1 + official `densify_lff`).

    selected = g >= tau or g_abs >= Q. Selected Gaussians not selected at the previous LFCF step
    are enlarged; for the others, "decent" (gradient did not rise) = g <= |g_prev - tol| or
    g_abs <= |g_abs_prev - tol|: decent -> shrink (and maybe split), otherwise enlarge.
    All inputs are [N] tensors (prev_selected bool). Returns (selected, enlarged, shrunk, both).
    """
    selected = (grads >= grad_threshold) | (grads_abs >= q_abs)
    both = prev_selected & selected
    enlarged = ~prev_selected & selected
    decent = (grads <= (prev_grads - tolerance).abs()) | (
        grads_abs <= (prev_grads_abs - tolerance).abs()
    )
    shrunk = both & decent
    enlarged = enlarged | (both & ~decent)
    return selected, enlarged, shrunk, both


def split_probability(theta, lower, upper=1.0):
    """App. C strategy 3: threshold = theta (ub - lb) + lb; deeper Gaussians split more often."""
    return theta * (upper - lower) + lower
