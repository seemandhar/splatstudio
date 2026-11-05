"""Pure functions of the Improving-ADC density rules (arXiv 2503.14274), CPU-testable.

Official code references are to https://github.com/fraunhoferhhi/Improving-ADC-3DGS (65901d8).
"""

import math

import torch


def camera_extent(centers):
    """3DGS getNerfppNorm radius, paper Eq. 11/12: 1.1 * max_i ||C_i - mean(C)||."""
    c = centers.mean(dim=0)
    return 1.1 * float(torch.linalg.vector_norm(centers - c, dim=1).max())


def corrected_extent(points, centers):
    """Paper Eq. 13 (Sec. 4.1): mean_i ||mean(C) - p_i|| over the SfM points p_i
    (official scene/__init__.py own_distance with center = -translate = mean camera centre)."""
    c = centers.mean(dim=0)
    return float(torch.linalg.vector_norm(points.double() - c.double(), dim=1).mean())


def threshold(call, start=1e-4, end=4e-4, intervals=145):
    """Gradient threshold of the `call`-th (1-based) densification (Sec. 4.2, Eq. 14).

    Official train.py:62-67,151: dense_factor = (end/start)^(1/dense_iters) with
    dense_iters = (densify_until - densify_from) // interval = 145, and `current *= dense_factor`
    before each densify_and_prune call, so call k uses start * (end/start)^(k/145), i.e. Eq. 14
    sampled at i/i_max = k/145.
    """
    return start * (end / start) ** (call / intervals)


def eq14(i, i_max, t_s=1e-4, t_f=4e-4):
    """Paper Eq. 14 written as printed (independent of `threshold`, used by probes)."""
    return math.exp(math.log(t_s) * (1 - i / i_max) + math.log(t_f) * i / i_max)


def depth_factor(depths, gamma_extent, power=2.0):
    """PixelGS depth scaling of the 2D gradient (paper Eq. 10; official rasterizer
    __init__.py:145 squares it): min(1, (depth / (gamma * e_cam)) ** power)."""
    return torch.clamp((depths / gamma_extent) ** power, max=1.0)


def revised_opacity(o):
    """Bulo et al. revised opacity, paper Eq. 8: 1 - sqrt(1 - o) (post-sigmoid)."""
    return 1.0 - torch.sqrt(1.0 - o)


def significance_prune_mask(sigma_old, split_old, opacity_now, min_opacity):
    """Significance-aware pruning (Sec. 4.3, Alg. 2), official densify_and_prune lines 446-462.

    sigma_old [N_old]: blend-weight sums accumulated since the last densification (pre-densify
    population); split_old [N_old] bool: which of them were split (and removed); opacity_now
    [N_now] post-sigmoid opacities after clone + split, laid out as [old non-split (in order),
    clones, split children]. Returns the [N_now] prune mask.
    """
    n_old = sigma_old.shape[0]
    keep_old = ~split_old
    scores = sigma_old[keep_old]
    low = opacity_now < min_opacity
    mask = torch.zeros_like(low)
    n_prune = int(low.sum())
    if scores.numel() == 0 or n_prune == 0:
        return mask
    # torch.quantile(scores, prunes / numbers) with numbers = N_old (official); clamp is ours
    q = torch.quantile(scores.float(), min(n_prune / n_old, 1.0))
    n_surv = scores.shape[0]
    mask[:n_surv] = scores < q
    return mask & low
