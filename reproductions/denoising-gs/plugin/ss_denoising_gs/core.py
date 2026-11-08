"""Pure functions for Denoising-GS (arXiv 2605.14880). Equation numbers refer to the paper."""

import numpy as np
import torch
import torch.nn.functional as F
from gsplat.utils import normalized_quat_to_rotmat

# Per-scene budgets = the paper's own 3DGS-MCMC primitive counts (Table 2, Mip-NeRF 360).
# Denoising-GS counts are 0.9x these (+ the tiny SCR growth), so the MCMC cap is its budget.
SCENE_CAPS = {
    "bicycle": 5_900_000,
    "bonsai": 1_300_000,
    "counter": 1_200_000,
    "garden": 5_200_000,
    "kitchen": 1_800_000,
    "room": 1_500_000,
    "stump": 4_750_000,
}


def momentum_drift(momentum, grad_means, lr, alpha, beta1, denom=None, bias_correction=1.0):
    """Eqs. m2 / msgld. Returns (offset, new_momentum); m_t holds the gradients up to t-1 and the
    new momentum folds in the current gradient for the next step.

    denom None  -> literal raw drift  -alpha * lr * m_t   (numerically ~1e-5 of an Adam step)
    denom given -> Adam-normalised    -alpha * lr * m^_t / denom, m^ = m / bias_correction,
                   denom = sqrt(v^) + eps from the means optimizer (our default interpretation)."""
    if denom is None:
        offset = -alpha * lr * momentum
    else:
        offset = -alpha * lr * (momentum / bias_correction) / denom
    new_m = beta1 * momentum + (1.0 - beta1) * grad_means
    return offset, new_m


def denoise_term(grad_means, grad_scales, quats, sign_mode="descent", magnitude=None):
    """Eqs. m3-m4 and the R Delta-mu of Eq. m5 (world frame, before the beta2 factor).

    grad_means [N,3] world-frame mean gradient, grad_scales [N,3] scaling gradient (per local
    axis, selects the dominant axis), quats [N,4] wxyz. When the dominant (max |.|) axes of the
    local mean gradient and the scaling gradient coincide (axis j):
      sign_mode "descent" (default): Delta-mu_j = -|M_j| sign(dM^G_j)  -- same direction as the
          gradient update -lambda dM (Fig. 2, Sec. 4.3 text);
      sign_mode "literal": Delta-mu_j = +|M_j| sign(dM^G_j)  -- Eq. m4 as printed.
    M = magnitude [N,3] (default grad_scales, i.e. |dS_j|; alternative: the applied scale step)."""
    if sign_mode not in ("descent", "literal"):
        raise ValueError("sign_mode must be 'descent' or 'literal'")
    magnitude = grad_scales if magnitude is None else magnitude
    R = normalized_quat_to_rotmat(F.normalize(quats, dim=-1))  # columns = local axes
    g_local = torch.einsum("nji,nj->ni", R, grad_means)  # Eq. m3: R^T dM
    j_m = g_local.abs().argmax(dim=-1)
    j_s = grad_scales.abs().argmax(dim=-1)
    aligned = j_m == j_s
    j = j_m[:, None]
    sgn = torch.sign(g_local.gather(1, j))
    if sign_mode == "descent":
        sgn = -sgn
    mag = magnitude.gather(1, j).abs() * sgn  # Eq. m4
    dmu = torch.zeros_like(g_local).scatter_(1, j, mag * aligned[:, None].to(mag.dtype))
    return torch.einsum("nij,nj->ni", R, dmu), aligned


def uncertainty_from_sq_norms(sq_norm_sum):
    """Eq. f3: u_i = sum_j sigma_j(F_i). F_i = sum_v g g^T is PSD, so the singular values are its
    eigenvalues and their sum is trace(F_i) = sum_v ||g_v||^2 (accumulated by the caller)."""
    return sq_norm_sum


def prune_mask(u, fraction):
    """Remove the `fraction` primitives with the smallest u (highest uncertainty, Sec. 4.4)."""
    n = int(fraction * u.numel())
    mask = torch.zeros(u.numel(), dtype=torch.bool, device=u.device)
    if n > 0:
        mask[torch.topk(u, n, largest=False).indices] = True
    return mask


def knn_sparsity(points, k=3):
    """Eq. 4.1: d_i = mean over the k nearest neighbours (self excluded) of ||x_i - x_ij||^2."""
    from scipy.spatial import cKDTree

    x = points.detach().double().cpu().numpy()
    dist, _ = cKDTree(x).query(x, k=k + 1, workers=-1)
    d = (dist[:, 1:] ** 2).mean(axis=1)
    return torch.from_numpy(np.ascontiguousarray(d)).to(points.device, torch.float32)


def sparsest(d, fraction):
    n = max(1, int(fraction * d.numel()))
    return torch.topk(d, n, largest=True).indices


def split_in_two(opacities, scales):
    """Eqs. p6/p7 for N = 2 in a cancellation-free closed form.

    o_new = 1 - sqrt(1 - o) = o / (1 + sqrt(1 - o));  inner sum of Eq. p7 for N = 2 is
    2 o_new - o_new^2 / sqrt(2), so s_new / s = o / (o_new (2 - o_new / sqrt 2))
    = (1 + sqrt(1 - o)) / (2 - o_new / sqrt 2).  gsplat's compute_relocation evaluates
    1 - (1 - o)^(1/2) directly, which is 0 in float32 for o < ~1e-7 and gives s_new = inf
    (the cause of the counter run's nonfinite scales after an SCR split)."""
    root = torch.sqrt((1.0 - opacities).clamp_min(0.0))
    o_new = opacities / (1.0 + root)
    coeff = (1.0 + root) / (2.0 - o_new / 2**0.5)
    return o_new, scales * coeff[..., None]


def schedule(total_steps, prune_offset=5000, densify_every=500, densify_times=5):
    """Our guess (Appendix missing): prune at total - 5000, then SCR five times every 500 steps."""
    prune = total_steps - prune_offset
    return prune, [prune + densify_every * (i + 1) for i in range(densify_times)]
