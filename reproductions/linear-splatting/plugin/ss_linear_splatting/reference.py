"""Dense (every Gaussian x every pixel) float64 PyTorch reference of the 3DLS rasterizer.

Written independently of the CUDA kernel from the paper's equations plus gsplat's numerics
(pixel centres at +0.5, alpha clamp 0.999 whose gradient is zero above the clamp, skip
alpha < 1/255, front-to-back by depth, stop when the next transmittance <= 1e-4).
Differentiating it with autograd gives the exact gradient; `ags=True` replaces dk/dD by
dk/dD * exp(-D^2) (paper Eq. 13-14) with a straight-through construction.
"""

import torch

ALPHA_THRESHOLD = 1.0 / 255.0


def kernel_value(D, lam, ags):
    """k = max(0, 1 - D/lam) (Eq. 12). With AGS coefficient c = float(ags) > 0 the forward value
    is unchanged and d k / d D is scaled by omega = exp(-c D^2) (Eq. 13: c = 1, D unscaled)."""
    c = float(ags)
    if c > 0:
        omega = torch.exp(-c * D.detach() ** 2)
        D = D.detach() + omega * (D - D.detach())
    return 1.0 - D / lam


def dense_render(means2d, conics, colors, opacities, depths, width, height, lam=2.5, ags=False,
                 kernel="linear"):
    """means2d [N,2], conics [N,3], colors [N,C], opacities [N], depths [N] -> (rgb [H,W,C],
    alpha [H,W,1]) without background. kernel 'gaussian' gives gsplat's exp(-sigma)."""
    order = torch.argsort(depths)
    m, q, c, o = means2d[order], conics[order], colors[order], opacities[order]
    ys, xs = torch.meshgrid(torch.arange(height, dtype=m.dtype, device=m.device) + 0.5,
                            torch.arange(width, dtype=m.dtype, device=m.device) + 0.5,
                            indexing="ij")
    dx = m[:, 0, None, None] - xs  # [N,H,W]
    dy = m[:, 1, None, None] - ys
    sigma = 0.5 * (q[:, 0, None, None] * dx * dx + q[:, 2, None, None] * dy * dy) + \
        q[:, 1, None, None] * dx * dy
    if kernel == "linear":
        D = torch.sqrt((2.0 * sigma).clamp_min(1e-24))  # Eq. 11
        k = kernel_value(D, lam, ags)
        support = (sigma >= 0) & (k > 0)
    elif kernel == "gaussian":
        k = torch.exp(-sigma)
        support = sigma >= 0
    else:
        raise ValueError(kernel)
    raw = o[:, None, None] * k
    alpha = torch.clamp(raw, max=0.999)  # gradient 0 above the clamp = gsplat's gate
    valid = support & (alpha.detach() >= ALPHA_THRESHOLD)
    alpha = torch.where(valid, alpha, torch.zeros_like(alpha))
    one_minus = 1.0 - alpha
    T_after = torch.cumprod(one_minus, dim=0)  # transmittance after each Gaussian
    # T_after is non-increasing, so 'next_T > 1e-4' holds for a prefix: exactly gsplat's stop
    keep = (T_after.detach() > 1e-4)
    alpha = torch.where(keep, alpha, torch.zeros_like(alpha))
    T_after = torch.cumprod(1.0 - alpha, dim=0)
    T_before = torch.cat((torch.ones_like(T_after[:1]), T_after[:-1]), dim=0)
    w = alpha * T_before  # [N,H,W]
    rgb = torch.einsum("nhw,nc->hwc", w, c)
    return rgb, (1.0 - T_after[-1])[..., None]
