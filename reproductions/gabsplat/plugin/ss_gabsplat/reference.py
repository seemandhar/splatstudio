"""Dense (every primitive x every pixel) float64 PyTorch reference of the 3DGabSplat rasterizer.

Written from the paper's Eq. 12-13 plus gsplat's numerics (pixel centres +0.5, alpha clamp 0.999
whose gradient is zero above the clamp, skip alpha < 1/255, front-to-back by depth, stop when the
next transmittance <= 1e-4). Autograd of it gives the exact gradients the CUDA backward must match.
"""

import math

import torch

ALPHA_THRESHOLD = 1.0 / 255.0


def dense_render(means2d, conics, colors, opacities, depths, width, height, gabs=None):
    """means2d [N,2], conics [N,3], colors [N,C], opacities [N], depths [N], gabs [N,3F] or None
    (pure Gaussian) -> (rgb [H,W,C], alpha [H,W,1]) without background."""
    order = torch.argsort(depths)
    m, q, c, o = means2d[order], conics[order], colors[order], opacities[order]
    ys, xs = torch.meshgrid(torch.arange(height, dtype=m.dtype, device=m.device) + 0.5,
                            torch.arange(width, dtype=m.dtype, device=m.device) + 0.5,
                            indexing="ij")
    dx = m[:, 0, None, None] - xs  # [N,H,W]
    dy = m[:, 1, None, None] - ys
    sigma = 0.5 * (q[:, 0, None, None] * dx * dx + q[:, 2, None, None] * dy * dy) + \
        q[:, 1, None, None] * dx * dy
    k = torch.exp(-sigma)
    if gabs is not None:
        g = gabs[order].reshape(len(order), -1, 3)
        mod = torch.ones_like(k)
        for i in range(g.shape[1]):
            fx, fy, w = (g[:, i, j, None, None] for j in range(3))
            mod = mod + w * (torch.cos(2 * math.pi * (fx * dx + fy * dy)) - 1.0)  # Eq. 12
        k = k * mod
    raw = o[:, None, None] * k
    alpha = torch.clamp(raw, max=0.999)  # gradient 0 above the clamp = gsplat's gate
    valid = (sigma >= 0) & (alpha.detach() >= ALPHA_THRESHOLD)
    alpha = torch.where(valid, alpha, torch.zeros_like(alpha))
    T_after = torch.cumprod(1.0 - alpha, dim=0)
    keep = T_after.detach() > 1e-4  # prefix: gsplat's early stop
    alpha = torch.where(keep, alpha, torch.zeros_like(alpha))
    T_after = torch.cumprod(1.0 - alpha, dim=0)
    T_before = torch.cat((torch.ones_like(T_after[:1]), T_after[:-1]), dim=0)
    w = alpha * T_before
    rgb = torch.einsum("nhw,nc->hwc", w, c)
    return rgb, (1.0 - T_after[-1])[..., None]
