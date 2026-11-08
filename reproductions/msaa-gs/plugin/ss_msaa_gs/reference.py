"""Dense (every Gaussian x every sample) float64 PyTorch reference of the MSAA rasterizer.

Written independently of the CUDA kernel from paper Eq. 2-5 plus gsplat's numerics (alpha clamp
0.999 whose gradient is zero above the clamp, skip alpha < 1/255, front-to-back by depth, stop
when the next transmittance <= 1e-4). Each sample position is composited separately; the pixel
value is the mean over samples. Autograd gives the exact gradient.
"""

import torch

ALPHA_THRESHOLD = 1.0 / 255.0


def composite_at(means2d, conics, colors, opacities, depths, xs, ys):
    """Single-sample 3DGS compositing at positions xs, ys [H,W] -> (rgb [H,W,C], alpha [H,W,1])."""
    order = torch.argsort(depths)
    m, q, c, o = means2d[order], conics[order], colors[order], opacities[order]
    dx = m[:, 0, None, None] - xs  # [N,H,W]
    dy = m[:, 1, None, None] - ys
    sigma = 0.5 * (q[:, 0, None, None] * dx * dx + q[:, 2, None, None] * dy * dy) + \
        q[:, 1, None, None] * dx * dy
    alpha = torch.clamp(o[:, None, None] * torch.exp(-sigma), max=0.999)
    valid = (sigma >= 0) & (alpha.detach() >= ALPHA_THRESHOLD)
    alpha = torch.where(valid, alpha, torch.zeros_like(alpha))
    keep = torch.cumprod(1.0 - alpha, dim=0).detach() > 1e-4  # a prefix: gsplat's stop rule
    alpha = torch.where(keep, alpha, torch.zeros_like(alpha))
    T_after = torch.cumprod(1.0 - alpha, dim=0)
    T_before = torch.cat((torch.ones_like(T_after[:1]), T_after[:-1]), dim=0)
    rgb = torch.einsum("nhw,nc->hwc", alpha * T_before, c)
    return rgb, (1.0 - T_after[-1])[..., None]


def dense_msaa(means2d, conics, colors, opacities, depths, width, height, offsets):
    """Eq. 4-5: mean over sample offsets (dx, dy) in [0,1]^2 of composites at (j+dx, i+dy)."""
    jj, ii = torch.meshgrid(torch.arange(width, dtype=means2d.dtype, device=means2d.device),
                            torch.arange(height, dtype=means2d.dtype, device=means2d.device),
                            indexing="xy")
    rgbs, alphas = [], []
    for dx, dy in offsets:
        rgb, alpha = composite_at(means2d, conics, colors, opacities, depths, jj + dx, ii + dy)
        rgbs.append(rgb)
        alphas.append(alpha)
    return torch.stack(rgbs).mean(0), torch.stack(alphas).mean(0)
