"""Dense float64 PyTorch references, written from the paper equations and the official code's
numerics, independently of the CUDA kernels.

Rasterizer: pixel centres at +0.5; r^2 = d^T conic d; support r^2 < 1; alpha =
min(0.999, o (1 - r^2)^beta) whose gradient is zero above the clamp (official bwd gate); no 1/255
skip; front-to-back by depth; stop when the next transmittance <= 1e-4.
Spherical Beta: c0 + sum_m [dot > 0] dot^(4 e^{b_m}) c_m with dot = <v/|v|, R(theta, phi)>.
"""

import torch


def dense_render(means2d, conics, colors, opacities, betas, depths, width, height,
                 kernel="beta"):
    """means2d [N,2], conics [N,3], colors [N,C], opacities [N], betas [N] (activated), depths [N]
    -> (rgb [H,W,C], alpha [H,W,1]) without background. kernel 'gaussian' = gsplat's exp(-r^2/2)
    with the 1/255 skip (used to validate this reference against gsplat)."""
    order = torch.argsort(depths)
    m, q, c, o, b = means2d[order], conics[order], colors[order], opacities[order], betas[order]
    ys, xs = torch.meshgrid(torch.arange(height, dtype=m.dtype, device=m.device) + 0.5,
                            torch.arange(width, dtype=m.dtype, device=m.device) + 0.5,
                            indexing="ij")
    dx = m[:, 0, None, None] - xs
    dy = m[:, 1, None, None] - ys
    r2 = q[:, 0, None, None] * dx * dx + q[:, 2, None, None] * dy * dy + \
        2.0 * q[:, 1, None, None] * dx * dy
    if kernel == "beta":
        support = (r2 >= 0) & (r2 < 1)
        base = torch.where(support, 1.0 - r2, torch.ones_like(r2))
        k = torch.pow(base, b[:, None, None])
        raw = o[:, None, None] * k
        alpha = torch.clamp(raw, max=0.999)
        valid = support
    elif kernel == "gaussian":
        raw = o[:, None, None] * torch.exp(-0.5 * r2)
        alpha = torch.clamp(raw, max=0.999)
        valid = (r2 >= 0) & (alpha.detach() >= 1.0 / 255.0)
    else:
        raise ValueError(kernel)
    alpha = torch.where(valid, alpha, torch.zeros_like(alpha))
    T_after = torch.cumprod(1.0 - alpha, dim=0)
    keep = T_after.detach() > 1e-4  # prefix: exactly the CUDA early stop
    alpha = torch.where(keep, alpha, torch.zeros_like(alpha))
    T_after = torch.cumprod(1.0 - alpha, dim=0)
    T_before = torch.cat((torch.ones_like(T_after[:1]), T_after[:-1]), dim=0)
    w = alpha * T_before
    rgb = torch.einsum("nhw,nc->hwc", w, c)
    return rgb, (1.0 - T_after[-1])[..., None]


def spherical_beta_ref(dirs, c0, coeffs):
    """dirs [N,3], c0 [N,3], coeffs [N,M,6] (colour already activated)."""
    v = dirs / dirs.norm(dim=-1, keepdim=True)
    theta, phi, b = coeffs[..., 3], coeffs[..., 4], coeffs[..., 5]
    R = torch.stack((torch.sin(theta) * torch.cos(phi), torch.sin(theta) * torch.sin(phi),
                     torch.cos(theta)), dim=-1)
    dot = (R * v[:, None, :]).sum(-1)
    pos = dot > 0
    term = torch.where(pos, torch.pow(torch.where(pos, dot, torch.ones_like(dot)),
                                      4.0 * torch.exp(b)), torch.zeros_like(dot))
    return c0 + (term[..., None] * coeffs[..., :3]).sum(dim=1)
