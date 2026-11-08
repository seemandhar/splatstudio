"""Dense float64 PyTorch reference of the official t rasterizer, written from the paper equations
and the official renderCUDA numerics, independently of the CUDA kernel.

Pixel centres at +0.5; base = d^T conic d; skipped where base <= 0 or |alpha| < 1/255;
alpha = clamp(o (1 + base/nu)^(-(nu+2)/2), -0.99, 0.99) with straight-through gradient (official
backward ignores the clamp); dL/dnu official = derivative with the exponent's nu held fixed
(exact=True keeps the full derivative); only pixels inside the gsplat tile rectangle of each
splat (radius from projection) are touched; front-to-back by depth; a pixel stops at the first
splat whose T (1 - alpha) < 1e-4 (excluded), so T is tracked through negative alphas.
"""

import math

import torch


def tile_mask(means2d, radii, width, height, tile_size):
    tw, th = math.ceil(width / tile_size), math.ceil(height / tile_size)
    r = radii.to(means2d.dtype)
    x0 = torch.floor((means2d[:, 0] - r) / tile_size).clamp(0, tw)
    x1 = torch.ceil((means2d[:, 0] + r) / tile_size).clamp(0, tw)
    y0 = torch.floor((means2d[:, 1] - r) / tile_size).clamp(0, th)
    y1 = torch.ceil((means2d[:, 1] + r) / tile_size).clamp(0, th)
    tx = torch.arange(width, device=means2d.device) // tile_size
    ty = torch.arange(height, device=means2d.device) // tile_size
    in_x = (tx[None] >= x0[:, None]) & (tx[None] < x1[:, None])
    in_y = (ty[None] >= y0[:, None]) & (ty[None] < y1[:, None])
    return in_y[:, :, None] & in_x[:, None, :] & (radii[:, None, None] > 0)


def dense_render(means2d, conics, colors, opacities, nus, depths, radii, width, height,
                 tile_size=16, kernel="t", exact=False, clip=0.99):
    """-> rgb [H,W,C] (black background), 1-T [H,W,1]. kernel 'gaussian' = gsplat's
    exp(-base/2) with alpha clamp 0.999 and 1/255 skip (validates this reference vs gsplat)."""
    order = torch.argsort(depths)
    m, q, c, o, nu, rad = (t[order] for t in (means2d, conics, colors, opacities, nus, radii))
    ys, xs = torch.meshgrid(torch.arange(height, dtype=m.dtype, device=m.device) + 0.5,
                            torch.arange(width, dtype=m.dtype, device=m.device) + 0.5,
                            indexing="ij")
    dx = m[:, 0, None, None] - xs
    dy = m[:, 1, None, None] - ys
    base = q[:, 0, None, None] * dx * dx + q[:, 2, None, None] * dy * dy + \
        2.0 * q[:, 1, None, None] * dx * dy
    inside = tile_mask(m, rad, width, height, tile_size)
    if kernel == "t":
        n = nu[:, None, None]
        expo = -(n + 2) / 2 if exact else -(n.detach() + 2) / 2
        support = base > 0
        G = torch.pow(1.0 + torch.where(support, base, torch.ones_like(base)) / n, expo)
        raw = o[:, None, None] * G
        clipped = raw.clamp(-clip, clip)
        alpha = raw + (clipped - raw).detach()  # straight-through
        valid = support & (clipped.detach().abs() >= 1.0 / 255.0) & inside
    elif kernel == "gaussian":
        raw = o[:, None, None] * torch.exp(-0.5 * base)
        alpha = torch.clamp(raw, max=0.999)
        valid = (base >= 0) & (alpha.detach() >= 1.0 / 255.0) & inside
    else:
        raise ValueError(kernel)
    alpha = torch.where(valid, alpha, torch.zeros_like(alpha))
    T_after = torch.cumprod(1.0 - alpha, dim=0)
    ok = (T_after.detach() >= 1e-4).to(torch.int64)
    keep = torch.cumprod(ok, dim=0).bool()  # stop at the first failing splat
    alpha = torch.where(keep, alpha, torch.zeros_like(alpha))
    T_after = torch.cumprod(1.0 - alpha, dim=0)
    T_before = torch.cat((torch.ones_like(T_after[:1]), T_after[:-1]), dim=0)
    w = alpha * T_before
    rgb = torch.einsum("nhw,nc->hwc", w, c)
    return rgb, (1.0 - T_after[-1])[..., None]
