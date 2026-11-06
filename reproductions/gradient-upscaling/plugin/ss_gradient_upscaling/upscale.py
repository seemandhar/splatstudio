"""Image upscalers (pure torch, differentiable through autograd).

`spline_upscale` is the paper's gradient-aware bicubic spline (supp. "Spline Image
Interpolation"): on every low-res pixel cell the patch p(x,y) = [1 x x^2 x^3] A [1 y y^2 y^3]^T
with A = C^-1 F C^-T is fitted to the four corner values f and derivatives f_x, f_y, f_xy.
With C = [[1,0,0,0],[1,1,1,1],[0,1,0,0],[0,1,2,3]], [1 t t^2 t^3] C^-1 is the cubic Hermite
basis (h00, h01, h10, h11), so the patch is the tensor product of 1D Hermite interpolation and
is evaluated separably (x first, then y). Derivatives are in units of low-res pixels.

Coordinates: output pixel X (centre X+0.5 at output resolution) maps to the continuous low-res
pixel index u = (X+0.5) w/W - 0.5 (PyTorch `align_corners=False`, consistent with intrinsics
scaled by w/W). Cell index floor(u) is clamped to [0, w-2], so the outermost half pixel is a
cubic extrapolation of the edge cell (brush mirrors the fraction instead; see paper.md).
"""

import torch
import torch.nn.functional as F


def hermite_basis(t):
    t2, t3 = t * t, t * t * t
    return (2 * t3 - 3 * t2 + 1, -2 * t3 + 3 * t2, t3 - 2 * t2 + t, t3 - t2)  # h00 h01 h10 h11


def _axis(n_out, n_in, device, dtype):
    u = (torch.arange(n_out, device=device, dtype=dtype) + 0.5) * (n_in / n_out) - 0.5
    i0 = u.floor().clamp(0, n_in - 2).long()
    return i0, i0 + 1, u - i0


def spline_upscale(f, fx, fy, fxy, height, width):
    """f, fx, fy, fxy: [h, w, C] value and derivative images -> [height, width, C]."""
    h, w = f.shape[:2]
    if h < 2 or w < 2:
        raise ValueError("spline upscaling needs at least 2x2 input pixels")
    x0, x1, tx = _axis(width, w, f.device, f.dtype)
    y0, y1, ty = _axis(height, h, f.device, f.dtype)
    a00, a01, a10, a11 = (b[None, :, None] for b in hermite_basis(tx))
    # along x: value rows from (f, f_x), y-derivative rows from (f_y, f_xy)
    v = f[:, x0] * a00 + f[:, x1] * a01 + fx[:, x0] * a10 + fx[:, x1] * a11
    dv = fy[:, x0] * a00 + fy[:, x1] * a01 + fxy[:, x0] * a10 + fxy[:, x1] * a11
    b00, b01, b10, b11 = (b[:, None, None] for b in hermite_basis(ty))
    return v[y0] * b00 + v[y1] * b01 + dv[y0] * b10 + dv[y1] * b11


def central_differences(f):
    """Finite-difference estimates (f_x, f_y, f_xy) with edge replication, low-res pixel units."""
    p = F.pad(f.permute(2, 0, 1)[None], (1, 1, 1, 1), mode="replicate")[0].permute(1, 2, 0)
    fx = (p[1:-1, 2:] - p[1:-1, :-2]) * 0.5
    fy = (p[2:, 1:-1] - p[:-2, 1:-1]) * 0.5
    fxy = (p[2:, 2:] - p[2:, :-2] - p[:-2, 2:] + p[:-2, :-2]) * 0.25
    return fx, fy, fxy


def bicubic_upscale(f, height, width):
    """PyTorch differentiable bicubic (Keys a=-0.75), the paper's 'Bicubic' baseline."""
    x = f.permute(2, 0, 1)[None]
    return F.interpolate(x, size=(height, width), mode="bicubic",
                         align_corners=False)[0].permute(1, 2, 0)


def bilinear_upscale(f, height, width):
    x = f.permute(2, 0, 1)[None]
    return F.interpolate(x, size=(height, width), mode="bilinear",
                         align_corners=False)[0].permute(1, 2, 0)
