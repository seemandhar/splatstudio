"""Pure torch pieces of Weighted Sum Rendering (arXiv 2410.18931); CPU-testable."""

import math

import torch

# 3DGS utils/sh_utils.py constants (identical to gsplat's spherical_harmonics)
C0 = 0.28209479177387814
C1 = 0.4886025119029199
C2 = (1.0925484305920792, -1.0925484305920792, 0.31539156525252005, -1.0925484305920792,
      0.5462742152960396)
C3 = (-0.5900435899266435, 2.890611442640554, -0.4570457994644658, 0.3731763325901154,
      -0.4570457994644658, 1.445305721320277, -0.5900435899266435)


def eval_sh_scalar(degree, coeffs, dirs):
    """Real SH expansion of one channel: coeffs [N,K>=(degree+1)^2], dirs [N,3] (normalised
    here). Same basis/sign convention as 3DGS eval_sh and gsplat spherical_harmonics."""
    if not 0 <= degree <= 3:
        raise ValueError("SH degree must be 0..3")
    d = torch.nn.functional.normalize(dirs, dim=-1)
    x, y, z = d[..., 0], d[..., 1], d[..., 2]
    c = coeffs
    out = C0 * c[..., 0]
    if degree > 0:
        out = out - C1 * y * c[..., 1] + C1 * z * c[..., 2] - C1 * x * c[..., 3]
    if degree > 1:
        xx, yy, zz, xy, yz, xz = x * x, y * y, z * z, x * y, y * z, x * z
        out = (out + C2[0] * xy * c[..., 4] + C2[1] * yz * c[..., 5]
               + C2[2] * (2.0 * zz - xx - yy) * c[..., 6] + C2[3] * xz * c[..., 7]
               + C2[4] * (xx - yy) * c[..., 8])
        if degree > 2:
            out = (out + C3[0] * y * (3 * xx - yy) * c[..., 9] + C3[1] * xy * z * c[..., 10]
                   + C3[2] * y * (4 * zz - xx - yy) * c[..., 11]
                   + C3[3] * z * (2 * zz - 3 * xx - 3 * yy) * c[..., 12]
                   + C3[4] * x * (4 * zz - xx - yy) * c[..., 13]
                   + C3[5] * z * (xx - yy) * c[..., 14]
                   + C3[6] * x * (xx - 3 * yy) * c[..., 15])
    return out


def view_dependent_opacity(degree, opa_sh0, opa_shN, dirs, offset=0.1):
    """Eq. 11: u_i = Y(view dir, H_i), unclamped. `offset` is the initial value (0.1, the 3DGS
    opacity init; the unofficial code stores it as SH-DC 0.1/C0): u = offset + Y(dir; H)."""
    coeffs = torch.cat((opa_sh0, opa_shN), dim=-1)
    return offset + eval_sh_scalar(degree, coeffs, dirs)


def depth_weight(kind, depths, v, sigma):
    """Eq. 10 LC-WSR w = max(0, 1 - d/sigma) v ;  Eq. 8 DIR-WSR (v only, as w = 1 with the
    per-Gaussian factor kept so the ablation isolates the depth term)."""
    if kind == "lc":
        return torch.clamp_min(1.0 - depths / sigma, 0.0) * v
    if kind == "dir":
        return v.expand_as(depths)
    raise ValueError(f"unknown weight function {kind!r} (lc, dir)")


def compose(feat_sum, w_sum, w_bg, bg_color, den_eps=1e-4):
    """Eq. 7: C = (c_B w_B + sum c_i a_i w_i) / (w_B + sum a_i w_i). feat_sum [H,W,3],
    w_sum [H,W,1]. Returns colour and the non-background weight fraction."""
    den = torch.clamp_min(w_sum + w_bg, den_eps)
    return (feat_sum + w_bg * bg_color) / den, w_sum / den


def expon_lr(step, lr_init, lr_final, max_steps):
    """3DGS get_expon_lr_func with lr_delay_steps = 0 (log-linear interpolation)."""
    t = min(max(step / max_steps, 0.0), 1.0)
    return math.exp(math.log(lr_init) * (1 - t) + math.log(lr_final) * t)
