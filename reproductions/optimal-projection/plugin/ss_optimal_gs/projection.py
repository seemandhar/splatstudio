"""Optimal (tangent-plane) projection of 3D Gaussians, Sec. 5.1 Eq. 11-15 and Q of Eq. 18.

Ported from the official op43dgs pinhole rasterizer (forward.cu `computeCov2D`, backward.cu
`computeCov2DCUDA`). PyTorch autograd carries every gradient except, by default
(`official_cov_grad=True`), d J / d t, which uses the official 18 closed-form coefficients
verbatim (including its `j12_x` typo, see paper.md Sec. 7).
"""

import torch
from gsplat.cuda._wrapper import quat_scale_to_covar_preci


def tangent_jacobian_exact(t, fpx, fpy):
    """J = diag(f'x, f'y) [e_u; e_v] / |t|  ([N,2,3]) for camera-space means t [N,3].

    This is Q J_p (Eq. 14/15 left-multiplied by Q of Eq. 18, rows 1-2) scaled by f'; with
    theta = atan2(-mu_y, sqrt(mu_x^2+mu_z^2)), phi = atan2(mu_x, mu_z):
    e_u = (cos phi, 0, -sin phi), e_v = (sin phi sin theta, cos theta, sin theta cos phi)."""
    x, y, z = t.unbind(-1)
    rho = torch.sqrt(x * x + z * z)
    r = torch.sqrt(x * x + y * y + z * z)
    zero = torch.zeros_like(x)
    ju = torch.stack((z / (rho * r), zero, -x / (rho * r)), -1) * fpx
    jv = torch.stack((-x * y / (rho * r * r), rho / (r * r), -y * z / (rho * r * r)), -1) * fpy
    return torch.stack((ju, jv), -2)


def official_jacobian_coefficients(t, fx, fy):
    """d J_ik / d t_{x,y,z} transcribed verbatim from backward.cu computeCov2DCUDA (lines
    331-365 of the official file; fx, fy there are max(512, focal)). Returns [N,2,3,3] with
    index [i, k, axis]."""
    x, y, z = t.unbind(-1)
    x2, y2, z2 = x * x, y * y, z * z
    j00_x = fx * x * z * (-2.0 * x2 - y2 - 2.0 * z2) / (x2 + z2) ** 1.5 / (x2 + y2 + z2) ** 1.5
    j01_x = torch.zeros_like(x)
    j02_x = (fx * (x2 * x2 - y2 * z2 - z2 * z2) / (x2 + z2) ** 0.5 / (x2 + y2 + z2) ** 0.5
             / (x2 * x2 + x2 * y2 + 2.0 * x2 * z2 + y2 * z2 + z2 * z2))
    j10_x = (fy * y * (2.0 * x2 * x2 + x2 * z2 - y2 * z2 - z2 * z2) / (x2 + z2) ** 0.5
             / (x2 * x2 * x2 + 2.0 * x2 * x2 * y2 + 3.0 * x2 * x2 * z2 + x2 * y2 * y2
                + 4.0 * x2 * y2 * z2 + 3.0 * x2 * z2 * z2 + y2 * y2 * z2 + 2.0 * y2 * z2 * z2
                + z2 * z2 * z2))
    j11_x = (fy * x * (-x2 + y2 - z2) / (x2 + z2) ** 0.5
             / (x2 * x2 + 2.0 * x2 * y2 + 2.0 * x2 * z2 + y2 * y2 + 2.0 * y2 * z2 + z2 * z2))
    # official typo: `2.f * y2 * (x2 * z2)` (exact derivative has x2 + z2, cf. j10_z)
    j12_x = (fy * x * y * z * (2.0 * y2 * (x2 * z2) + y2 * (x2 + y2 + z2)
                               + 2.0 * (x2 + z2) ** 2.0 + (x2 + z2) * (x2 + y2 + z2))
             / (x2 + z2) ** 1.5 / (x2 + y2 + z2) ** 3.0)
    j00_y = -fx * y * z / (x2 + z2) ** 0.5 / (x2 + y2 + z2) ** 1.5
    j01_y = torch.zeros_like(x)
    j02_y = fx * x * y / (x2 + z2) ** 0.5 / (x2 + y2 + z2) ** 1.5
    quad = x2 * x2 + 2.0 * x2 * y2 + 2.0 * x2 * z2 + y2 * y2 + 2.0 * y2 * z2 + z2 * z2
    j10_y = fy * x * (-x2 + y2 - z2) / (x2 + z2) ** 0.5 / quad
    j11_y = -2.0 * fy * y * (x2 + z2) ** 0.5 / quad
    j12_y = fy * z * (-x2 + y2 - z2) / (x2 + z2) ** 0.5 / quad
    j00_z = (fx * (x2 * x2 + x2 * y2 - z2 * z2) / (x2 + z2) ** 0.5 / (x2 + y2 + z2) ** 0.5
             / (x2 * x2 + x2 * y2 + 2.0 * x2 * z2 + y2 * z2 + z2 * z2))
    j01_z = torch.zeros_like(x)
    j02_z = fx * x * z * (2.0 * x2 + y2 + 2.0 * z2) / (x2 + z2) ** 1.5 / (x2 + y2 + z2) ** 1.5
    j10_z = (fy * x * y * z * (2.0 * y2 * (x2 + z2) + y2 * (x2 + y2 + z2)
                               + 2.0 * (x2 + z2) ** 2.0 + (x2 + z2) * (x2 + y2 + z2))
             / (x2 + z2) ** 1.5 / (x2 + y2 + z2) ** 3.0)
    j11_z = fy * z * (-x2 + y2 - z2) / (x2 + z2) ** 0.5 / quad
    j12_z = (fy * y * (-x2 * x2 - x2 * y2 + x2 * z2 + 2.0 * z2 * z2) / (x2 + z2) ** 0.5
             / (x2 * x2 * x2 + 2.0 * x2 * x2 * y2 + 3.0 * x2 * x2 * z2 + x2 * y2 * y2
                + 4.0 * x2 * y2 * z2 + 3.0 * x2 * z2 * z2 + y2 * y2 * z2 + 2.0 * y2 * z2 * z2
                + z2 * z2 * z2))
    rows = [
        [(j00_x, j00_y, j00_z), (j01_x, j01_y, j01_z), (j02_x, j02_y, j02_z)],
        [(j10_x, j10_y, j10_z), (j11_x, j11_y, j11_z), (j12_x, j12_y, j12_z)],
    ]
    return torch.stack([torch.stack([torch.stack(c, -1) for c in row], -2) for row in rows], -3)


class _OfficialJacobian(torch.autograd.Function):
    """Forward: exact J. Backward: official closed-form d J / d t (backward.cu)."""

    @staticmethod
    def forward(ctx, t, fpx, fpy):
        ctx.save_for_backward(t)
        ctx.f = (fpx, fpy)
        return tangent_jacobian_exact(t, fpx, fpy)

    @staticmethod
    def backward(ctx, gJ):
        (t,) = ctx.saved_tensors
        coef = official_jacobian_coefficients(t, *ctx.f)  # [N,2,3,3]
        return (gJ[..., None] * coef).sum((-3, -2)), None, None


def tangent_jacobian(t, fpx, fpy, official_grad=True):
    if official_grad:
        return _OfficialJacobian.apply(t, fpx, fpy)
    return tangent_jacobian_exact(t, fpx, fpy)


def ewa_jacobian(t, fx, fy, tan_fovx, tan_fovy):
    """3DGS EWA J (computeNaiveCov2D) with the 1.3 tan(fov) clamp; [N,2,3]."""
    x, y, z = t.unbind(-1)
    limx, limy = 1.3 * tan_fovx, 1.3 * tan_fovy
    x = (x / z).clamp(-limx, limx) * z
    y = (y / z).clamp(-limy, limy) * z
    zero = torch.zeros_like(z)
    return torch.stack((torch.stack((fx / z, zero, -fx * x / (z * z)), -1),
                        torch.stack((zero, fy / z, -fy * y / (z * z)), -1)), -2)


def _sym2(J, cov, eps):
    c = J @ cov @ J.transpose(-1, -2)
    return c[..., 0, 0] + eps, c[..., 0, 1], c[..., 1, 1] + eps


def project_gaussians(means, quats, log_scales, viewmat, K, width, height, mode="optimal",
                      focal_floor=512.0, eps2d=0.3, near=0.2, official_cov_grad=True):
    """Returns dict with means2d [N,2] (pixels), conics [N,3], depths [N], radii [N,2] int32,
    valid [N] bool. Invisible Gaussians get radius 0 and finite dummy values."""
    R, T = viewmat[:3, :3], viewmat[:3, 3]
    t = means @ R.T + T
    valid = t[:, 2] > near  # auxiliary.h in_frustum: p_view.z <= 0.2 culled
    dummy = torch.tensor([0.0, 0.0, 1.0], device=t.device, dtype=t.dtype)
    ts = torch.where(valid[:, None], t, dummy)
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    means2d = torch.stack((fx * ts[:, 0] / ts[:, 2] + cx, fy * ts[:, 1] / ts[:, 2] + cy), -1)
    covar = quat_scale_to_covar_preci(quats, log_scales.exp(), compute_covar=True,
                                      compute_preci=False, triu=False)[0]
    cov_cam = R @ covar @ R.T
    tan_fovx, tan_fovy = 0.5 * width / float(fx), 0.5 * height / float(fy)
    J_ewa = ewa_jacobian(ts, fx, fy, tan_fovx, tan_fovy)
    if mode == "optimal":
        fpx, fpy = max(focal_floor, float(fx)), max(focal_floor, float(fy))
        J = tangent_jacobian(ts, fpx, fpy, official_cov_grad)
    elif mode == "ewa":
        J = J_ewa
    else:
        raise ValueError(f"unknown projection mode {mode!r}")
    a, b, c = _sym2(J, cov_cam, eps2d)
    det = a * c - b * b
    valid = valid & (det > 0)
    det_s = torch.where(valid, det, torch.ones_like(det))
    conics = torch.stack((c / det_s, -b / det_s, a / det_s), -1)
    with torch.no_grad():
        # forward.cu preprocess: extent from the naive EWA covariance's mid and the projected
        # covariance's determinant, radius = ceil(3 sqrt(lambda_max))
        an, _, cn = _sym2(J_ewa, cov_cam, eps2d)
        mid = 0.5 * (an + cn)
        lam1 = mid + torch.sqrt((mid * mid - det).clamp_min(0.1))
        radius = torch.ceil(3.0 * torch.sqrt(lam1.clamp_min(0)))
        radius = torch.where(valid, radius, torch.zeros_like(radius))
        radii = torch.stack((radius, radius), -1).to(torch.int32)
    return {"means2d": means2d, "conics": conics, "depths": ts[:, 2], "radii": radii,
            "valid": valid, "t": t}
