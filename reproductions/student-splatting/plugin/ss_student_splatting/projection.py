"""Inria-style EWA projection in PyTorch, matching the official diff-t-rasterization
(`forward.cu preprocessCUDA` / `computeCov2D`, `backward.cu computeCov2DCUDA`):

  t      = W mu + t_w (camera frame); culled if t_z <= 0.2 (auxiliary.h in_frustum)
  J      = [[fx/tz, 0, -fx tx'/tz^2], [0, fy/tz, -fy ty'/tz^2]], tx' = clamp(tx/tz, +-1.3 tanfovx) tz
  Sigma2 = J W Sigma W^T J^T   (NO +0.3 low-pass: commented out in the official code)
  conic  = Sigma2^-1, backward with Inria's 1 / (det^2 + 1e-7) (custom autograd below)
  radius = ceil(e(nu) sqrt(lambda_max)), lambda_max = mid + sqrt(max(0.1, mid^2 - det)),
           e(nu) = official piecewise-linear interpolation of t critical values (3.0 for nu >= 13)
means2d use gsplat's pixel convention (pixel centres at +0.5), i.e. fx x/z + cx; the Inria
ndc2Pix convention (centres at integers, principal point at the image centre) is the same up to
that 0.5 shift for the centred-principal-point cameras of the benchmark datasets.
"""

import math

import torch
import torch.nn.functional as F

# forward.cu: t critical values at nu = 1..13 (two-sided 99.5 %), linearly interpolated
T_CRITICAL = (63.657, 9.925, 5.841, 4.604, 4.032, 3.707, 3.499, 3.355, 3.250, 3.169, 3.106,
              3.055, 3.012)


def empirical_factor(nu):
    """Official switch(floor(nu)) (forward.cu:246-281), verbatim: case k gives
    (T[k-1] - T[k]) (nu - k) + T[k] (T 0-based, T[0] = 63.657 at nu = 1), i.e. 9.925 at nu = 1
    rising towards 63.657 as nu -> 2 (the official ordering, kept as is);
    default (nu >= 13) 3.0."""
    table = torch.tensor(T_CRITICAL, device=nu.device, dtype=nu.dtype)
    vv = torch.floor(nu)
    k = vv.long().clamp(1, 12)
    frac = nu - vv
    interp = (table[k - 1] - table[k]) * frac + table[k]
    return torch.where((vv >= 1) & (vv <= 12), interp, torch.full_like(nu, 3.0))


class _InriaConic(torch.autograd.Function):
    """(a, b, c) of Sigma2 -> conic (c, -b, a)/det; backward = Inria computeCov2DCUDA lines
    209-218 (denom2inv = 1/(det^2 + 1e-7); conic.y gradient taken as half the full derivative,
    the Inria convention)."""

    @staticmethod
    def forward(ctx, a, b, c):
        det = a * c - b * b
        inv = 1.0 / det
        ctx.save_for_backward(a, b, c)
        return torch.stack((c * inv, -b * inv, a * inv), dim=-1)

    @staticmethod
    def backward(ctx, v):
        a, b, c = ctx.saved_tensors
        gx, gy, gz = v[..., 0], 0.5 * v[..., 1], v[..., 2]
        denom = a * c - b * b
        d2 = 1.0 / (denom * denom + 1e-7)
        da = d2 * (-c * c * gx + 2 * b * c * gy + (denom - a * c) * gz)
        dc = d2 * (-a * a * gz + 2 * a * b * gy + (denom - a * c) * gx)
        db = d2 * 2 * (b * c * gx - (denom + 2 * b * b) * gy + a * b * gz)
        return da, db, dc


def inria_conic(a, b, c):
    return _InriaConic.apply(a, b, c)


def quat_to_rotmat(q):
    """wxyz (normalised) -> R, same formula as computeCov3D."""
    r, x, y, z = q.unbind(-1)
    return torch.stack((
        1 - 2 * (y * y + z * z), 2 * (x * y - r * z), 2 * (x * z + r * y),
        2 * (x * y + r * z), 1 - 2 * (x * x + z * z), 2 * (y * z - r * x),
        2 * (x * z - r * y), 2 * (y * z + r * x), 1 - 2 * (x * x + y * y),
    ), dim=-1).reshape(*q.shape[:-1], 3, 3)


def covariance3d(quats, log_scales):
    """Sigma = R S S^T R^T with R from the normalised quaternion (official rotation_activation
    normalises; the kernel uses the quaternion as given)."""
    R = quat_to_rotmat(F.normalize(quats, dim=-1))
    M = R * torch.exp(log_scales)[..., None, :]
    return M @ M.transpose(-1, -2)


def project(means, quats, log_scales, nus, viewmat, K, width, height, tile_size=16,
            near=0.2, fov_clamp=1.3):
    """Returns means2d [N,2], conics [N,3], depths [N], radii [N,2] int32 (0 = culled), and the
    camera-frame means. Differentiable in means/quats/scales (radii are not)."""
    Rw, tw = viewmat[:3, :3], viewmat[:3, 3]
    pc = means @ Rw.T + tw
    tx, ty, tz = pc.unbind(-1)
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    limx = fov_clamp * (0.5 * width / fx)
    limy = fov_clamp * (0.5 * height / fy)
    valid = tz > near
    tzs = torch.where(valid, tz, torch.ones_like(tz))
    txc = torch.clamp(tx / tzs, -limx, limx) * tzs
    tyc = torch.clamp(ty / tzs, -limy, limy) * tzs
    zero = torch.zeros_like(tzs)
    J = torch.stack((fx / tzs, zero, -fx * txc / (tzs * tzs),
                     zero, fy / tzs, -fy * tyc / (tzs * tzs)), dim=-1).reshape(-1, 2, 3)
    T = J @ Rw
    cov2 = T @ covariance3d(quats, log_scales) @ T.transpose(-1, -2)
    a, b, c = cov2[:, 0, 0], cov2[:, 0, 1], cov2[:, 1, 1]
    det = a * c - b * b
    valid = valid & (det != 0)
    safe = lambda v, d: torch.where(valid, v, torch.full_like(v, d))
    conics = inria_conic(safe(a, 1.0), safe(b, 0.0), safe(c, 1.0))
    means2d = torch.stack((fx * tx / tzs + cx, fy * ty / tzs + cy), dim=-1)
    with torch.no_grad():
        mid = 0.5 * (a + c)
        lam = mid + torch.sqrt(torch.clamp(mid * mid - det, min=0.1))
        r = torch.ceil(empirical_factor(nus.detach()) * torch.sqrt(lam.clamp_min(0)))
        tw_, th_ = math.ceil(width / tile_size), math.ceil(height / tile_size)
        mx, my = means2d[:, 0], means2d[:, 1]
        # empty tile rectangle (gsplat isect_tiles formula) -> culled, like Inria getRect area 0
        x0 = torch.floor((mx - r) / tile_size).clamp(0, tw_)
        x1 = torch.ceil((mx + r) / tile_size).clamp(0, tw_)
        y0 = torch.floor((my - r) / tile_size).clamp(0, th_)
        y1 = torch.ceil((my + r) / tile_size).clamp(0, th_)
        valid = valid & torch.isfinite(r) & (r > 0) & ((x1 - x0) * (y1 - y0) > 0)
        radii = torch.where(valid, r, torch.zeros_like(r)).to(torch.int32)[:, None].repeat(1, 2)
    return {"means2d": means2d, "conics": conics, "depths": tz, "radii": radii, "valid": valid,
            "camera_means": pc}
