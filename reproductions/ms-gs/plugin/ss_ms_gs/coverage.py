"""Pixel coverage (C1), selective-rendering rule (C3) and the S_max/S_min range update (Eq. 6).

Numerics follow the official MS-GS rasterizer (github.com/JokerYan/MS-GS-rasterizer @ e7a2b4f,
cuda_rasterizer/forward.cu preprocessCUDA lines 205-283) and scene/gaussian_model.py
update_pixel_sizes of github.com/JokerYan/MS-GS @ 50e5040. Everything here is metadata
bookkeeping without gradients.
"""

import math

import torch
from gsplat.utils import normalized_quat_to_rotmat


@torch.no_grad()
def cov2d_undilated(means, quats, log_scales, viewmat, K, width, height):
    """3DGS computeCov2D (forward.cu:74-117) without the +0.3 low-pass term.

    Returns (a, b, c, z): the 2D covariance [[a, b], [b, c]] in pixels^2 and camera depth.
    """
    R = viewmat[:3, :3]
    t = means @ R.T + viewmat[:3, 3]
    fx, fy = K[0, 0], K[1, 1]
    tan_fovx = width / (2 * fx)
    tan_fovy = height / (2 * fy)
    z = t[:, 2]
    limx, limy = 1.3 * tan_fovx, 1.3 * tan_fovy
    tx = (t[:, 0] / z).clamp(-limx, limx) * z
    ty = (t[:, 1] / z).clamp(-limy, limy) * z
    zero = torch.zeros_like(z)
    J = torch.stack(
        (
            torch.stack((fx / z, zero, -(fx * tx) / (z * z)), -1),
            torch.stack((zero, fy / z, -(fy * ty) / (z * z)), -1),
        ),
        1,
    )  # [N, 2, 3]
    rot = normalized_quat_to_rotmat(torch.nn.functional.normalize(quats, dim=-1))
    M = rot * log_scales.exp()[:, None, :]  # R diag(s)
    sigma = M @ M.transpose(1, 2)  # Vrk
    T = J @ R  # J W
    cov = T @ sigma @ T.transpose(1, 2)
    return cov[:, 0, 0], cov[:, 0, 1], cov[:, 1, 1], z


@torch.no_grad()
def pixel_coverage(params, camera, opacity_level=1.0 / 255.0, near=0.2):
    """Per-Gaussian pixel coverage S_k (forward.cu:240-259).

    level_set = max(0, -2 log(opacity_level / sigma)); with the undilated conic Q = V^-1,
    S = min(sqrt(level_set / Q11), sqrt(level_set / Q22)) = min(sqrt(ls det / c), sqrt(ls det / a)).
    S = 0 for Gaussians failing 3DGS in_frustum (z <= 0.2) or with a degenerate covariance,
    which is the value the official kernel leaves in `pixel_sizes`.
    """
    a, b, c, z = cov2d_undilated(
        params["means"], params["quats"], params["scales"], camera["viewmat"], camera["K"],
        int(camera["width"]), int(camera["height"]),
    )
    det = a * c - b * b
    sigma = params["opacities"].sigmoid()
    level_set = (-2 * torch.log(opacity_level / sigma)).clamp_min(0)
    ok = (z > near) & (det > 0) & (a > 0) & (c > 0)
    dx = torch.sqrt((level_set * det / c).clamp_min(0))
    dy = torch.sqrt((level_set * det / a).clamp_min(0))
    s = torch.minimum(dx, dy)
    return torch.where(ok, s, torch.zeros_like(s))


@torch.no_grad()
def select(px, max_px, min_px, *, filter_small=True, filter_large=True, min_rel=0.5,
           max_rel=2.0, abs_threshold=2.0):
    """Selective rendering (paper Eq. 7 / Alg. 2 in the code form of forward.cu:261-283).

    rel = S / stored when stored > 0, else 1 (uninitialised range never culls). A Gaussian is
    culled when (rel_min < min_rel and S < abs_threshold) or rel_max > max_rel. As in the
    official kernel the large filter is nested inside the small one.
    """
    rel_min = torch.where(min_px > 0, px / min_px.clamp_min(1e-30), torch.ones_like(px))
    rel_max = torch.where(max_px > 0, px / max_px.clamp_min(1e-30), torch.ones_like(px))
    keep = torch.ones_like(px, dtype=torch.bool)
    if filter_small:
        keep &= ~((rel_min < min_rel) & (px < abs_threshold))
        if filter_large:
            keep &= ~(rel_max > max_rel)
    return keep


@torch.no_grad()
def update_ranges(max_px, min_px, level, visible, px, train_level, n_levels, decay_max=0.95,
                  decay_min=1.05):
    """Eq. 6 in the code form of gaussian_model.py:update_pixel_sizes (in place).

    Only Gaussians visible in this render whose level equals the training resolution index are
    updated. Level 0 never gets a maximum (finest never culled as too large); the coarsest
    training index never gets a minimum (coarsest never culled as too small).
    """
    mask = visible & (level == train_level)
    if train_level > 0:
        mx = max_px[mask] * decay_max
        max_px[mask] = torch.maximum(mx, px[mask])
    if train_level < n_levels - 1:
        mn = (min_px[mask] * decay_min).clamp_min(-1)
        p = px[mask]
        valid = p > 0
        min_px[mask] = torch.where(
            mn < 0, torch.where(valid, p, mn), torch.where(valid, torch.minimum(mn, p), mn)
        )
    return mask


def level_set_extent(sigma, opacity_level=1.0 / 255.0):
    """Analytic helper: half-extent factor sqrt(2 ln(sigma / opacity_level)) (0 if below)."""
    return math.sqrt(max(0.0, -2 * math.log(opacity_level / sigma)))
