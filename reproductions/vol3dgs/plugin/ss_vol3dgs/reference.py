"""Pure-torch Vol3DGS formulas (CPU/GPU, any dtype): density reparameterisation, camera-space
Gaussian quantities, and a dense (tile-free) float64 reference of the volumetric renderer.

Sources: official code chinmay0301ucsd/Vol3DGS (utils/general_utils.py alpha2density /
density2alpha, reparam "ours") and slang-gaussian-rasterization utils.slang:evaluate_splat_volr.
"""

import torch
import torch.nn.functional as F

SQRT_2PI = 2.506  # the constant the official code uses for sqrt(2 pi)
EPS = 1e-7
MAX_ALPHA = 0.99
ALPHA_THRESHOLD = 1.0 / 255.0
T_MIN = 1e-4


def alpha2density(theta, scales):
    """Code `alpha2density` ("ours"): kappa = -log(1 - 0.99 theta) * mean(1/s) / 2.506.

    Paper Eq. 20 writes (1/3) sum(1/s) without the 1/sqrt(2 pi); the code divides by 2.506 so a
    centre ray through an isotropic Gaussian has alpha = 0.99 theta."""
    return -torch.log(1.0 - 0.99 * theta) * (1.0 / scales).mean(dim=-1) / SQRT_2PI


DENSITY2ALPHA_MODES = ("official", "inverse")


def density2alpha(kappa, scales, mode="official"):
    """Code `density2alpha` ("ours"), clamped to [0, 0.99999].

    mode "official" (default) mirrors general_utils.py:56 exactly as written, where operator
    precedence gives `1 - exp(-x) / 0.99`, x = kappa 2.506 / mean(1/s). It is therefore NOT the
    inverse of alpha2density: round trip = theta - 0.0101 (theta >= 0.99 is unreachable).
    mode "inverse" is the mathematically intended inverse `(1 - exp(-x)) / 0.99`."""
    if mode not in DENSITY2ALPHA_MODES:
        raise ValueError(f"density2alpha mode must be one of {DENSITY2ALPHA_MODES}")
    mean_inv = (1.0 / scales).mean(dim=-1)
    e = torch.exp(-kappa * SQRT_2PI / mean_inv)
    alpha = 1.0 - e / 0.99 if mode == "official" else (1.0 - e) / 0.99
    return alpha.clamp(0.0, 0.99999)


def halve_theta(theta, scales, mode="official"):
    """theta of a Gaussian with half the density (clone/split in gaussian_model_volr.py).

    official: 1 - sqrt(1 - 0.99 theta) / 0.99 (~0.0102 below the true halving, 0 for theta < ~0.02)."""
    return density2alpha(alpha2density(theta, scales) / 2.0, scales, mode)


THETA_FLOOR_LOGIT = -30.0  # official inverse_sigmoid(0) = -inf; a finite stand-in (kappa ~ 0)


def inverse_sigmoid(x):
    """log(x / (1 - x)); theta = 0 (official clamp) maps to THETA_FLOOR_LOGIT instead of -inf so the
    runtime's finiteness check holds. Such Gaussians render with kappa ~ 1e-13 and are pruned
    (theta < min_theta) at the next densification step, as in the official code."""
    return torch.log(x / (1 - x)).clamp_min(THETA_FLOOR_LOGIT)


def quat_to_rotmat(quats):
    """wxyz (normalised inside) -> [N,3,3], same convention as gsplat and the slang code."""
    w, x, y, z = F.normalize(quats, dim=-1).unbind(-1)
    return torch.stack(
        (
            1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
            2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
            2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
        ),
        dim=-1,
    ).reshape(*quats.shape[:-1], 3, 3)


PRECIS_INDEX = ((0, 0, 0, 1, 1, 2), (0, 1, 2, 1, 2, 2))


def camera_gaussians(means, quats, log_scales, viewmat):
    """Camera-space means [N,3] and inverse covariances [N,3,3] (A = M diag(1/s^2) M^T, M = R Rq)."""
    R, t = viewmat[:3, :3], viewmat[:3, 3]
    mean_c = means @ R.T + t
    M = R @ quat_to_rotmat(quats)
    inv_s2 = torch.exp(-2.0 * log_scales)
    precis = (M * inv_s2[:, None, :]) @ M.transpose(1, 2)
    return mean_c, precis


def pixel_rays(K, width, height, dtype=torch.float64, device="cpu"):
    """Unit camera-space rays through pixel centres [H,W,3]."""
    ys, xs = torch.meshgrid(
        torch.arange(height, dtype=dtype, device=device) + 0.5,
        torch.arange(width, dtype=dtype, device=device) + 0.5,
        indexing="ij",
    )
    d = torch.stack(((xs - K[0, 2]) / K[0, 0], (ys - K[1, 2]) / K[1, 1], torch.ones_like(xs)), -1)
    return d / d.norm(dim=-1, keepdim=True)


def volumetric_alpha(mean_c, precis, kappa, rays, clamp=True):
    """alpha [P,N] of every Gaussian along every ray (evaluate_splat_volr), and a validity mask."""
    m = torch.einsum("nij,pj->pni", precis, rays)  # A d   [P,N,3]
    den_raw = (rays[:, None, :] * m).sum(-1)
    den = den_raw.clamp_min(EPS)
    num = (mean_c[None] * m).sum(-1)
    t = num / den
    diff = t[..., None] * rays[:, None, :] - mean_c[None]
    q = torch.einsum("pni,nij,pnj->pn", diff, precis, diff)
    pdf = torch.exp(-0.5 * q)
    sig = torch.sqrt(1.0 / (den + EPS))
    alpha = 1.0 - torch.exp(-SQRT_2PI * kappa[None] * sig * pdf)
    if clamp:
        alpha = alpha.clamp(max=MAX_ALPHA)
    valid = num > 0
    return torch.where(valid, alpha, torch.zeros_like(alpha)), valid


def dense_render(mean_c, precis, kappa, colors, K, width, height):
    """Dense reference: depth-sorted volumetric alpha blending with the kernel's skip and stop rules.

    Returns colours [H,W,C] (no background) and alpha [H,W,1]. Tile-box culling is not modelled."""
    rays = pixel_rays(K, width, height, mean_c.dtype, mean_c.device).reshape(-1, 3)
    order = torch.argsort(mean_c[:, 2].detach())
    mean_c, precis, kappa, colors = mean_c[order], precis[order], kappa[order], colors[order]
    alpha, valid = volumetric_alpha(mean_c, precis, kappa, rays)
    keep = valid & (alpha >= ALPHA_THRESHOLD)
    alpha = torch.where(keep, alpha, torch.zeros_like(alpha))
    T_after = torch.cumprod(1.0 - alpha, dim=1)
    # stop rule: a Gaussian is used only if T after it stays >= 1e-4 (T is non-increasing)
    used = T_after.detach() >= T_MIN
    alpha = torch.where(used, alpha, torch.zeros_like(alpha))
    T_after = torch.cumprod(1.0 - alpha, dim=1)
    T_before = torch.cat((torch.ones_like(T_after[:, :1]), T_after[:, :-1]), dim=1)
    w = alpha * T_before
    rgb = w @ colors
    acc = 1.0 - T_after[:, -1:]
    return rgb.reshape(height, width, -1), acc.reshape(height, width, 1)
