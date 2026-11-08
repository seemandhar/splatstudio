"""Pure VoD-3DGS math (arXiv 2501.17978). Ŝ is stored as 6 params [Sxx, Sxy, Sxz, Syy, Syz, Szz]."""

import torch
import torch.nn.functional as F

_CONTEXT = {"train_centers": None}  # training camera centers registered by the density component


def camera_center(viewmat):
    """World camera center of an OpenCV world-to-camera matrix: c = -Rᵀ t."""
    R, t = viewmat[:3, :3], viewmat[:3, 3]
    return -R.T @ t


def view_dirs(means, center):
    """Unit view vector between Gaussian center and camera center (§4.1). Sign is irrelevant
    because the quadratic form is even in ω."""
    return F.normalize(center[None, :] - means, dim=-1, eps=1e-12)


def quad_form(S6, omega):
    """ωᵀ Ŝ ω for symmetric Ŝ given by six coefficients (Eq. 5 layout); Eq. 9 term."""
    x, y, z = omega.unbind(-1)
    sxx, sxy, sxz, syy, syz, szz = S6.unbind(-1)
    return (
        sxx * x * x + syy * y * y + szz * z * z
        + 2.0 * (sxy * x * y + sxz * x * z + syz * y * z)
    )


def vod_logits(opacity_logits, S6, means, center):
    """Eq. 9 logit: γ_i + ω_{i,j}ᵀ Ŝ_i ω_{i,j}."""
    return opacity_logits + quad_form(S6, view_dirs(means, center))


def to_matrix(S6):
    sxx, sxy, sxz, syy, syz, szz = S6.unbind(-1)
    return torch.stack(
        [torch.stack([sxx, sxy, sxz], -1), torch.stack([sxy, syy, syz], -1), torch.stack([sxz, syz, szz], -1)],
        -2,
    )


def from_matrix(M):
    return torch.stack([M[..., 0, 0], M[..., 0, 1], M[..., 0, 2], M[..., 1, 1], M[..., 1, 2], M[..., 2, 2]], -1)


@torch.no_grad()
def eigen_reset(S6, variant, chunk=1_000_000):
    """Eq. 11: Ŝ* = Q diag(λ) Qᵀ keeping only the largest ([L]) or smallest ([S]) eigenvalue."""
    if variant not in ("S", "L"):
        raise ValueError("variant must be 'S' or 'L'")
    out = torch.empty_like(S6)
    for s in range(0, len(S6), chunk):
        M = to_matrix(S6[s : s + chunk].double())
        evals, evecs = torch.linalg.eigh(M)  # ascending eigenvalues
        k = 0 if variant == "S" else 2
        v = evecs[..., :, k]
        out[s : s + chunk] = from_matrix(evals[..., k, None, None] * v[..., :, None] * v[..., None, :]).to(S6.dtype)
    return out


@torch.no_grad()
def max_view_logits(opacity_logits, S6, means, centers):
    """Eq. 10 support: γ + max_j ωᵀŜω over the training cameras (σ is monotone, so
    max_j σ(γ + q_j) = σ(γ + max_j q_j))."""
    best = torch.full_like(opacity_logits, -float("inf"))
    for c in centers:
        best = torch.maximum(best, quad_form(S6, view_dirs(means, c)))
    return opacity_logits + best


def view_consistency(opacity_logits, S6, means, center_i, center_j):
    """Eq. 12: mean_g max(cos θ, 0) (α̂(ω_{g,i}) − α̂(ω_{g,j}))². Means are treated as fixed
    geometry (detached); gradients reach γ and Ŝ."""
    m = means.detach()
    wi, wj = view_dirs(m, center_i), view_dirs(m, center_j)
    cos = (wi * wj).sum(-1).clamp_min(0.0)
    ai = torch.sigmoid(opacity_logits + quad_form(S6, wi))
    aj = torch.sigmoid(opacity_logits + quad_form(S6, wj))
    return (cos * (ai - aj).square()).mean()
