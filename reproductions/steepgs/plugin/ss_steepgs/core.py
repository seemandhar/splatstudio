"""Pure tensor functions of SteepGS density control (official scene/gaussian_model.py,
utils/general_utils.py of github.com/facebookresearch/SteepGS, commit 63eee45)."""

import torch
import torch.nn.functional as F

OPTIONS = ("stationary", "no_uncertain", "no_eig_cond", "no_saddle", "no_eig_upd", "zero_upd",
           "no_div_opacity")


def quat_to_rotmat(quats):
    """utils/general_utils.build_rotation: wxyz (any norm) -> rotation matrices [N,3,3]."""
    r, x, y, z = F.normalize(quats, dim=-1).unbind(-1)
    return torch.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - r * z), 2 * (x * z + r * y),
        2 * (x * y + r * z), 1 - 2 * (x * x + z * z), 2 * (y * z - r * x),
        2 * (x * z - r * y), 2 * (y * z + r * x), 1 - 2 * (x * x + y * y)], -1).reshape(-1, 3, 3)


def inverse_covariance(log_scales, quats):
    """get_covariance_inv_matrix: L = R diag(1 / (s + 1e-8)), Sigma^-1 = L L^T."""
    L = quat_to_rotmat(quats) * (1.0 / (log_scales.exp() + 1e-8))[:, None, :]
    return L @ L.transpose(1, 2)


def smallest_eig(mats, batch_size=1_000_000):
    """eigh_in_batch(least_k=1): ascending eigh, first eigenvalue / eigenvector column."""
    vals, vecs = [], []
    for A in torch.split(mats, batch_size, dim=0):
        w, v = torch.linalg.eigh(A)
        vals.append(w[:, 0])
        vecs.append(v[:, :, 0])
    if not vals:
        return mats.new_zeros(0), mats.new_zeros(0, 3)
    return torch.cat(vals), torch.cat(vecs)


def averaged(accum, denom):
    """`accum / denom` with NaN (denom 0) -> 0, as in densify_and_prune."""
    out = accum / denom.reshape(-1, *([1] * (accum.dim() - 1)))
    out[out.isnan()] = 0.0
    return out


def select(options, viewspace_grad, lam_min, grad_threshold, S_threshold, xyz_grad_norm=None,
           denom=None, min_grad=1e-4):
    """densify_and_split_steepest filters (gaussian_model.py:543-559).

    Default options [] -> uncertain (mean screen-gradient >= threshold) AND lambda_min < S_thr.
    """
    if S_threshold >= 0:
        raise ValueError("S_threshold must be a negative number.")
    filters = []
    if "no_uncertain" not in options:
        filters.append(viewspace_grad >= grad_threshold)
    if "no_eig_cond" not in options and "no_saddle" not in options:
        filters.append(lam_min < S_threshold)
    if "stationary" in options and "no_saddle" not in options:
        if xyz_grad_norm is None or denom is None:
            raise ValueError("'stationary' needs the accumulated xyz gradient")
        filters.append((xyz_grad_norm <= min_grad) & (denom >= 1))
    if not filters:
        raise ValueError("SteepGS options disable every selection filter")
    mask = filters[0]
    for m in filters[1:]:
        mask = mask & m
    return mask


def split_offsets(eigvecs, scales, options=(), generator=None):
    """Offsets of the 2K split offspring (rows i and K+i belong to parent i, `repeat(N, 1)`).

    Official (gaussian_model.py:577-587): direction v_min, length ||n||, n ~ N(0, diag(s)),
    sign + on rows 0::2 and - on rows 1::2 of the repeated array (so for an even K both
    offspring of a parent fall on the same side; replicated as-is).
    """
    stds = scales.repeat(2, 1)
    if "no_eig_upd" in options:
        raise ValueError("use random_split_offsets for 'no_eig_upd'")
    n = torch.normal(torch.zeros_like(stds), stds, generator=generator)
    length = torch.norm(n, dim=-1, keepdim=True)
    offsets = eigvecs.repeat(2, 1)
    offsets[0::2] = offsets[0::2] * length[0::2]
    offsets[1::2] = offsets[1::2] * length[1::2] * -1.0
    return offsets


def random_split_offsets(scales, quats, options=(), generator=None):
    """'no_eig_upd' ablation: 3DGS split sampling R n, n ~ N(0, diag(s)) (zero with zero_upd)."""
    stds = scales.repeat(2, 1)
    n = torch.normal(torch.zeros_like(stds), stds, generator=generator)
    out = torch.bmm(quat_to_rotmat(quats).repeat(2, 1, 1), n[..., None])[..., 0]
    return torch.zeros_like(out) if "zero_upd" in options else out
