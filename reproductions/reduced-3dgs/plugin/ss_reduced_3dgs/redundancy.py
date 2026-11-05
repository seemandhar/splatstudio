"""C1 — scale- and resolution-aware redundancy score and the "mercy" pruning rule (paper §4.1).

Official code: scene/__init__.py:calculate_redundancy_metric, reduced_3dgs/redundancy_score.cu
(transformCentersNDC, sphereEllipsoidIntersection, findMinimumRedundancyValue) and
scene/gaussian_model.py:mercy_points (mercy_type = "redundancy_opacity_opacity").
"""

import math

import torch

from . import kernels

# 3DGS getProjectionMatrix near/far planes: the official frustum test is NDC z in [0, 1].
ZNEAR, ZFAR = 0.01, 100.0


@torch.no_grad()
def min_pixel_footprint(means, cameras, znear=ZNEAR, zfar=ZFAR, init=10000.0):
    """a_min per Gaussian: smallest world-space pixel size over views whose frustum holds it.

    Official transformCentersNDC un-projects NDC (2/W, 0, z) and (0, 0, z) (landscape) or
    (0, 2/H, z) and measures their distance: at camera depth z this is z / f_x (z / f_y).
    Inside = NDC x, y in [-1, 1] (pixel u in [0, W], v in [0, H]) and depth in [znear, zfar].
    """
    out = torch.full((len(means),), init, device=means.device)
    for cam in cameras:
        view = cam["viewmat"].to(means)
        K = cam["K"].to(means)
        W, H = int(cam["width"]), int(cam["height"])
        p = means @ view[:3, :3].T + view[:3, 3]
        z = p[:, 2]
        zs = z.clamp_min(1e-12)
        u = K[0, 0] * p[:, 0] / zs + K[0, 2]
        v = K[1, 1] * p[:, 1] / zs + K[1, 2]
        inside = (z >= znear) & (z <= zfar) & (u >= 0) & (u <= W) & (v >= 0) & (v <= H)
        size = z / (K[0, 0] if W > H else K[1, 1])
        out = torch.where(inside, torch.minimum(out, size), out)
    return out


def quat_to_rotmat(quats):
    """wxyz -> local-to-world rotation matrices (official buildRotationMatrixCUDA)."""
    w, x, y, z = torch.nn.functional.normalize(quats, dim=-1).unbind(-1)
    return torch.stack(
        [
            1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
            2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
            2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
        ],
        dim=-1,
    ).reshape(-1, 3, 3)


@torch.no_grad()
def intersections(means, scales, quats, neighbours, radius, chunk=1 << 19):
    """[N,K] mask: centre of i's sphere inside neighbour j's ellipsoid with axes s_j + r_i.

    Official sphereEllipsoidIntersectionCUDA: d = R_j^T (mu_i - mu_j);
    sum_k d_k^2 / (s_jk + r_i)^2 < 1. Neighbour slots < 0 (fewer than K points) never count.
    """
    n, k = neighbours.shape
    rot = quat_to_rotmat(quats)
    out = torch.zeros((n, k), dtype=torch.bool, device=means.device)
    for s in range(0, n, chunk):
        e = min(n, s + chunk)
        nb = neighbours[s:e]
        valid = nb >= 0
        nbc = nb.clamp_min(0)
        diff = means[s:e, None, :] - means[nbc]  # [c,K,3]
        local = torch.einsum("ckij,cki->ckj", rot[nbc], diff)  # R_j^T diff
        axes = scales[nbc] + radius[s:e, None, None]
        out[s:e] = ((local.square() / axes.square()).sum(-1) < 1) & valid
    return out


@torch.no_grad()
def redundancy_scores(means, scales, quats, cameras, knn=30, box_size=1.0):
    """Propagated (minimum) redundancy score per Gaussian (official calculate_redundancy_metric).

    scales are activated (world units), quats wxyz. Returns int64 [N].
    """
    n = len(means)
    footprint = min_pixel_footprint(means, cameras) * box_size
    half_diagonal = footprint * math.sqrt(3.0) / 2.0
    _, nbr = kernels.knn_index(means, knn)
    hit = intersections(means, scales, quats, nbr, half_diagonal)
    score = hit.sum(dim=1) + 1  # +1: the Gaussian at the centre of its own sphere
    # findMinimumRedundancyValue: every Gaussian whose ellipsoid contains sphere i's centre
    # (and i itself) receives min(current, score_i); initial value P.
    out = torch.full((n,), n, dtype=torch.long, device=means.device)
    src = score[:, None].expand(-1, nbr.shape[1])[hit]
    out.scatter_reduce_(0, nbr[hit], src, reduce="amin", include_self=True)
    out = torch.minimum(out, score)
    return out


def quantile_linear(x, q):
    """torch.quantile(x, q) ('linear' interpolation) without its 2^24-element limit."""
    s = torch.sort(x.flatten()).values
    pos = q * (len(s) - 1)
    lo = int(math.floor(pos))
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (pos - lo)


@torch.no_grad()
def mercy_mask(score, opacity, lambda_mercy=1.0, minimum=3.0, quantile=0.03, cap=0.05):
    """Official mercy_points, mercy_type="redundancy_opacity_opacity".

    candidates: score > max(mean + lambda * std, minimum) (std unbiased, as torch.var);
    prune candidates whose opacity is below the candidates' median opacity, plus every Gaussian
    with opacity < min(quantile_0.03(opacity), 0.05). Returns (mask, info dict).
    """
    s = score.float()
    mean, std = s.mean(), s.var().sqrt()
    threshold = max(float(mean + lambda_mercy * std), float(minimum))
    cand = s > threshold
    mask = torch.zeros_like(cand)
    if bool(cand.any()):
        mask[cand] = opacity[cand] < opacity[cand].median()
    opa_threshold = min(float(quantile_linear(opacity, quantile)), float(cap))
    mask |= opacity < opa_threshold
    info = {
        "redundancy_threshold": float(mean + lambda_mercy * std),
        "used_threshold": threshold,
        "candidates": int(cand.sum()),
        "opacity_threshold": opa_threshold,
        "pruned": int(mask.sum()),
    }
    return mask, info
