"""Pure functions of MH-3DGS (arXiv 2506.12945), ported from the official code
https://github.com/SNU-VGILab/MH-3DGS (commit 71ef8f9): train.py and scene/gaussian_model.py
(`combined_mcmc_and_reloc_multiview`, `importance_scores`, `sample_scores_from_maps`,
`compute_voxel_counts`, `voxel_density_factor`). Where the code and the paper disagree the code
is followed; the paper variant is available through config where it is well defined."""

import math

import torch
import torch.nn.functional as F

DEFAULTS = {
    # Densification window / interval (code arguments: densify_from_iter, densify_until_iter,
    # densification_interval; train.py uses strict `>` / `<`).
    "densify_from_iter": 500,
    "densify_until_iter": 15000,
    "densification_interval": 100,
    # Paper Sec. 5.1 / code arguments: linear coarse->fine anneals in densification progress.
    "coarse_max": 10.0,
    "coarse_min": 5.0,
    "fine_max": 2.0,
    "fine_min": 1.0,
    "voxel_max": 0.02,
    "voxel_min": 0.005,
    "coarse_batch_size": 4500,
    "fine_batch_size": 16000,
    # Voxel prior weight lambda_v (Eq. energychange; train.py passes alpha=5.0).
    "lambda_v": 5.0,
    # Importance weights alpha/beta/gamma (Eq. importance; code importance_scores defaults).
    "opacity_weight": 0.8,
    "ssim_weight": 0.5,
    "l1_weight": 0.5,
    # Acceptance rho = sigmoid(I - accept_shift) * D(v') (code); paper Sec. 4.3 has shift 0.
    "accept_shift": 0.5,
    # Relocation threshold tau (paper Sec. 4.2.4, code opacity_threshold).
    "opacity_threshold": 0.005,
    # Code arguments opacity_lr (runtime default is 0.05).
    "opacity_lr": 0.025,
    # "world_xy" = official sample_scores_from_maps (world x/y used as normalised pixel coords);
    # "projected" is not offered: the paper does not say which view the centre is projected into.
    "pixel_lookup": "world_xy",
    # "similarity" = official (aggregates the SSIM map itself); "dissimilarity" = paper 1-SSIM.
    "ssim_kind": "similarity",
    # 3DGS Scene shuffles the train cameras once; round robin walks that list.
    "shuffle_cameras": True,
    # Error-map renders use the training background (official: black).
    "background": [0.0, 0.0, 0.0],
    "seed": 0,
    # Training-image source for the error maps; default: ss-train's --data / --downscale.
    "data": None,
    "downscale": None,
}


def schedule(step, c):
    """Densification event at `step` (train.py lines 'if iteration < densify_until_iter and
    iteration > densify_from_iter and iteration % densification_interval == 0')."""
    lo, hi, every = int(c["densify_from_iter"]), int(c["densify_until_iter"]), int(
        c["densification_interval"])
    if not (lo < step < hi and step % every == 0):
        return None
    p = (step - lo) / (hi - lo)
    return {
        "progress": p,
        "coarse_scale": c["coarse_max"] * (1 - p) + c["coarse_min"] * p,
        "fine_scale": c["fine_max"] * (1 - p) + c["fine_min"] * p,
        "voxel_size": (1 - p) * c["voxel_max"] + p * c["voxel_min"],
        "coarse_batch": max(1, int((1 - p) * c["coarse_batch_size"])),
        "fine_batch": max(1, int((1 - p) * c["fine_batch_size"])),
    }


def subset_size(n_views, progress):
    """k_t = max(1, floor((1 - eta_t)|C|)) (paper Sec. 4.2.1; code int())."""
    return max(int(n_views * (1 - progress)), 1)


def round_robin(n_views, step, k):
    """train.py round_robin_cameras: contiguous block of k views starting at (step*k) % n."""
    start = (step * k) % n_views
    return [(start + i) % n_views for i in range(k)]


def ssim_map(img, gt, window_size=11):
    """utils/loss_utils.py _ssim(size_average=False): CHW inputs, Gaussian window sigma 1.5,
    zero 'same' padding, per-channel map [C,H,W]."""
    ch = img.shape[-3]
    coords = torch.arange(window_size, device=img.device, dtype=img.dtype) - window_size // 2
    g = torch.exp(-(coords**2) / (2 * 1.5**2))
    g = g / g.sum()
    window = (g[:, None] @ g[None, :])[None, None].expand(ch, 1, window_size, window_size)
    pad = window_size // 2

    def conv(t):
        return F.conv2d(t, window, padding=pad, groups=ch)

    mu1, mu2 = conv(img), conv(gt)
    s11 = conv(img * img) - mu1 * mu1
    s22 = conv(gt * gt) - mu2 * mu2
    s12 = conv(img * gt) - mu1 * mu2
    c1, c2 = 0.01**2, 0.03**2
    return ((2 * mu1 * mu2 + c1) * (2 * s12 + c2)) / ((mu1 * mu1 + mu2 * mu2 + c1) * (s11 + s22 + c2))


def lookup_world_xy(means, ssim_agg, l1_agg):
    """sample_scores_from_maps: pixel (x=int(mu_x*(W-1)), y=int(mu_y*(H-1))) clamped; the
    world-space mean is used directly (no projection). Maps are [3,H,W]; channel mean."""
    h, w = ssim_agg.shape[-2], ssim_agg.shape[-1]
    xs = (means[:, 0] * (w - 1)).long().clamp(0, w - 1)  # .long() truncates toward zero
    ys = (means[:, 1] * (h - 1)).long().clamp(0, h - 1)
    return ssim_agg[:, ys, xs].mean(dim=0), l1_agg[:, ys, xs].mean(dim=0)


def robust_normalize(x):
    """importance_scores.robust_normalize: (x - mean) / (std + 1e-6), unbiased std."""
    return (x - x.mean()) / (x.std() + 1e-6)


def importance(opacity, ssim_scores, l1_scores, c):
    """Eq. importance + surr-importance (code): sigmoid(a z(O) + b z(SSIM) + g z(L1))."""
    z = (
        c["opacity_weight"] * robust_normalize(opacity)
        + c["ssim_weight"] * robust_normalize(ssim_scores)
        + c["l1_weight"] * robust_normalize(l1_scores)
    )
    return torch.sigmoid(z).clamp(min=0)


def voxel_keys(points, voxel_size):
    """point_to_voxel_id: floor(x / v) per axis (code computes in Python double)."""
    return torch.floor(points.double() / voxel_size).long()


def voxel_counts_at(existing, queries, voxel_size):
    """compute_voxel_counts + dict lookup: number of `existing` points in each query's voxel."""
    ke, kq = voxel_keys(existing, voxel_size), voxel_keys(queries, voxel_size)
    _, inverse = torch.unique(torch.cat([ke, kq]), dim=0, return_inverse=True)
    counts = torch.bincount(inverse[: len(ke)], minlength=int(inverse.max()) + 1)
    return counts[inverse[len(ke):]]


def acceptance(importance_sel, counts, c):
    """Sec. 4.3 practical MH rule (code): sigmoid(I - shift) * 1 / (1 + lambda_v c(v'))."""
    return torch.sigmoid(importance_sel - c["accept_shift"]) / (1.0 + c["lambda_v"] * counts.float())


def propose(means, scales_act, denom, imp, batch, scale, existing, voxel_size, c, gen):
    """One proposal phase (coarse or fine) of combined_mcmc_and_reloc_multiview.
    Returns (parent indices of accepted proposals, their new means, proposal count, rho)."""
    idx = torch.multinomial(imp, batch, replacement=True, generator=gen)
    idx = torch.unique(idx)
    factor = 1.0 / (1.0 + torch.log(1.0 + denom[idx]))
    std = (scale * scales_act[idx] * factor[:, None]).clamp(min=1e-6)
    noise = torch.randn(std.shape, device=std.device, generator=gen, dtype=std.dtype)
    proposal = means[idx] + noise * std
    rho = acceptance(imp[idx], voxel_counts_at(existing, proposal, voxel_size), c)
    accept = torch.rand(rho.shape, device=rho.device, generator=gen) < rho
    return idx[accept], proposal[accept], int(idx.numel()), rho


def expected_accept(i_value, count, c):
    return 1.0 / (1.0 + math.exp(-(i_value - c["accept_shift"]))) / (1.0 + c["lambda_v"] * count)
