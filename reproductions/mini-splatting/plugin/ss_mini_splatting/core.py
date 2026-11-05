"""Pure tensor functions of Mini-Splatting (CPU-testable; no CUDA, no gsplat)."""

import math

import torch

SH_C0 = 0.28209479177387814


def expon_lr(step, lr_init, lr_final_ratio=0.01, max_steps=30000):
    """3DGS get_expon_lr_func with delay_steps = 0: log-linear from lr_init to ratio*lr_init."""
    t = min(max(step / max_steps, 0.0), 1.0)
    return lr_init * lr_final_ratio**t


def means_lr(step, lr_init, restart_step=15000, restart_offset=5000, **kw):
    """Official `update_learning_rate`: iteration < simp1 -> lr(it); else lr(it - simp1 + 5000)."""
    if step < restart_step:
        return expon_lr(step, lr_init, **kw)
    return expon_lr(step - restart_step + restart_offset, lr_init, **kw)


def sh_degree(step, restart_step=15000, max_degree=3):
    """Active SH degree at a step: 0 until the restart; oneupSHdegree at every multiple of 1000
    strictly after it (official `iteration % 1000 == 0 and iteration > simp_iteration1`)."""
    ups = step // 1000 - restart_step // 1000
    return int(min(max(ups, 0), max_degree))


def active_rest_coeffs(degree):
    """Number of shN coefficients (bands >= 1) used by an SH degree."""
    return (degree + 1) ** 2 - 1


def weighted_sample(weights, n, replace=False, generator=None):
    """Indices drawn with probability proportional to `weights`.

    replace=False: successive sampling without replacement (numpy `choice(replace=False, p=)`),
    via the Efraimidis-Spirakis exponential race (smallest E_i / w_i, E_i ~ Exp(1)), which has
    the same distribution. Only positive-weight entries can be drawn; n is clipped to their
    count. replace=True: i.i.d. draws, then unique (Mini-Splatting-D's depth sampling).
    """
    weights = weights.flatten().double()
    if replace:
        idx = torch.multinomial(weights, n, replacement=True, generator=generator)
        return torch.unique(idx)
    positive = int((weights > 0).sum())
    n = min(int(n), positive)
    if n == 0:
        return torch.zeros(0, dtype=torch.long, device=weights.device)
    e = torch.empty_like(weights).exponential_(generator=generator)
    keys = torch.where(weights > 0, e / weights, torch.full_like(weights, math.inf))
    return torch.topk(keys, n, largest=False).indices


def cdf_keep_mask(importance, thres=0.99):
    """Official `init_cdf_mask`: keep Gaussians above the value at which the ascending cumulative
    importance first exceeds (1 - thres) of the total."""
    importance = importance.flatten()
    if thres == 1.0:
        return torch.ones_like(importance, dtype=torch.bool)
    vals, _ = torch.sort(importance + 1e-6)
    cumsum = torch.cumsum(vals, dim=0)
    split_index = ((cumsum / vals.sum()) > (1 - thres)).nonzero().min()
    return importance > vals[split_index]


def accumulate_importance(imp, accum_area_max, view, metric):
    """One view's contribution to the importance score (official ms/train.py loop).

    outdoor (I^2): imp[m] += accum_weights / area_proj for Gaussians that are the argmax of some
    pixel in this view (m = area_max != 0); indoor (I^1): imp += accum_weights.
    """
    accum_area_max += view["area_max"].float()
    if metric == "outdoor":
        m = view["area_max"] != 0
        imp[m] += view["accum_weights"][m] / view["area_proj"][m].float()
    elif metric == "indoor":
        imp += view["accum_weights"]
    else:
        raise ValueError("imp_metric must be 'outdoor' or 'indoor'")
    return imp, accum_area_max


def sampling_count(importance, factor):
    """Official: int(N * factor * (#nonzero / N)) Gaussians survive importance sampling."""
    n = importance.numel()
    nnz = int((importance != 0).sum())
    return int(n * factor * (nnz / n))


def depth_samples_per_view(height, width, n_views, num_depth):
    """Official: factor = 1 / (H*W*n_views/num_depth); int(H*W*factor) pixels per view."""
    factor = 1 / (height * width * n_views / num_depth)
    return int(height * width * factor)


def midpoint_reference(origin, direction, mean, rotmat, scale):
    """Independent t* maximizing the Gaussian density along r(t) = o + t d (App. t_opt = -B/2C),
    written with the world-space precision matrix Sigma^-1 = R diag(s^-2) R^T."""
    prec = rotmat @ torch.diag(scale.pow(-2)) @ rotmat.T
    diff = origin - mean
    t = -(direction @ prec @ diff) / (direction @ prec @ direction)
    return origin + t * direction, t


def blend_reference(means2d, conics, opacities, depths, width, height):
    """Brute-force per-pixel front-to-back blend with gsplat numerics (independent of the CUDA
    kernel): returns accum_weights, area_proj, area_max, argmax [H,W], T [H,W]."""
    n = means2d.shape[0]
    order = torch.argsort(depths)
    accum_w = torch.zeros(n, dtype=torch.float64)
    area_proj = torch.zeros(n, dtype=torch.int64)
    area_max = torch.zeros(n, dtype=torch.int64)
    argmax = torch.full((height, width), -1, dtype=torch.int64)
    T_out = torch.ones(height, width, dtype=torch.float64)
    for i in range(height):
        for j in range(width):
            px, py = j + 0.5, i + 0.5
            T, wmax, imax = 1.0, 0.0, -1
            for g in order.tolist():
                dx, dy = float(means2d[g, 0]) - px, float(means2d[g, 1]) - py
                a, b, c = (float(v) for v in conics[g])
                sigma = 0.5 * (a * dx * dx + c * dy * dy) + b * dx * dy
                alpha = min(0.999, float(opacities[g]) * math.exp(-sigma))
                if sigma < 0 or alpha < 1 / 255:
                    continue
                nxt = T * (1 - alpha)
                if nxt <= 1e-4:
                    break
                vis = alpha * T
                if vis > wmax:
                    wmax, imax = vis, g
                accum_w[g] += vis
                area_proj[g] += 1
                T = nxt
            if imax >= 0:
                area_max[imax] += 1
            argmax[i, j] = imax
            T_out[i, j] = T
    return {
        "accum_weights": accum_w,
        "area_proj": area_proj,
        "area_max": area_max,
        "argmax": argmax,
        "T": T_out,
    }
