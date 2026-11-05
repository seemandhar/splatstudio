"""Pure tensor functions of Mini-Splatting2 (CPU-testable; no CUDA, no gsplat).

`cdf_keep_mask`, `weighted_sample`, `accumulate_importance`, `sampling_count`,
`depth_samples_per_view`, `midpoint_reference` and `blend_reference` are copied from the
SplatStudio Mini-Splatting port (reproductions/mini-splatting/plugin/ss_mini_splatting/core.py,
plan #28); the rest follows the official mini-splatting2 code (msv2/train.py,
scene/gaussian_model.py, submodules/diff-gaussian-rasterization_ms).
"""

import math

import torch

SH_C0 = 0.28209479177387814


# ---------------------------------------------------------------- schedules
def expon_lr(step, lr_init, lr_final_ratio=0.01, max_steps=30000):
    """3DGS get_expon_lr_func with delay_steps = 0. MSv2 keeps position_lr_max_steps = 30000
    although it trains 18000 iterations (config/fast does not override it)."""
    t = min(max(step / max_steps, 0.0), 1.0)
    return lr_init * lr_final_ratio**t


def sh_degree(step, sh_from=3000, max_degree=3):
    """Official: GaussianModel(sh_degree=0) until simp_iteration1, then max degree 3 and
    oneupSHdegree() at `iteration % 1000 == 0 and iteration > simp_iteration1`."""
    ups = step // 1000 - sh_from // 1000
    return int(min(max(ups, 0), max_degree))


def active_rest_coeffs(degree):
    """Number of shN coefficients (bands >= 1) used by an SH degree."""
    return (degree + 1) ** 2 - 1


def half_res_size(width, height):
    """3DGS loadCam with resolution_scale 2 and -r -1 (width <= 1600): int(W/2), int(H/2)."""
    return int(width / 2.0), int(height / 2.0)


def scale_intrinsics(K, width, height, new_width, new_height):
    """Same FoV at the new size: focal and principal point scale with the image size."""
    K = K.clone()
    K[0, :] *= new_width / width
    K[1, :] *= new_height / height
    K[2, :] = torch.tensor([0.0, 0.0, 1.0], dtype=K.dtype, device=K.device)
    return K


def events(step, cfg):
    """Density events of the official msv2 loop at iteration `step` (in execution order)."""
    out = []
    if step < cfg["densify_until"]:
        if (step > cfg["densify_from"] and step % cfg["densify_every"] == 0
                and step != cfg["depth_reinit_iter"]):
            out.append("densify")
        if step == cfg["depth_reinit_iter"]:
            out.append("depth_reinit")
        if (step >= cfg["clone_from"] and step % cfg["clone_every"] == 0
                and step != cfg["depth_reinit_iter"]):
            out.append("culling_with_clone")
    if step == cfg["simp_iteration1"]:
        out.append("simplify1")
    if step == cfg["simp_iteration2"]:
        out.append("simplify2")
    if step == (cfg["simp_iteration2"] + cfg["total_steps"]) // 2:
        out.append("culling_off")
    return out


# ---------------------------------------------------------------- C2 aggressive clone
def clone_values(opacity_old, scale_old, min_opacity=0.0051):
    """Official GaussianModel.clone (3DGS-MCMC state transition with N = 2):
    alpha_new = 1 - sqrt(1 - alpha_old); s_new = alpha_old / (2 alpha_new - alpha_new^2/sqrt 2)
    * s_old (i.e. Sigma_new = (...)^2 Sigma_old, paper Sec. 4); opacity clamped to
    [0.0051, 1 - eps] after the scale is computed. Returns activated (opacity, scale)."""
    a_new = 1 - (1 - opacity_old) ** 0.5
    s_new = (opacity_old / (2 * a_new - 0.5**0.5 * a_new**2))[..., None] * scale_old
    eps = torch.finfo(torch.float32).eps
    a_new = a_new.clamp(min=min_opacity, max=1.0 - eps)
    return a_new, s_new


def mcmc_reference(opacity_old, n=2):
    """Independent 3DGS-MCMC (Kheradmand et al., Eq. 9-10) for n copies with the binomial sum:
    alpha_new = 1 - (1 - alpha)^(1/n); scale ratio = alpha / sum_{i=1}^{n} sum_{k=0}^{i-1}
    C(i-1,k) (-1)^k alpha_new^(k+1) / sqrt(k+1)."""
    a_new = 1 - (1 - opacity_old) ** (1.0 / n)
    denom = torch.zeros_like(opacity_old)
    for i in range(1, n + 1):
        for k in range(i):
            denom = denom + math.comb(i - 1, k) * (-1) ** k * a_new ** (k + 1) / math.sqrt(k + 1)
    return a_new, opacity_old / denom


def culling_factor(count_vis, count_rad):
    """Official factor_culling = count_vis / (count_rad + 0.1)."""
    return count_vis / (count_rad + 1e-1)


def sparse_adam_lr(lr, step, b1=0.9, b2=0.999):
    """Taming-3DGS SparseGaussianAdam: bias correction folded into the step size."""
    return lr * (1 - b2**step) ** 0.5 / (1 - b1**step)


# ---------------------------------------------------------------- copied from Mini-Splatting
def weighted_sample(weights, n, replace=False, generator=None):
    """Indices drawn with probability proportional to `weights`.

    replace=False: successive sampling without replacement (numpy `choice(replace=False, p=)`),
    via the Efraimidis-Spirakis exponential race (smallest E_i / w_i, E_i ~ Exp(1)), which has
    the same distribution. Only positive-weight entries can be drawn; n is clipped to their
    count.
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
    """One view's contribution to the importance score (official culling_with_* loops).

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
    """Independent t* maximizing the Gaussian density along r(t) = o + t d (t_opt = -B/2C),
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
