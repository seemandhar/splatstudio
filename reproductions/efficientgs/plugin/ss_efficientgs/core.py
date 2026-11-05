"""Pure tensor functions of EfficientGS (CPU-testable; no CUDA, no gsplat)."""

import math

import torch

SH_C0 = 0.28209479177387814
MAX_SH_DEGREE = 3


def band_of_rest_coeff(n_rest=15):
    """SH band l of each shN coefficient (index k of bands >= 1): 3 x l=1, 5 x l=2, 7 x l=3."""
    return torch.tensor([int(math.isqrt(k + 1)) for k in range(n_rest)])


def rest_mask(order, max_degree):
    """[N, (max_degree+1)^2 - 1] float mask: 1 where the coefficient's band <= the Gaussian's
    own SH degree. Zeroed bands contribute nothing to colour or gradients, so evaluating
    degree `max_degree` with this mask equals evaluating each Gaussian at its own degree
    (official computeColorFromSH(idx, sparse_shs_degree[idx]))."""
    k = (max_degree + 1) ** 2 - 1
    band = band_of_rest_coeff(15)[:k].to(order.device)
    return (band[None, :] <= order[:, None]).to(torch.float32)


def sparse_increment_mask(loss_weight, rate):
    """Official get_sparse_shs_mask: k = int(n * r_s); thr = lw[argsort(-lw)[k]]; lw >= thr."""
    n = loss_weight.numel()
    k = int(n * rate)
    if k >= n:
        return torch.ones(n, dtype=torch.bool, device=loss_weight.device)
    idx = torch.argsort(-loss_weight)[k]
    return loss_weight >= loss_weight[idx]


def increment_orders(order, mask, max_degree=MAX_SH_DEGREE):
    """+1 SH degree where `mask`, capped at max_degree (official train.py)."""
    out = order.clone()
    out[mask] += 1
    return out.clamp(max=max_degree)


def sparse_steps(start=15500, interval=1000, count=3):
    """Iterations (official: `iteration % interval == 0 and iteration >= start`, sparse_cnt)."""
    first = ((start + interval - 1) // interval) * interval
    return tuple(first + i * interval for i in range(count))


def densify_select(stat, count, max_scale, threshold, percent_dense, extent):
    """3DGS densify_and_clone / densify_and_split masks on the averaged statistic.

    stat/count: accumulated S_g (Eq. 6) and the number of views where g was visible.
    Returns (grads, clone, split) for the population before densification; split excludes
    the clones, which the official code appends with zero padded gradient.
    """
    grads = stat / count
    grads[grads.isnan()] = 0.0
    high = grads >= threshold
    small = max_scale <= percent_dense * extent
    return grads, high & small, high & ~small


def blend_reference(means2d, conics, opacities, colors, depths, width, height,
                    top_k=1, weight_after_T=True, pix_loss=None):
    """Brute-force per-pixel front-to-back blend with gsplat numerics, independent of the CUDA
    kernels. Differentiable in means2d (torch ops). Returns per-pixel colour [H,W,3], alpha
    [H,W], dominant flags [N] (Eq. 8) and loss weights [N] (Eq. 9)."""
    n = means2d.shape[0]
    order = torch.argsort(depths).tolist()
    dominant = torch.zeros(n, dtype=torch.int64)
    lw = torch.zeros(n, dtype=torch.float64)
    rgb_rows, alpha_rows = [], []
    for i in range(height):
        rgb_row, alpha_row = [], []
        for j in range(width):
            px, py = j + 0.5, i + 0.5
            T = torch.ones((), dtype=means2d.dtype)
            c = torch.zeros(3, dtype=means2d.dtype)
            best = []
            for g in order:
                d = means2d[g] - torch.tensor([px, py], dtype=means2d.dtype)
                a, b, cc = conics[g]
                sigma = 0.5 * (a * d[0] ** 2 + cc * d[1] ** 2) + b * d[0] * d[1]
                raw = opacities[g] * torch.exp(-sigma)
                alpha = torch.clamp(raw, max=0.999)
                if float(sigma) < 0 or float(alpha) < 1 / 255:
                    continue
                nxt = T * (1 - alpha)
                if float(nxt) <= 1e-4:
                    break
                c = c + colors[g] * alpha * T
                w = float(alpha * (nxt if weight_after_T else T))
                if pix_loss is not None:
                    lw[g] += float(pix_loss[i, j].sum()) * w
                best.append((w, g))
                T = nxt
            # strict '>' insertion keeps the earliest of equal weights (official top-K)
            ranked = sorted(enumerate(best), key=lambda e: (-e[1][0], e[0]))[:top_k]
            for _, (_, g) in ranked:
                dominant[g] = 1
            rgb_row.append(c)
            alpha_row.append(1 - T)
        rgb_rows.append(torch.stack(rgb_row))
        alpha_rows.append(torch.stack(alpha_row))
    return {
        "rgb": torch.stack(rgb_rows),
        "alpha": torch.stack(alpha_rows),
        "dominant": dominant,
        "loss_weight": lw,
    }


def gradnorm_reference(means2d, conics, opacities, colors, depths, width, height, v_rgb, v_alpha):
    """Eq. 6 by autograd: for every pixel p, d(v_rgb[p]·C_p + v_alpha[p]·A_p)/d mean2d, scaled
    to NDC (x W/2, y H/2), norm per Gaussian, summed over pixels. Also returns the plain sum of
    per-pixel vectors (pixel units), which must equal gsplat's means2d gradient."""
    n = means2d.shape[0]
    norm = torch.zeros(n, dtype=torch.float64)
    vec = torch.zeros(n, 2, dtype=torch.float64)
    scale = torch.tensor([width / 2.0, height / 2.0], dtype=torch.float64)
    for i in range(height):
        for j in range(width):
            m = means2d.detach().clone().double().requires_grad_(True)
            # single-pixel blend at (i, j): shift the image so this pixel is (0, 0)
            shifted = m - torch.tensor([j, i], dtype=torch.float64)
            ref = blend_reference(shifted, conics.double(), opacities.double(),
                                  colors.double(), depths, 1, 1)
            obj = (ref["rgb"][0, 0] * v_rgb[i, j].double()).sum() + ref["alpha"][0, 0] * float(
                v_alpha[i, j]
            )
            if not obj.requires_grad:
                continue
            (g,) = torch.autograd.grad(obj, m, allow_unused=True)
            if g is None:
                continue
            vec += g
            norm += (g * scale).norm(dim=-1)
    return norm, vec
