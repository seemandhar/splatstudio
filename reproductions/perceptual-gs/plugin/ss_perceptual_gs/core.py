"""Pure tensor functions of Perceptual-GS (CPU-testable; no CUDA, no gsplat).

Equation numbers refer to arXiv 2506.12400; `code:` references are to the official repository
https://github.com/eezkni/Perceptual-GS.
"""

import math

import torch
import torch.nn.functional as F

SH_C0 = 0.28209479177387814
SOBEL_X = ((-1.0, 0.0, 1.0), (-2.0, 0.0, 2.0), (-1.0, 0.0, 1.0))  # Eq. 5
SOBEL_Y = ((-1.0, -2.0, -1.0), (0.0, 0.0, 0.0), (1.0, 2.0, 1.0))


def sobel_magnitude(image):
    """Eq. 6 on a [1,1,H,W] luminance tensor (code: preprocess.sobel_operator, padding 1)."""
    gx = torch.tensor(SOBEL_X, dtype=image.dtype, device=image.device)[None, None]
    gy = torch.tensor(SOBEL_Y, dtype=image.dtype, device=image.device)[None, None]
    ex, ey = F.conv2d(image, gx, padding=1), F.conv2d(image, gy, padding=1)
    return torch.sqrt(ex**2 + ey**2)


def sensitivity_map(image, tau_e=0.05, tau_s=0.3, pool=5):
    """Eqs. 6-7 + perception-oriented smoothing (code: perceptual_sensitivity_extraction).

    image: [1,1,H,W] gamma-corrected luminance in [0,1]. Returns a binary float map [1,1,H,W]:
    G_E = 1[G > tau_e]; avg_pool2d(kernel=pool) (stride = kernel, floor), nearest upsample back to
    H x W, then 1[. > tau_s].
    """
    enhanced = (sobel_magnitude(image) > tau_e).to(torch.float32)
    pooled = F.avg_pool2d(enhanced, kernel_size=pool)
    smoothed = F.interpolate(pooled, size=enhanced.shape[2:], mode="nearest")
    return (smoothed > tau_s).to(torch.float32)


def scene_sensitivity(maps):
    """Eqs. 15-16, code train.py:196-204: mean over views of #(sens > 0) / #pixels."""
    return float(torch.stack([(m > 0).float().mean() for m in maps]).mean())


def sparse_indicator(max_scales, sens, tau_l=0.3, tau_h=0.9, quantile_fraction=0.25):
    """Eqs. 17-18 (code train.py:178-183): gamma = |G_l ∩ G_m| / |G_l| with G_l the top-25%
    largest Gaussians by longest axis (scale >= k-th largest, k = int(0.25 N)) and G_m the open
    interval tau_l < sigma(eps) < tau_h."""
    k = max(int(max_scales.numel() * quantile_fraction), 1)
    threshold = torch.topk(max_scales.flatten(), k, largest=True).values.min()
    masked = sens.flatten() * (max_scales.flatten() >= threshold)
    mixed = ((masked > tau_l) & (masked < tau_h)).sum()
    return float(mixed) / float(torch.count_nonzero(masked))


def opacity_decline(alpha, k=1.2):
    """Eqs. 19-20: alpha_hat = 1 - sqrt(1 - alpha^k); code writes alpha^k / (1 + sqrt(1 - alpha^k))."""
    a = alpha**k
    return a / (1 + torch.sqrt(1 - a))


def bce(pred, target):
    """torch.nn.BCELoss (mean, log clamped at -100), Eq. 9."""
    return F.binary_cross_entropy(pred, target)


def predicted_sensitivity(render_one_minus):
    """Official prediction 1 - R[(1 - sigma(eps))] over a black background (train.py:111-117)."""
    return (1 - render_one_minus).clamp(0, 1)


def weighted_sample_unique(weights, n, generator=None):
    """np.random.choice(N, n, p=w) (with replacement) followed by np.unique (train.py:241-245)."""
    weights = weights.flatten().double()
    if n <= 0 or float(weights.sum()) <= 0:
        return torch.zeros(0, dtype=torch.long, device=weights.device)
    idx = torch.multinomial(weights, n, replacement=True, generator=generator)
    return torch.unique(idx)


def depth_samples_per_view(height, width, n_views, num_depth=3_500_000):
    """Official: factor = 1 / (H*W*|V| / 3.5M); num_sampled = int(H*W*factor)."""
    factor = 1 / (height * width * n_views / num_depth)
    return int(height * width * factor)


def logit(p):
    return math.log(p / (1 - p))


def midpoint_reference(origin, direction, mean, rotmat, scale):
    """Point on r(t) = o + t d maximizing the Gaussian density (official render_depth uses the
    3-sigma ellipsoid; the arg-max is scale-invariant). Independent world-space formula."""
    prec = rotmat @ torch.diag(scale.pow(-2)) @ rotmat.T
    diff = origin - mean
    t = -(direction @ prec @ diff) / (direction @ prec @ direction)
    return origin + t * direction, t


def blend_reference(means2d, conics, opacities, depths, width, height):
    """Brute-force per-pixel front-to-back blend with gsplat numerics (independent of the CUDA
    kernel): accum_weights [N] = sum_pixels alpha T, argmax [H,W], T [H,W]."""
    n = means2d.shape[0]
    order = torch.argsort(depths)
    accum_w = torch.zeros(n, dtype=torch.float64)
    argmax = torch.full((height, width), -1, dtype=torch.int64)
    t_out = torch.ones(height, width, dtype=torch.float64)
    for i in range(height):
        for j in range(width):
            px, py = j + 0.5, i + 0.5
            t, wmax, imax = 1.0, 0.0, -1
            for g in order.tolist():
                dx, dy = float(means2d[g, 0]) - px, float(means2d[g, 1]) - py
                a, b, c = (float(v) for v in conics[g])
                sigma = 0.5 * (a * dx * dx + c * dy * dy) + b * dx * dy
                alpha = min(0.999, float(opacities[g]) * math.exp(-sigma))
                if sigma < 0 or alpha < 1 / 255:
                    continue
                nxt = t * (1 - alpha)
                if nxt <= 1e-4:
                    break
                vis = alpha * t
                if vis > wmax:
                    wmax, imax = vis, g
                accum_w[g] += vis
                t = nxt
            argmax[i, j] = imax
            t_out[i, j] = t
    return {"accum_weights": accum_w, "argmax": argmax, "T": t_out}
