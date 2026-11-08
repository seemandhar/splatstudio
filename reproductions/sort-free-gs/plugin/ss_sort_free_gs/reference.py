"""Dense (all pixels x all Gaussians) torch reference of the WSR sums, written independently of
the CUDA kernel: S_f = sum_i f_i u_i G_i w_i, S_w = sum_i u_i G_i w_i (paper Eq. 7 numerator /
denominator without the background term), with the same skip rule (|u G| < 1/255 or
Mahalanobis distance > 3.33) and only Gaussians that are live (radius > 0)."""

import torch

from .kernels import ALPHA_THRESHOLD, MAX_MAHALANOBIS


def dense_wsr(means2d, conics, feats, opac, weights, live, width, height):
    dt = means2d.dtype
    ys, xs = torch.meshgrid(torch.arange(height, dtype=dt, device=means2d.device) + 0.5,
                            torch.arange(width, dtype=dt, device=means2d.device) + 0.5,
                            indexing="ij")
    px = torch.stack((xs, ys), dim=-1).reshape(-1, 1, 2)  # [P,1,2]
    d = means2d[None] - px  # [P,N,2]
    a, b, c = conics[:, 0], conics[:, 1], conics[:, 2]
    sig = 0.5 * (a * d[..., 0] ** 2 + c * d[..., 1] ** 2) + b * d[..., 0] * d[..., 1]
    G = torch.exp(-sig)
    alpha = opac[None] * G
    with torch.no_grad():
        mask = (sig >= 0) & (sig <= 0.5 * MAX_MAHALANOBIS**2) & (alpha.abs() >= ALPHA_THRESHOLD)
        mask &= live[None]
    aw = torch.where(mask, alpha * weights[None], torch.zeros_like(alpha))  # [P,N]
    feat_sum = (aw @ feats).reshape(height, width, -1)
    w_sum = aw.sum(dim=1).reshape(height, width, 1)
    return feat_sum, w_sum


def alpha_composite(means2d, conics, feats, opac, depths, live, width, height):
    """Sorted front-to-back alpha compositing (3DGS Eq. 1) of the same splats, for contrast."""
    order = torch.argsort(depths)
    ones = torch.ones_like(opac)
    out = torch.zeros(height * width, feats.shape[1], dtype=feats.dtype, device=feats.device)
    T = torch.ones(height * width, 1, dtype=feats.dtype, device=feats.device)
    for i in order.tolist():
        if not bool(live[i]):
            continue
        sl = slice(i, i + 1)
        _, a = dense_wsr(means2d[sl], conics[sl], feats[sl], opac[sl], ones[sl], live[sl], width,
                         height)
        a = a.reshape(-1, 1).clamp(max=0.999)
        out = out + T * a * feats[i]
        T = T * (1 - a)
    return out.reshape(height, width, -1)
