"""AtomGS objective (official utils/loss_utils.py + train.py @5b9ce904).

L = (1 - l_ms) L1 + l_ms (1 - MS-SSIM) + [step < t_a] l_n L_normal
"""

import math

import torch
import torch.nn.functional as F

PHOTO_DEFAULTS = {"lambda_ms_ssim": 0.1, "ms_ssim_scales": 3}
NORMAL_DEFAULTS = {"lambda_normal": 0.1, "until": 7000, "edge_power": 500}


def _window(channels, device, dtype, size=11, sigma=1.5):
    g = torch.tensor([math.exp(-(x - size // 2) ** 2 / float(2 * sigma ** 2))
                      for x in range(size)], device=device, dtype=dtype)
    g = g / g.sum()
    return (g[:, None] @ g[None, :])[None, None].expand(channels, 1, size, size).contiguous()


def ssim_same(x, y, size=11):
    """Official Inria ssim(): CHW inputs, Gaussian window, zero 'same' padding, mean."""
    c = x.shape[-3]
    w = _window(c, x.device, x.dtype, size)

    def conv(t):
        return F.conv2d(t, w, padding=size // 2, groups=c)

    mu1, mu2 = conv(x), conv(y)
    s1 = conv(x * x) - mu1 ** 2
    s2 = conv(y * y) - mu2 ** 2
    s12 = conv(x * y) - mu1 * mu2
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    m = ((2 * mu1 * mu2 + C1) * (2 * s12 + C2)) / ((mu1 ** 2 + mu2 ** 2 + C1) * (s1 + s2 + C2))
    return m.mean()


def ms_ssim(x, y, scales=3, size=11):
    """Official ms_ssim(): mean of SSIM over `scales` 2x2 average-pooled levels."""
    out = 0
    for _ in range(scales):
        out = out + ssim_same(x, y, size) / scales
        x = F.avg_pool2d(x, 2, stride=2)
        y = F.avg_pool2d(y, 2, stride=2)
    return out


def create_photometric(spec):
    unknown = set(spec.config) - set(PHOTO_DEFAULTS)
    if unknown:
        raise ValueError(f"Unknown AtomGS photometric options: {sorted(unknown)}")
    cfg = {**PHOTO_DEFAULTS, **spec.config}
    lam, scales = float(cfg["lambda_ms_ssim"]), int(cfg["ms_ssim_scales"])

    def loss(params, outputs, target):
        x, y = outputs["rgb"].permute(2, 0, 1), target.permute(2, 0, 1)
        l1 = torch.abs(x - y).mean()
        return (1.0 - lam) * l1 + lam * (1.0 - ms_ssim(x, y, scales))

    return loss


_SOBEL_X = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
_SOBEL_Y = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]


def edge_aware_normal_loss(image, normal, power=500):
    """Official edge_aware_normal_loss(I, D): image, normal CHW (normal = 3 xyz channels)."""
    sx = torch.tensor(_SOBEL_X, device=image.device).float()[None, None] / 4
    sy = torch.tensor(_SOBEL_Y, device=image.device).float()[None, None] / 4

    def per_channel(t, k):
        return F.conv2d(t[:, None], k, padding=1)[:, 0]

    dN_dx, dN_dy = per_channel(normal, sx), per_channel(normal, sy)
    dI_dx = per_channel(image, sx).abs().mean(0, keepdim=True)
    dI_dy = per_channel(image, sy).abs().mean(0, keepdim=True)
    wx = (dI_dx - 1) ** power
    wy = (dI_dy - 1) ** power
    return (dN_dx.abs() * wx + dN_dy.abs() * wy).norm(dim=0, keepdim=True).mean()


def create_normal_loss(spec):
    unknown = set(spec.config) - set(NORMAL_DEFAULTS)
    if unknown:
        raise ValueError(f"Unknown AtomGS normal-loss options: {sorted(unknown)}")
    cfg = {**NORMAL_DEFAULTS, **spec.config}
    lam, until, power = float(cfg["lambda_normal"]), int(cfg["until"]), int(cfg["edge_power"])

    def loss(params, outputs, target):
        normal = outputs["surface_normal"]
        if outputs["step"] >= until:  # train.py: `if iteration < opt.atom_proliferation_until`
            return normal.sum() * 0.0
        return lam * edge_aware_normal_loss(target.permute(2, 0, 1), normal.permute(2, 0, 1),
                                            power)

    return loss
