"""Image-space operations of BAGS (train.py of github.com/snldmt/BAGS @ed04144)."""

from math import exp

import torch
import torch.nn.functional as F


def spatial_conv(image, weights, K):
    """Spatially varying K x K convolution of the rendered image (Eq. 1).

    image [C,H,W], weights [K*K,H,W] (softmax-normalised per pixel). Official code:
    ``patches = Unfold(K, padding=K//2)(rgb).view(1, C, K*K, H, W)``;
    ``rgb = sum(patches * kernel_weights.unsqueeze(1), 2)``. Zero padding; kernel index
    k = ky*K + kx reads pixel (y + ky - K//2, x + kx - K//2) (correlation). Gradients flow into
    both the image and the kernels.
    """
    C, H, W = image.shape
    if weights.shape != (K * K, H, W):
        raise ValueError(f"kernel weights {tuple(weights.shape)} != {(K * K, H, W)}")
    patches = F.unfold(image[None], kernel_size=(K, K), padding=K // 2).view(1, C, K * K, H, W)
    return torch.sum(patches * weights[None, None], 2)[0]


def normalized_depth(depth):
    """train.py: ``d = depth - depth.min(); d = d / d.max()`` (keeps the gradient)."""
    d = depth - depth.min()
    return d / d.max().clamp_min(1e-12)  # official divides without a guard


def tv_loss(grids):
    """train.py tv_loss (TensoRF): grids [B,C,H,W]."""
    n = grids.shape[0]
    h_count = grids[:, :, 1:, :].shape[1] * grids[:, :, 1:, :].shape[2] * grids[:, :, 1:, :].shape[3]
    w_count = grids[:, :, :, 1:].shape[1] * grids[:, :, :, 1:].shape[2] * grids[:, :, :, 1:].shape[3]
    h_tv = torch.pow(grids[:, :, 1:, :] - grids[:, :, :-1, :], 2).sum()
    w_tv = torch.pow(grids[:, :, :, 1:] - grids[:, :, :, :-1], 2).sum()
    return 2 * (h_tv / h_count + w_tv / w_count) / n


def _window(size, sigma, channel, device, dtype):
    g = torch.tensor([exp(-((x - size // 2) ** 2) / float(2 * sigma**2)) for x in range(size)])
    g = (g / g.sum()).to(device=device, dtype=dtype)
    w2 = (g[:, None] @ g[None, :])[None, None]
    return w2.expand(channel, 1, size, size).contiguous()


def ssim_same(img1, img2, window_size=11):
    """utils/loss_utils.ssim (Inria 3DGS): CHW, Gaussian 11x11 sigma 1.5, same padding, mean."""
    c = img1.shape[-3]
    w = _window(window_size, 1.5, c, img1.device, img1.dtype)
    x, y = img1[None], img2[None]
    pad = window_size // 2
    mu1 = F.conv2d(x, w, padding=pad, groups=c)
    mu2 = F.conv2d(y, w, padding=pad, groups=c)
    s11 = F.conv2d(x * x, w, padding=pad, groups=c) - mu1 * mu1
    s22 = F.conv2d(y * y, w, padding=pad, groups=c) - mu2 * mu2
    s12 = F.conv2d(x * y, w, padding=pad, groups=c) - mu1 * mu2
    C1, C2 = 0.01**2, 0.03**2
    return (((2 * mu1 * mu2 + C1) * (2 * s12 + C2))
            / ((mu1 * mu1 + mu2 * mu2 + C1) * (s11 + s22 + C2))).mean()


def photometric(image, gt, lam=0.2):
    """(1 - lambda_dssim) L1 + lambda_dssim (1 - SSIM), CHW."""
    return (1.0 - lam) * (image - gt).abs().mean() + lam * (1.0 - ssim_same(image, gt))


def resize_target(target_hwc, height, width):
    """Observed image at the current scale (official: PIL resize of the source JPG; here an
    antialiased bicubic resize of the stored 600x400 image). Returns CHW."""
    img = target_hwc.permute(2, 0, 1)
    if img.shape[1:] == (height, width):
        return img
    out = F.interpolate(img[None], size=(height, width), mode="bicubic", antialias=True,
                        align_corners=False)[0]
    return out.clamp(0.0, 1.0)
