"""Trick-GS objective (paper Eq. 7): (1-l) L1 + l D-SSIM + lambda_m L_m + lambda_sh L_sh.

Photometric target transforms (DS + BL): the full-resolution training image is downsampled to
the render size with EAGLES `downsample_image` numerics (bilinear, align_corners=False, no
antialias) and blurred with the scheduled Gaussian kernel (torchvision GaussianBlur numerics:
kernel exp(-x^2/(2 sigma^2)) on x in [-r, r], normalised, separable, reflect padding).
D-SSIM: 3DGS `ssim` (11x11 Gaussian, sigma 1.5, zero "same" padding, mean) evaluated with two
1-D convolutions as in Taming 3DGS's fused SSIM (AT, §3.2.5).
"""

import torch
import torch.nn.functional as F

from .renderer import GS_MASK, SH_MASK

SH_WEIGHTS = (3 / 15, 5 / 15, 7 / 15)  # (2l+1) / ((k+1)^2 - 1), k = 3 (RDO-Gaussian Eq. 8)


def gaussian_kernel1d(radius, sigma, device=None, dtype=torch.float32):
    x = torch.linspace(-radius, radius, 2 * radius + 1, device=device, dtype=dtype)
    k = torch.exp(-0.5 * (x / sigma) ** 2)
    return k / k.sum()


def blur_image(image, radius, sigma):
    """HWC image blurred with a separable (2r+1)^2 Gaussian, reflect padding."""
    k = gaussian_kernel1d(radius, sigma, image.device, image.dtype)
    x = image.permute(2, 0, 1)[None]
    c = x.shape[1]
    x = F.pad(x, (radius, radius, radius, radius), mode="reflect")
    x = F.conv2d(x, k.view(1, 1, 1, -1).expand(c, 1, 1, -1), groups=c)
    x = F.conv2d(x, k.view(1, 1, -1, 1).expand(c, 1, -1, 1), groups=c)
    return x[0].permute(1, 2, 0)


def downsample_target(target, height, width):
    if target.shape[0] == height and target.shape[1] == width:
        return target
    x = target.permute(2, 0, 1)[None]
    y = F.interpolate(x, size=(height, width), mode="bilinear", align_corners=False)
    return y[0].permute(1, 2, 0)


def transformed_target(target, trick, height, width):
    gt = downsample_target(target, height, width)
    if trick is not None and trick.get("blur") is not None:
        gt = blur_image(gt, *trick["blur"])
    return gt


_WINDOW = {}


def ssim_separable(image, target, window=11, sigma=1.5):
    """3DGS same-padded SSIM computed with separable 1-D Gaussian convolutions."""
    key = (image.device, image.dtype)
    if key not in _WINDOW:
        coords = torch.arange(window, device=image.device, dtype=image.dtype) - window // 2
        g = torch.exp(-(coords**2) / (2 * sigma**2))
        _WINDOW[key] = g / g.sum()
    g = _WINDOW[key]
    pad = window // 2
    x = torch.stack([image, target, image * image, target * target, image * target])
    x = x.permute(0, 3, 1, 2).reshape(1, -1, *image.shape[:2])
    c = x.shape[1]
    y = F.conv2d(x, g.view(1, 1, 1, -1).expand(c, 1, 1, -1), padding=(0, pad), groups=c)
    y = F.conv2d(y, g.view(1, 1, -1, 1).expand(c, 1, -1, 1), padding=(pad, 0), groups=c)
    mx, my, xx, yy, xy = y.reshape(5, 3, *image.shape[:2])
    vx, vy, cov = xx - mx * mx, yy - my * my, xy - mx * my
    c1, c2 = 0.01**2, 0.03**2
    return (((2 * mx * my + c1) * (2 * cov + c2)) / ((mx * mx + my * my + c1) * (vx + vy + c2))).mean()


def create_photometric(spec):
    unknown = set(spec.config) - {"ssim_weight"}
    if unknown:
        raise ValueError(f"Unknown trick photometric options: {sorted(unknown)}")
    weight = float(spec.config.get("ssim_weight", 0.2))
    if not 0 <= weight <= 1:
        raise ValueError("ssim_weight must be in [0, 1]")

    def loss(params, outputs, target):
        rgb = outputs["rgb"]
        gt = transformed_target(target, outputs.get("trick"), rgb.shape[0], rgb.shape[1])
        return (1 - weight) * F.l1_loss(rgb, gt) + weight * (1 - ssim_separable(rgb, gt))

    return loss


def gaussian_mask_term(params):
    """L_m = mean_n sigmoid(m_n) (Compact-3DGS train.py)."""
    return torch.sigmoid(params[GS_MASK]).mean()


def sh_mask_term(params):
    """L_sh = 1/N sum_i sum_l (2l+1)/15 sigmoid(m^l_i) (RDO-Gaussian Eq. 8)."""
    w = torch.tensor(SH_WEIGHTS, device=params[SH_MASK].device)
    return (torch.sigmoid(params[SH_MASK]) * w).sum(dim=1).mean()


def create_gaussian_mask(spec):
    unknown = set(spec.config) - {"weight"}
    if unknown:
        raise ValueError(f"Unknown gaussian-mask options: {sorted(unknown)}")
    weight = float(spec.config.get("weight", 0.05))

    def loss(params, outputs, target):
        return weight * gaussian_mask_term(params)

    return loss


def create_sh_mask(spec):
    unknown = set(spec.config) - {"weight"}
    if unknown:
        raise ValueError(f"Unknown sh-mask options: {sorted(unknown)}")
    weight = float(spec.config.get("weight", 0.05))

    def loss(params, outputs, target):
        trick = outputs.get("trick") or {}
        if not trick.get("sh_mask_active", True):
            return params[SH_MASK].sum() * 0.0
        return weight * sh_mask_term(params)

    return loss
