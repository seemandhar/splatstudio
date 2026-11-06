"""Pure functions of RobustSplat's mask pipeline (official ``utils/mask_utils.py``,
``utils/loss_utils.py`` and ``train.py`` of github.com/fcyycf/RobustSplat @ a130281).

Image tensors are CHW like the official (Inria) code.
"""

import math

import torch
import torch.nn.functional as F
from torch import nn


# ------------------------------------------------------------------ SpotLessSplats residual masks
def generate_mask(residual, threshold):
    """``generate_mask``: residual [H,W] -> static mask [1,H,W]; inlier = residual < thr, mask =
    inlier OR (3x3 zero-padded box mean of inliers > 0.5)."""
    inlier = (residual < threshold).float()[None, None]
    window = torch.ones((1, 1, 3, 3), dtype=inlier.dtype, device=inlier.device) / 9.0
    neighbours = F.conv2d(inlier, window, padding=1)
    return (((neighbours > 0.5).float() + inlier) > 1e-3).float()[0]


def histogram_threshold(hist, fraction):
    """``linspace(0, 1, bins + 1)[first k with cumsum(hist)[k] >= fraction * sum(hist)]``."""
    bins = hist.numel()
    k = torch.nonzero(torch.cumsum(hist, 0) >= hist.sum() * fraction)[0, 0]
    return k.to(hist.dtype) / bins


def residual_masks(gt, render, hist, lower=0.6, upper=0.8, decay=0.95):
    """``calculate_residual_mask``: returns (lower_mask [1,H,W], upper_mask [1,H,W], new hist).

    residual = mean_c |gt - render| (detached); hist <- decay * hist + histogram(residual,
    bins, [0, 1]); thresholds at the ``lower`` / ``upper`` cumulative fractions."""
    residual = (gt - render).detach().abs().mean(dim=0)
    new = torch.histc(residual.float(), bins=hist.numel(), min=0.0, max=1.0)
    hist = decay * hist + new
    lo = generate_mask(residual, histogram_threshold(hist, lower))
    up = generate_mask(residual, histogram_threshold(hist, upper))
    return lo, up, hist


def residual_hinge(mask, lower_mask, upper_mask):
    """``MLPModel.get_residual_loss``: mean(relu(M - upper) + relu(lower - M)) (supp. Eq. S1)."""
    return torch.mean(torch.relu(mask - upper_mask) + torch.relu(lower_mask - mask))


def upsample(t, height, width):
    """Official ``interpolation``: bilinear, align_corners=False, [C,h,w] -> [C,H,W]."""
    return F.interpolate(t[None], size=(height, width), mode="bilinear")[0]


# ------------------------------------------------------------------ feature similarity (Eq. 4)
def cosine_target(gt_feat, render_feat, height, width):
    """Official: c = clip(1 - (cos - 0.5) / 0.5, 0, 1) at the feature grid, upsampled, M_cos =
    1 - up(c) (== up(max(2 cos - 1, 0)), Eq. 4). Returns [1,H,W]."""
    cos = F.cosine_similarity(gt_feat, render_feat, dim=0)[None]
    c = (1.0 - cos.sub(0.5).div(0.5)).clip(0.0, 1.0)
    return 1.0 - upsample(c, height, width)


def reg_weight(step, beta):
    """Eq. 7 decay e^{-i / beta_reg}."""
    return math.exp(-float(step) / float(beta))


# ------------------------------------------------------------------ mask MLP (Eq. 3)
class MaskMLP(nn.Module):
    """``MLPModel``: Linear(384,16) - ReLU - Linear(16,1) - Sigmoid per pixel of the bilinearly
    upsampled DINOv2 feature map."""

    def __init__(self, num_features=384, hidden=16):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(num_features, hidden), nn.ReLU(),
                                 nn.Linear(hidden, 1), nn.Sigmoid())

    def forward_naive(self, feat, height, width):
        """Official order: upsample all 384 channels, then the MLP per pixel -> [1,H,W]."""
        up = upsample(feat, height, width)
        x = up.reshape(up.shape[0], -1).permute(1, 0)
        return self.mlp(x).reshape(height, width, 1).permute(2, 0, 1)

    def forward(self, feat, height, width):
        """Same function with the first Linear applied before the bilinear upsampling (linear,
        weights summing to 1, so W up(F) + b == up(W F) + b exactly; 24x less memory)."""
        lin1, lin2 = self.mlp[0], self.mlp[2]
        low = torch.einsum("oc,chw->ohw", lin1.weight, feat)
        hid = torch.relu(upsample(low, height, width) + lin1.bias[:, None, None])
        out = torch.einsum("oc,chw->ohw", lin2.weight, hid) + lin2.bias[:, None, None]
        return torch.sigmoid(out)


def loss_mask(mask, threshold=0.25, kernel=7):
    """Official: m = (M.detach() > 0.25), eroded by a k x k min-pool (= dilated transient
    region, supp. "dilation with kernel size 7"). [1,H,W] -> [1,H,W]."""
    m = (mask.detach() > threshold).float()
    return -F.max_pool2d(-m[None], kernel_size=kernel, stride=1, padding=kernel // 2)[0]


# ------------------------------------------------------------------ Inria SSIM
def _gaussian_window(size, sigma, channels, device, dtype):
    x = torch.arange(size, dtype=torch.float64) - size // 2
    g = torch.exp(-(x ** 2) / (2 * sigma ** 2))
    g = (g / g.sum()).to(dtype)
    w2 = g[:, None] @ g[None, :]
    return w2.expand(channels, 1, size, size).contiguous().to(device)


def ssim_map(img1, img2, window_size=11):
    """Inria ``utils.loss_utils._ssim`` map (Gaussian 11x11, sigma 1.5, same zero padding)."""
    c = img1.shape[0]
    w = _gaussian_window(window_size, 1.5, c, img1.device, img1.dtype)
    a, b = img1[None], img2[None]
    pad = window_size // 2
    mu1 = F.conv2d(a, w, padding=pad, groups=c)
    mu2 = F.conv2d(b, w, padding=pad, groups=c)
    mu1_sq, mu2_sq, mu12 = mu1 * mu1, mu2 * mu2, mu1 * mu2
    s1 = F.conv2d(a * a, w, padding=pad, groups=c) - mu1_sq
    s2 = F.conv2d(b * b, w, padding=pad, groups=c) - mu2_sq
    s12 = F.conv2d(a * b, w, padding=pad, groups=c) - mu12
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    return ((2 * mu12 + C1) * (2 * s12 + C2)) / ((mu1_sq + mu2_sq + C1) * (s1 + s2 + C2))


def ssim(img1, img2):
    return ssim_map(img1, img2).mean()


def masked_photometric(image, gt, mask, lambda_dssim=0.2, threshold=0.25, kernel=7):
    """Official masked loss: m = erode(M.detach() > thr); (1 - l) mean(m |I - I*|) +
    l mean(1 - SSIM_map(m I, m I*)). Returns (loss, m)."""
    m = loss_mask(mask, threshold, kernel)
    l1 = (m * (image - gt).abs()).mean()
    lssim = (1.0 - ssim_map(m * image, m * gt)).mean()
    return (1.0 - lambda_dssim) * l1 + lambda_dssim * lssim, m
