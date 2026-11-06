"""DINO-based uncertainty of WildGaussians (paper Sec. 3.3, Eq. 8-10; official method.py
`UncertaintyModel`, uncertainty_mode "dino"). Images are [B, 3, H, W] in [0, 1]."""

import itertools
import math

import torch
import torch.nn.functional as F
from torch import nn

IMG_MEAN = (0.485, 0.456, 0.406)  # (123.675, 116.28, 103.53) / 255
IMG_STD = (0.229, 0.224, 0.225)  # (58.395, 57.12, 57.375) / 255


def dino_downsample(x, max_size=None):
    """official dino_downsample: scale so that max(h, w) <= max_size, round up to /14, bilinear."""
    if max_size is None:
        return x
    h, w = x.shape[2:]
    if max_size < h or max_size < w:
        s = min(max_size / h, max_size / w)
        nh, nw = int(h * s), int(w * s)
        nh, nw = ((nh + 13) // 14) * 14, ((nw + 13) // 14) * 14
        x = F.interpolate(x, size=(nh, nw), mode="bilinear")
    return x


def patch_pad(size, patch=14):
    new = math.ceil(size / patch) * patch
    left = (new - size) // 2
    return left, new - size - left


def pads_for(x, patch=14):
    """[w_left, w_right, h_top, h_bottom] as in UncertaintyModel (x.shape[:1:-1] = (W, H))."""
    return list(itertools.chain.from_iterable(patch_pad(m, patch) for m in x.shape[:1:-1]))


def normalize(x):
    mean = torch.tensor(IMG_MEAN, device=x.device, dtype=x.dtype)[None, :, None, None]
    std = torch.tensor(IMG_STD, device=x.device, dtype=x.dtype)[None, :, None, None]
    return (x - mean) / std


def dino_part_from_cosine(cosine):
    """Eq. 8 as coded: clip(1 - (cos - 0.5) / 0.5, 0, 1) = min(1, 2 - 2 cos) for cos <= 1."""
    return (1 - cosine.sub(0.5).div(0.5)).clip(0.0, 1)


class UncertaintyHead(nn.Module):
    """Dropout2d(0.5) -> BatchNorm2d(384) -> Conv1x1(384 -> 1) on the DINO features, + log(e-1),
    softplus, bilinear upsample to the padded size, clamp_min(clip_min), crop."""

    def __init__(self, in_features=384, dropout=0.5, clip_min=0.1, generator=None):
        super().__init__()
        self.dropout, self.clip_min = float(dropout), float(clip_min)
        self.conv_seg = nn.Conv2d(in_features, 1, kernel_size=1)
        # official: nn.SyncBatchNorm, which is plain batch norm without a process group
        self.bn = nn.BatchNorm2d(in_features)
        with torch.no_grad():
            self.conv_seg.weight.normal_(0, 0.01, generator=generator)
            self.conv_seg.bias.zero_()

    def forward(self, features, padded_hw, pads, hw):
        x = F.dropout2d(features, p=self.dropout, training=self.training)
        logits = self.conv_seg(self.bn(x)) + math.log(math.exp(1) - 1)
        sigma = F.softplus(logits)
        sigma = F.interpolate(sigma, size=padded_hw, mode="bilinear", align_corners=False)
        sigma = sigma.clamp(min=self.clip_min)
        h, w = hw
        return sigma[:, :, pads[2]:h + pads[2], pads[0]:w + pads[0]]


class DinoFeatures:
    """Frozen backbone + per-key CPU cache of GT features (official `_get_dino_cached`)."""

    def __init__(self, backbone):
        self.backbone = backbone
        self.cache = {}

    @torch.no_grad()
    def __call__(self, x_padded_normalised, key=None):
        k = None if key is None else (key, tuple(x_padded_normalised.shape))
        if k is not None and k in self.cache:
            return self.cache[k].to(x_padded_normalised.device)
        f = self.backbone.intermediate(x_padded_normalised)
        if k is not None:
            self.cache[k] = f.detach().cpu()
        return f


def predict_sigma(head, feats, gt, key=None):
    """_forward_uncertainty_features on the full-resolution GT (uncertainty_dino_max_size None)."""
    h, w = gt.shape[2:]
    pads = pads_for(gt)
    x = F.pad(normalize(gt), pads)
    f = feats(x, None if key is None else ("unc", key))
    return head(f, x.shape[2:], pads, (h, w))


@torch.no_grad()
def dino_cosine(feats, gt, pred, key=None):
    """_compute_cosine_similarity(gt, pred) (max_size None, inputs already downsampled)."""
    h, w = gt.shape[2:]
    pads = pads_for(gt)
    x = F.pad(normalize(gt), pads)
    y = F.pad(normalize(pred), pads)
    fx = feats(x, None if key is None else ("cos", key))
    fy = feats(y, None)
    cos = F.cosine_similarity(fx, fy, dim=1).unsqueeze(1)
    cos = F.interpolate(cos, size=x.shape[2:], mode="bilinear", align_corners=False)
    return cos[:, :, pads[2]:h + pads[2], pads[0]:w + pads[0]].squeeze(1)


def uncertainty_terms(head, feats, gt, pred, key=None, prior_weight=0.5, dino_max=350,
                      mult_clip=3.0):
    """UncertaintyModel._compute_losses (mode "dino"). gt/pred [1,3,H,W]; pred is detached.
    Returns (loss, loss_mult [1,1,H,W] clamped at mult_clip, sigma)."""
    sigma = predict_sigma(head, feats, gt, key)
    loss_mult = 1 / (2 * sigma.pow(2))
    gt_down = dino_downsample(gt, dino_max)
    pred_down = dino_downsample(pred.detach(), dino_max)
    dino_part = dino_part_from_cosine(dino_cosine(feats, gt_down, pred_down, key)).detach()
    unc_loss = dino_part * dino_downsample(loss_mult, dino_max)
    loss = unc_loss.mean() + prior_weight * torch.log(sigma).mean()
    return loss, loss_mult.clamp_max(mult_clip), sigma


def warmup_mask(mask, iteration, start, iters):
    """train_iteration: 1 before `start`, then 1 + p (M - 1) over `iters` iterations."""
    if iteration < start:
        return torch.ones_like(mask)
    if iteration < start + iters:
        p = (iteration - start) / iters
        return 1 + p * (mask - 1)
    return mask


def protected(iteration, densify_until, reset_interval, protected_iters):
    """Detach window of the uncertainty loss after an opacity reset (official train_iteration)."""
    last = min(iteration, densify_until - 1)
    last = (last // reset_interval) * reset_interval
    return iteration < last + protected_iters
