"""CoherentGS objective (Eq. 7 with the official train.py numerics).

photometric: 0.8 * ML1 + 0.2 * (1 - SSIM) + 0.05 * VGG   (utils/loss_utils.py ML1Loss, ssim, VGGLoss)
  ML1  = sum_k w_k L1(pool^k(I), pool^k(I*)), w = [1, 1/4, 1/8, 1/16, 1/32], pool = AvgPool2d(2)
  SSIM = 3DGS ssim: 11x11 Gaussian (sigma 1.5), zero "same" padding, mean
  VGG  = (1/32) L1(relu1_1(I), relu1_1(I*)) of torchvision VGG19 (only the first slice is used:
         the official loop breaks after i = 0), raw [0, 1] input
smooth (Eq. 4-5): 5 * [gamma tv(S . disp) + (1 - gamma) tv(disp)], disp = 1/(1 + depth/(alpha+1e-6)),
  gamma = official iteration / 20000 (code; the paper states 0 -> 1)
flow (Eq. 6): 0.01 * warping_loss(xyz)  (code multiplies warp_coeff = 0.1 twice; paper beta_f 0.1)
"""

import hashlib
import math
from pathlib import Path

import torch
import torch.nn.functional as F

from .bank import tv

PHOTO_DEFAULTS = {"lambda_dssim": 0.2, "ml1_weights": [1.0, 1 / 4, 1 / 8, 1 / 16, 1 / 32],
                  "vgg_weight": 0.05, "vgg_layer_weight": 1 / 32,
                  "vgg_weights": str(Path.home() / "models/coherentgs/vgg19_conv1_1.pt"),
                  "vgg_sha256": "01808af48259fbd2305323a60c10a1e6b0320a6e101999e5ee2f7c174fefc1e1"}
SMOOTH_DEFAULTS = {"weight": 5.0, "iteration_offset": 7000, "gamma_denominator": 20000}
FLOW_DEFAULTS = {"weight": 0.1 * 0.1}


def ml1(x, y, weights):
    loss = 0
    for k, w in enumerate(weights):
        if k:  # official pools after every term; the last (unused) pool is skipped
            x, y = F.avg_pool2d(x, 2), F.avg_pool2d(y, 2)
        loss = loss + w * F.l1_loss(x, y)
    return loss


def ssim_same(img1, img2, window_size=11, sigma=1.5):
    """Official loss_utils.ssim on [1, C, H, W] inputs (zero 'same' padding)."""
    c = img1.shape[1]
    coords = torch.arange(window_size, dtype=img1.dtype, device=img1.device) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    window = (g[:, None] @ g[None, :])[None, None].expand(c, 1, window_size, window_size)
    pad = window_size // 2

    def conv(t):
        return F.conv2d(t, window, padding=pad, groups=c)

    mu1, mu2 = conv(img1), conv(img2)
    s1 = conv(img1 * img1) - mu1 ** 2
    s2 = conv(img2 * img2) - mu2 ** 2
    s12 = conv(img1 * img2) - mu1 * mu2
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    return (((2 * mu1 * mu2 + c1) * (2 * s12 + c2))
            / ((mu1 ** 2 + mu2 ** 2 + c1) * (s1 + s2 + c2))).mean()


class VGGRelu11(torch.nn.Module):
    """torchvision VGG19 features[0:2] (conv1_1 + ReLU) with the ImageNet weights."""

    def __init__(self, path, sha256):
        super().__init__()
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"VGG19 conv1_1 weights {path} missing (extracted from "
                                    "vgg19-dcbb9e9d.pth, see README)")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != sha256:
            raise ValueError(f"unexpected VGG19 conv1_1 weights hash {digest}")
        sd = torch.load(path, map_location="cpu", weights_only=True)
        self.conv = torch.nn.Conv2d(3, 64, 3, padding=1)
        self.conv.weight.data.copy_(sd["features.0.weight"])
        self.conv.bias.data.copy_(sd["features.0.bias"])
        self.requires_grad_(False)

    def forward(self, x):
        return F.relu(self.conv(x))


def create_photometric(spec):
    cfg = {**PHOTO_DEFAULTS, **spec.config}
    lam, wv, wl = float(cfg["lambda_dssim"]), float(cfg["vgg_weight"]), float(cfg["vgg_layer_weight"])
    weights = [float(w) for w in cfg["ml1_weights"]]
    vgg = {}

    def loss(params, outputs, target):
        x = outputs["rgb"].permute(2, 0, 1)[None]
        y = target.permute(2, 0, 1)[None]
        out = (1 - lam) * ml1(x, y, weights) + lam * (1 - ssim_same(x, y))
        if wv > 0:
            if x.device not in vgg:
                vgg[x.device] = VGGRelu11(cfg["vgg_weights"], cfg["vgg_sha256"]).to(x.device)
            net = vgg[x.device]
            out = out + wv * wl * F.l1_loss(net(x), net(y).detach())
        return out

    return loss


def smooth_value(disparity, mask, gamma):
    c = mask.shape[0]
    return gamma * tv(mask * disparity.repeat(c, 1, 1)) + (1 - gamma) * tv(disparity)


def create_smooth(spec):
    cfg = {**SMOOTH_DEFAULTS, **spec.config}
    weight = float(cfg["weight"])

    def loss(params, outputs, target):
        co = outputs["coherent"]
        gamma = (int(cfg["iteration_offset"]) + outputs["step"]) / float(cfg["gamma_denominator"])
        mask = co["bank"].dep_mask[co["view"]]
        return weight * smooth_value(co["disparity"], mask, gamma)

    return loss


def create_flow(spec):
    cfg = {**FLOW_DEFAULTS, **spec.config}
    weight = float(cfg["weight"])
    if not math.isfinite(weight) or weight < 0:
        raise ValueError("flow weight must be finite and >= 0")

    def loss(params, outputs, target):
        co = outputs["coherent"]
        return weight * co["bank"].warping_loss(co["xyz"])

    return loss
