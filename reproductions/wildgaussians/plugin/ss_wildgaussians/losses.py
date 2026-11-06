"""WildGaussians losses (official WildGaussians.train_iteration).

* ``wg-uncertainty`` (Eq. 8-10): DINO-cosine uncertainty loss of the uncertainty head (GT = the
  training image, prediction = detached toned render); it also publishes the binary loss mask
  M = (min(1/(2 sigma^2), 3) > 1) with the warm-up schedule as ``outputs["wg_mask"]``. In the
  protected window after an opacity reset the loss is detached (the head gets no gradient and
  its Adam step is skipped, as for the official ``uncertainty_loss.detach()``). Must be listed
  before ``wg-color``.
* ``wg-color`` (Eq. 6 / 11): (1 - l) mean(M |toned - gt|) + l mean(M (1 - SSIM_map(raw, gt))),
  l = 0.2, 3DGS same-padded SSIM map averaged over channels.
"""

import torch
import torch.nn.functional as F

from . import bank as bank_mod
from .uncertainty import protected, uncertainty_terms, warmup_mask

UNC_DEFAULTS = {
    "prior_weight": 0.5,          # uncertainty_regularizer_weight
    "dino_max_size": 350,
    "mult_clip": 3.0,
    "warmup_start": 2000,         # uncertainty_warmup_start
    "warmup_iters": 0,            # uncertainty_warmup_iters
    "protected_iters": 500,       # uncertainty_protected_iters
    "densify_until_iter": 15000,
    "opacity_reset_interval": 3000,
}
COLOR_DEFAULTS = {"lambda_dssim": 0.2}


def ssim_map(img1, img2, window_size=11, sigma=1.5):
    """Official method.ssim(size_average=False) for [3, H, W]: same (zero) padding, mean over
    channels -> [H, W]."""
    coords = torch.arange(window_size, device=img1.device, dtype=img1.dtype) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    w = (g[:, None] * g[None, :])[None, None].expand(3, 1, window_size, window_size).contiguous()

    def conv(t):
        return F.conv2d(t, w, padding=window_size // 2, groups=3)

    mu1, mu2 = conv(img1), conv(img2)
    s11 = conv(img1 * img1) - mu1 ** 2
    s22 = conv(img2 * img2) - mu2 ** 2
    s12 = conv(img1 * img2) - mu1 * mu2
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    m = ((2 * mu1 * mu2 + c1) * (2 * s12 + c2)) / ((mu1 ** 2 + mu2 ** 2 + c1) * (s11 + s22 + c2))
    return m.mean(-3)


def chw(x):
    return x.permute(2, 0, 1)


def color_loss(toned, raw, gt, mask, lambda_dssim=0.2):
    """toned/raw/gt [3,H,W], mask [1,H,W] or scalar 1."""
    l1 = (toned - gt).abs()
    ssim_value = ssim_map(raw, gt)
    return ((1.0 - lambda_dssim) * (l1 * mask).mean()
            + lambda_dssim * ((1.0 - ssim_value) * mask).mean())


def create_uncertainty(spec):
    unknown = set(spec.config) - set(UNC_DEFAULTS)
    if unknown:
        raise ValueError(f"Unknown wg-uncertainty options: {sorted(unknown)}")
    cfg = {**UNC_DEFAULTS, **spec.config}

    def loss(params, outputs, target):
        bank = bank_mod.ACTIVE
        it = int(outputs["step"])
        if bank is None or not bank.uncertainty:
            raise RuntimeError("wg-uncertainty needs a live WildGaussians bank with uncertainty")
        idx = outputs.get("view_index")
        gt = chw(target)[None]
        pred = chw(outputs["rgb"]).detach()[None]
        value, mult, _ = uncertainty_terms(bank.head, bank.features, gt, pred, key=idx,
                                           prior_weight=cfg["prior_weight"],
                                           dino_max=cfg["dino_max_size"],
                                           mult_clip=cfg["mult_clip"])
        mask = (mult > 1).to(mult.dtype)[0]  # [1, H, W]
        outputs["wg_mask"] = warmup_mask(mask, it, cfg["warmup_start"], cfg["warmup_iters"])
        outputs["wg_mask_fraction"] = float(mask.mean())
        if protected(it, cfg["densify_until_iter"], cfg["opacity_reset_interval"],
                     cfg["protected_iters"]):
            # official: uncertainty_loss.detach(); the runtime requires a graph-attached scalar,
            # so attach a zero that touches no learnable state of the head
            return value.detach() + 0.0 * outputs["rgb"].sum()
        return value

    return loss


def create_color(spec):
    unknown = set(spec.config) - set(COLOR_DEFAULTS) - {"use_mask"}
    if unknown:
        raise ValueError(f"Unknown wg-color options: {sorted(unknown)}")
    lam = float(spec.config.get("lambda_dssim", 0.2))
    use_mask = bool(spec.config.get("use_mask", True))

    def loss(params, outputs, target):
        mask = 1.0
        if use_mask:
            if "wg_mask" not in outputs:
                raise RuntimeError("wg-color needs wg-uncertainty to run first (outputs.wg_mask)")
            mask = outputs["wg_mask"]
        return color_loss(chw(outputs["rgb"]), chw(outputs["rgb_raw"]), chw(target), mask, lam)

    return loss

