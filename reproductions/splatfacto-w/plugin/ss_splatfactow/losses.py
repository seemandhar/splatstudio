"""Splatfacto-W losses (official ``SplatfactoWModel.get_loss_dict`` / ``robust_mask``).

``sfw-photometric``: 0.8 L1 + 0.2 (1 - SSIM) on the robust-masked images (C2, §4.2), target
area-downscaled to the render resolution (splatfacto ``resize_image``); left-half (masked test)
views are zero-padded back to full width = the code's ``gt * mask, pred * mask``.
``sfw-alpha``: L_alpha = 0.15 * mean(alpha[box3(mean_c |gt - bg| < 0.003) > 0.6]) (C3, §4.3).
"""

import torch
import torch.nn.functional as F
from splatstudio.losses import ssim

PHOTOMETRIC_DEFAULTS = {
    "ssim_lambda": 0.2,
    "robust_mask": True,
    "robust_mask_percentage": [0.0, 0.40],
    "robust_mask_reset_interval": 6000,
    "never_mask_upper": 0.4,
    "start_robust_mask_at": 6000,
    "box_size": 5,
    "box_threshold": 0.4,
}
ALPHA_DEFAULTS = {"weight": 0.15, "threshold": 0.003, "box_size": 3, "select": 0.6}


def box_filter(x, f):
    """[H, W, 1] float -> [H, W, 1], f x f mean filter, zero 'same' padding (torch conv2d)."""
    w = torch.ones((1, 1, f, f), device=x.device, dtype=x.dtype) / (f * f)
    return F.conv2d(x.permute(2, 0, 1)[None], w, stride=1, padding="same")[0].permute(1, 2, 0)


def downscale_target(target, d, shape):
    if d == 1:
        return target
    t = F.avg_pool2d(target.permute(2, 0, 1)[None], d, stride=d)[0].permute(1, 2, 0)
    return t[: shape[0], : shape[1]]


def pad_right(x, width):
    return F.pad(x, (0, 0, 0, width - x.shape[1]))


class RobustMask:
    """Stateful robust mask (running max / min of the image L1)."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.max_loss = 0.0
        self.min_loss = 1e10

    @torch.no_grad()
    def __call__(self, errors, k):
        c = self.cfg
        errors = errors.clone()
        errors[: int(errors.shape[0] * float(c["never_mask_upper"]))] = 0.0  # upper rows inliers
        l1 = float(errors.mean())
        if l1 > self.max_loss or k % int(c["robust_mask_reset_interval"]) == 0:
            self.max_loss = l1
        self.min_loss = min(self.min_loss, l1)
        lo, hi = (float(v) for v in c["robust_mask_percentage"])
        pct = (l1 - self.min_loss) / ((self.max_loss - self.min_loss) + 1e-6) * (hi - lo) + lo
        per_pixel = errors.mean(dim=-1, keepdim=True)
        thr = torch.quantile(per_pixel, 1 - pct)
        inlier = (per_pixel <= thr).float()
        neighbours = (box_filter(inlier, int(c["box_size"])) > float(c["box_threshold"])).float()
        return ((neighbours + inlier) > 1e-3).float(), pct


def photometric_terms(rgb, gt, k, robust, cfg, masked=False):
    """Returns (loss, masked gt (unpadded), robust mask or None, mask percentage or None)."""
    w = rgb.shape[1]
    if masked:
        rgb, gt = pad_right(rgb, 2 * w), pad_right(gt, 2 * w)
    errors = (gt - rgb).abs()
    mask = pct = None
    if robust is not None and cfg["robust_mask"] and k >= int(cfg["start_robust_mask_at"]):
        mask, pct = robust(errors.detach(), k)
        gt, rgb = gt * mask, rgb * mask
        l1 = (errors * mask).mean()
    else:
        l1 = errors.mean()
    lam = float(cfg["ssim_lambda"])
    loss = (1 - lam) * l1 + lam * (1 - ssim(rgb, gt))
    return loss, gt[:, :w], mask, pct


def create_photometric(spec):
    unknown = set(spec.config) - set(PHOTOMETRIC_DEFAULTS)
    if unknown:
        raise ValueError(f"Unknown sfw-photometric options: {sorted(unknown)}")
    cfg = {**PHOTOMETRIC_DEFAULTS, **spec.config}
    robust = RobustMask(cfg)

    def loss(params, outputs, target):
        rgb = outputs["rgb"]
        k = int(outputs.get("step", 1)) - 1
        gt = downscale_target(target, int(outputs.get("downscale", 1)), rgb.shape)
        value, gt_m, _mask, pct = photometric_terms(rgb, gt, k, robust, cfg,
                                                   bool(outputs.get("masked_view", False)))
        outputs["sfw_gt"] = gt_m
        if pct is not None:
            outputs["sfw_mask_percentage"] = pct
        return value

    loss.robust = robust
    return loss


def alpha_selection(gt, background, cfg):
    near = ((gt - background).abs().mean(dim=-1, keepdim=True) < float(cfg["threshold"])).float()
    return box_filter(near, int(cfg["box_size"])) > float(cfg["select"])


def alpha_loss_value(alpha, gt, background, cfg, sel=None):
    if sel is None:
        sel = alpha_selection(gt, background, cfg)
    if not bool(sel.any()):
        return alpha.sum() * 0.0
    return float(cfg["weight"]) * alpha[sel].mean()


def create_alpha_loss(spec):
    unknown = set(spec.config) - set(ALPHA_DEFAULTS)
    if unknown:
        raise ValueError(f"Unknown sfw-alpha options: {sorted(unknown)}")
    cfg = {**ALPHA_DEFAULTS, **spec.config}

    def loss(params, outputs, target):
        gt = outputs.get("sfw_gt")
        if gt is None:  # used without sfw-photometric (probes): plain downscaled target
            gt = downscale_target(target, int(outputs.get("downscale", 1)), outputs["rgb"].shape)
        sel = alpha_selection(gt, outputs["background"], cfg)
        outputs["sfw_alpha_selected"] = int(sel.sum())
        outputs["sfw_alpha_pixels"] = int(sel.numel())
        return alpha_loss_value(outputs["alpha"], gt, outputs["background"], cfg, sel)

    return loss


def create_alpha_diagnostic(spec):
    """Diagnostic only (no gradient): fraction of pixels selected by the alpha loss this step,
    so that training.jsonl shows whether C3's alpha loss is ever active. Its value adds a
    gradient-free constant (<= 1) to the logged total ``loss``; must follow ``sfw-alpha``."""
    if spec.config:
        raise ValueError("sfw-alpha-selected-frac takes no options")

    def loss(params, outputs, target):
        if "sfw_alpha_selected" not in outputs:
            raise RuntimeError("sfw-alpha-selected-frac must come after sfw-alpha")
        frac = outputs["sfw_alpha_selected"] / max(outputs["sfw_alpha_pixels"], 1)
        return outputs["alpha"].sum() * 0.0 + frac

    return loss
