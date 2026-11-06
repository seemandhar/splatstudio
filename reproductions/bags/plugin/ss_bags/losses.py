"""BAGS losses (train.py of github.com/snldmt/BAGS @ed04144, lines "if iteration > 250").

For a training view with the BPN active (phase-local iteration > 250):
    D_n   = (D - min D) / max(D - min D)
    h, m  = BPN_s(l(i), p(x), [C, D_n].detach())             (Eq. 2 / 4, kernel size K_s)
    C~    = h (*) C   (spatially varying, Unfold)             (Eq. 1)
    C_out = m C~ + (1 - m) C                                  (Eq. 3)
    L     = 0.8 L1(C_out, C_obs^s) + 0.2 (1 - SSIM(C_out, C_obs^s))     (Eq. 5)
          + 0.001 mean(m)                       [bags-mask, use_mask_loss]
          + alpha_d TV(D_n)                     [bags-depth-tv, use_depth_loss]
          + 0.001 TV(C)                         [bags-rgb-tv, use_rgbtv_loss]
Otherwise (iteration <= 250 of either phase, or a non-training camera): the 3DGS loss on the
sharp render C. C_obs^s is the observed image resized to the render's resolution.
The BPN forward is computed once per step and cached in outputs["bags"].
"""

import torch

from .blur import normalized_depth, photometric, resize_target, spatial_conv, tv_loss
from .schedule import bpn_active

OBJECTIVE_DEFAULTS = {"lambda_dssim": 0.2}
MASK_DEFAULTS = {"use_mask_loss": True, "mask_loss_alpha": 0.001}
DEPTH_TV_DEFAULTS = {"use_depth_loss": False, "depth_loss_alpha": 0.001}
RGB_TV_DEFAULTS = {"use_rgbtv_loss": False, "rgbtv_loss_alpha": 0.001}


def _chw(t):
    return t.permute(2, 0, 1)


def active(outputs):
    b = outputs.get("bags") or {}
    bank = b.get("bank")
    if bank is None or b.get("view") is None:
        return False
    return bpn_active(outputs["step"], bank.cfg["ms_steps"], bank.cfg["bpn_start"])


def bpn_forward(outputs):
    """Cached BPN result: dict(blur=C_out [3,H,W], mask [1,H,W], kernel [K^2,H,W], K, depth_n)."""
    b = outputs["bags"]
    if "bpn" not in b:
        bank, view, step = b["bank"], b["view"], outputs["step"]
        rgb = _chw(outputs["rgb"])
        depth_n = normalized_depth(_chw(outputs["depth"]))
        H, W = rgb.shape[1:]
        net, level, K = bank.net_for(step)
        weight, mask = net(view, bank.pos_enc(H, W),
                           torch.cat([rgb, depth_n], 0)[None].detach(), level)
        conv = spatial_conv(rgb, weight[0], K)
        mask = mask[0]
        blur = mask * conv + (1 - mask) * rgb
        b["bpn"] = {"blur": blur, "mask": mask, "kernel": weight[0], "K": K, "depth_n": depth_n}
    return b["bpn"]


def _check(spec, defaults, kind):
    unknown = set(spec.config) - set(defaults)
    if unknown:
        raise ValueError(f"Unknown BAGS {kind} options: {sorted(unknown)}")
    return {**defaults, **spec.config}


def create_objective(spec):
    lam = float(_check(spec, OBJECTIVE_DEFAULTS, "objective")["lambda_dssim"])

    def loss(params, outputs, target):
        rgb = _chw(outputs["rgb"])
        gt = resize_target(target, rgb.shape[1], rgb.shape[2])
        image = bpn_forward(outputs)["blur"] if active(outputs) else rgb
        return photometric(image, gt, lam)

    return loss


def _zero(outputs):
    return outputs["rgb"].sum() * 0.0


def create_mask_loss(spec):
    c = _check(spec, MASK_DEFAULTS, "mask-loss")
    use, alpha = bool(c["use_mask_loss"]), float(c["mask_loss_alpha"])

    def loss(params, outputs, target):
        if not (use and active(outputs)):
            return _zero(outputs)
        return alpha * bpn_forward(outputs)["mask"].mean()

    return loss


def create_depth_tv(spec):
    c = _check(spec, DEPTH_TV_DEFAULTS, "depth-TV")
    use, alpha = bool(c["use_depth_loss"]), float(c["depth_loss_alpha"])

    def loss(params, outputs, target):
        if not (use and active(outputs)):
            return _zero(outputs)
        return alpha * tv_loss(bpn_forward(outputs)["depth_n"][None])

    return loss


def create_rgb_tv(spec):
    c = _check(spec, RGB_TV_DEFAULTS, "rgb-TV")
    use, alpha = bool(c["use_rgbtv_loss"]), float(c["rgbtv_loss_alpha"])

    def loss(params, outputs, target):
        if not (use and active(outputs)):
            return _zero(outputs)
        return alpha * tv_loss(_chw(outputs["rgb"])[None])

    return loss
