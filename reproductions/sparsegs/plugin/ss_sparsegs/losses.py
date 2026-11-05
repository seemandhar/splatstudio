"""SparseGS losses: Pearson depth correlation (C2), depth warping (C4), SDS (C3).

Numerics follow the official utils/loss_utils.py and train.py; see paper.md §2.
"""

import math

import torch
import torch.nn.functional as F

from .context import CTX
from .renderer import rasterize

PEARSON_DEFAULTS = {"global_weight": 0.15, "local_weight": 0.15, "patch": 64, "p_corr": 0.5}
WARP_DEFAULTS = {"weight": 0.05, "start": 4999, "prob": 0.4, "angles": [-2.5, 2.5],
                 "ssim_weight": 0.2, "near_plane": 0.2, "background": 0.0}
SDS_DEFAULTS = {"weight": 5e-4, "total_steps": 10000, "start_frac": 2.0 / 3.0, "freq": 0.1,
                "step_ratio": 0.99, "guidance_scale": 100.0, "near_plane": 0.2,
                "background": 0.0, "model_dir": "~/models/sd21base",
                "pylib": "~/models/sd21base/pylib", "seed": 0}


def _cfg(spec, defaults, what):
    unknown = set(spec.config) - set(defaults)
    if unknown:
        raise ValueError(f"Unknown SparseGS {what} options: {sorted(unknown)}")
    return {**defaults, **spec.config}


def _zero(params):
    return params["means"].sum() * 0.0


# ------------------------------------------------------------------ C2 Pearson
def pearson_loss(a, b):
    """loss_utils.pearson_depth_loss: 1 - mean(z(a) z(b)), z = (x-mean)/(std+1e-6)."""
    a = a - a.mean()
    b = b - b.mean()
    a = a / (a.std() + 1e-6)
    b = b / (b.std() + 1e-6)
    return 1 - (a * b).mean()


def pearson_rows(a, b):
    """pearson_depth_loss per row of [n, S*S] (unbiased std, as torch.std)."""
    a = a - a.mean(dim=1, keepdim=True)
    b = b - b.mean(dim=1, keepdim=True)
    a = a / (a.std(dim=1, keepdim=True) + 1e-6)
    b = b / (b.std(dim=1, keepdim=True) + 1e-6)
    return 1 - (a * b).mean(dim=1)


def n_patches(H, W, S, p_corr):
    return int(p_corr * math.floor(H / S) * math.floor(W / S))


def local_pearson_loss(depth, target, S, p_corr, generator=None):
    """loss_utils.local_pearson_loss: n random SxS patches (top-left randint(0, H-S),
    randint(0, W-S), may overlap), mean of per-patch Pearson losses (vectorised)."""
    H, W = depth.shape
    n = n_patches(H, W, S, p_corr)
    dev = depth.device
    x0 = torch.randint(0, H - S, (n,), device=dev, generator=generator)
    y0 = torch.randint(0, W - S, (n,), device=dev, generator=generator)
    ar = torch.arange(S, device=dev)
    rows = (x0[:, None, None] + ar[None, :, None]).expand(n, S, S)
    cols = (y0[:, None, None] + ar[None, None, :]).expand(n, S, S)
    p = depth[rows, cols].reshape(n, -1)
    q = target[rows, cols].reshape(n, -1)
    return pearson_rows(p, q).sum() / n


def create_pearson(spec):
    cfg = _cfg(spec, PEARSON_DEFAULTS, "pearson")
    S, p_corr = int(cfg["patch"]), float(cfg["p_corr"])
    gw, lw = float(cfg["global_weight"]), float(cfg["local_weight"])
    gen = {}

    def loss(params, outputs, target):
        step = outputs["step"]
        if CTX.draws(step)[1]:  # official: `and (not pick_warp_cam)`
            return _zero(params)
        depth = outputs["depth_softmax"]
        mono = CTX.frames().lookup(outputs["camera"]["viewmat"], depth.device)
        if depth.device not in gen:
            gen[depth.device] = torch.Generator(device=depth.device).manual_seed(0)
        total = _zero(params)
        if gw > 0:
            total = total + gw * pearson_loss(depth, mono)
        if lw > 0:
            total = total + lw * local_pearson_loss(depth, mono, S, p_corr, gen[depth.device])
        return total

    return loss


# ------------------------------------------------------------------ C4 depth warping
def ssim_inria(img1, img2, window_size=11):
    """utils/loss_utils.ssim on [1,3,H,W] (same padding, Gaussian sigma 1.5, mean)."""
    coords = torch.arange(window_size, device=img1.device, dtype=img1.dtype)
    g = torch.exp(-(coords - window_size // 2) ** 2 / (2 * 1.5 ** 2))
    g = g / g.sum()
    w = (g[:, None] @ g[None, :])[None, None].expand(3, 1, window_size, window_size)
    pad = window_size // 2

    def conv(t):
        return F.conv2d(t, w, padding=pad, groups=3)

    mu1, mu2 = conv(img1), conv(img2)
    s1 = conv(img1 * img1) - mu1 ** 2
    s2 = conv(img2 * img2) - mu2 ** 2
    s12 = conv(img1 * img2) - mu1 * mu2
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    return (((2 * mu1 * mu2 + C1) * (2 * s12 + C2))
            / ((mu1 ** 2 + mu2 ** 2 + C1) * (s1 + s2 + C2))).mean()


def mask_l1(out, gt, mask):
    """loss_utils.mask_l1_loss: sum(|out-gt| * mask) / sum(mask) (mask [H,W], 3 channels)."""
    return ((out - gt) * mask[..., None]).abs().sum() / mask.sum()


def warp_photometric(rgb, gt, mask, ssim_weight=0.2):
    """train.py reg_loss = 0.8 masked L1 + 0.2 (1 - SSIM) (SSIM unmasked). HWC inputs."""
    x = rgb.permute(2, 0, 1)[None]
    y = gt.permute(2, 0, 1)[None]
    return (1 - ssim_weight) * mask_l1(rgb, gt, mask) + ssim_weight * (1 - ssim_inria(x, y))


def create_warp(spec):
    cfg = _cfg(spec, WARP_DEFAULTS, "warp")
    weight = float(cfg["weight"])
    CTX.draw_cfg.update(warp_prob=float(cfg["prob"]), warp_enabled=weight > 0,
                        warp_start=int(cfg["start"]))
    CTX.warp_angles = [float(a) for a in cfg["angles"]]
    CTX.warp_start = int(cfg["start"])

    def loss(params, outputs, target):
        step = outputs["step"]
        if not CTX.draws(step)[1]:
            return _zero(params)
        if CTX.warp_cams is None:
            raise RuntimeError("SparseGS warp: warp targets were not generated (the density "
                               "builds them at the warp start step)")
        wc = CTX.next_warp()
        rgb, _, info = rasterize(params, wc["camera"], _sh_degree(step), cfg["near_plane"],
                                 cfg["background"])
        info["means2d"].retain_grad()
        from .renderer import inria_radii

        info["visibility_radii"] = inria_radii(params, wc["camera"], cfg["near_plane"])
        CTX.warp_info = (step, info)
        return weight * warp_photometric(rgb, wc["target"], wc["mask"], cfg["ssim_weight"])

    return loss


def _sh_degree(step):
    return min((step - 1) // 1000, 3)  # the runtime's SH schedule for this step


# ------------------------------------------------------------------ C3 SDS
def create_sds(spec):
    cfg = _cfg(spec, SDS_DEFAULTS, "sds")
    weight = float(cfg["weight"])
    start = float(cfg["total_steps"]) * float(cfg["start_frac"])
    CTX.draw_cfg.update(sds_freq=float(cfg["freq"]), sds_enabled=weight > 0, sds_start=start)
    state = {"guide": None}
    import numpy as np

    rng = np.random.default_rng(int(cfg["seed"]))

    def loss(params, outputs, target):
        step = outputs["step"]
        if not CTX.draws(step)[0]:
            return _zero(params)
        if state["guide"] is None:
            from .sds import StableDiffusionGuidance

            state["guide"] = StableDiffusionGuidance(cfg["model_dir"], cfg["pylib"],
                                                     params["means"].device)
        if CTX.ellipse is None:
            from .context import Ellipse

            if not CTX.cameras:
                raise RuntimeError("SparseGS SDS needs the training cameras")
            CTX.ellipse = Ellipse([c["viewmat"] for c in CTX.cameras])
        base = CTX.cameras[0]  # official: diff_cam = deepcopy(train camera 0) (intrinsics)
        cam = {"viewmat": CTX.ellipse.sample(rng).to(params["means"].device), "K": base["K"],
               "width": base["width"], "height": base["height"]}
        rgb, _, _ = rasterize(params, cam, _sh_degree(step), cfg["near_plane"], cfg["background"])
        return weight * state["guide"].train_step(rgb.permute(2, 0, 1)[None],
                                                  float(cfg["step_ratio"]),
                                                  float(cfg["guidance_scale"]))

    return loss
