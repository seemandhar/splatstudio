"""Gaussian Surfels losses with the numerics of the official train.py / utils (commit 01d1249).

Paper (§3.3) vs code (used here, code overrides the paper):
  L_p   0.8 L1 + 0.2 D-SSIM (Eq. 9), GT composited with the step's random background
  L_m   paper: BCE, lambda 1; code: 0.1 * mean(opac * (1 - maxpool9(mask)))
  L_c   paper: lambda 0->0.1; code: (0.01 + 0.1 min(2 it / T, 1)) * cos_loss(N, depth2normal(D))
  L_n   paper: 0.04 (1 - N.N^) + 0.005 L1(grad N); code: (0.04 - 0.02 it / T) * cos_loss(N, N^)
        + 0.005 * mean(curv(N))
  L_o   paper: exp(-(o-0.5)^2/0.05), lambda 0.01; code: 0.01 * mean(exp(-20 (o-0.5)^2) * w),
        w = 0.01 on (0.01, 0.5], 1 on (0.5, 0.99], 0 elsewhere
All image tensors are CHW like the official code.
"""

import numpy as np
import torch
import torch.nn.functional as F

PHOTO_DEFAULTS = {"lambda_dssim": 0.2}
MASK_DEFAULTS = {"weight": 0.1, "pool": 9}
CONS_DEFAULTS = {"weight_start": 0.01, "weight_ramp": 0.1, "ramp_fraction": 0.5,
                 "total_steps": 15000}
PRIOR_DEFAULTS = {"weight_start": 0.04, "weight_decay": 0.02, "curvature_weight": 0.005,
                  "total_steps": 15000, "use_prior": True}
OPAC_DEFAULTS = {"weight": 0.01, "sharpness": 20.0, "low_band_weight": 0.01}


def _check(cfg, defaults, name):
    unknown = set(cfg) - set(defaults)
    if unknown:
        raise ValueError(f"Unknown {name} options: {sorted(unknown)}")
    return {**defaults, **cfg}


# ---------------------------------------------------------------- helpers (utils/*.py)
def _gaussian_window(size=11, sigma=1.5, channels=3, device=None):
    x = torch.arange(size, dtype=torch.float32, device=device) - size // 2
    g = torch.exp(-(x**2) / (2 * sigma**2))
    g = g / g.sum()
    return (g[:, None] @ g[None, :])[None, None].expand(channels, 1, size, size).contiguous()


def ssim(img1, img2):
    """utils/loss_utils.ssim: CHW, 11x11 Gaussian sigma 1.5, same padding, mean."""
    x, y = img1[None], img2[None]
    c = x.shape[1]
    w = _gaussian_window(channels=c, device=x.device)
    mu1, mu2 = F.conv2d(x, w, padding=5, groups=c), F.conv2d(y, w, padding=5, groups=c)
    s11 = F.conv2d(x * x, w, padding=5, groups=c) - mu1**2
    s22 = F.conv2d(y * y, w, padding=5, groups=c) - mu2**2
    s12 = F.conv2d(x * y, w, padding=5, groups=c) - mu1 * mu2
    c1, c2 = 0.01**2, 0.03**2
    return (((2 * mu1 * mu2 + c1) * (2 * s12 + c2))
            / ((mu1**2 + mu2**2 + c1) * (s11 + s22 + c2))).mean()


def cos_loss(output, gt, thrsh=0.0, weight=1.0):
    """utils/loss_utils.cos_loss: mean of (1 - cos) over pixels with cos < cos(thrsh)."""
    cos = torch.sum(output * gt * weight, 0)
    sel = cos[cos < np.cos(thrsh)]
    if sel.numel() == 0:  # official returns NaN (mean of empty); keep the graph and add nothing
        return output.sum() * 0.0
    return (1 - sel).mean()


def _focals(camera):
    K = camera["K"]
    return float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])


def _cam_points(depth, camera):
    """depth2normal/depth2wpos back-projection: integer pixel grid, (u - cx) * D / f with the
    official focal swap (K00 = fov2focal(FoVy, H) = fy for x, K11 = fx for y)."""
    fx, fy, cx, cy = _focals(camera)
    H, W = depth.shape[-2:]
    v, u = torch.meshgrid(torch.arange(H, device=depth.device, dtype=torch.float32),
                          torch.arange(W, device=depth.device, dtype=torch.float32), indexing="ij")
    d = depth[0]
    x = (u - cx) * d / fy
    y = (v - cy) * d / fx
    return torch.stack([x, y, d], -1)  # [H,W,3]


def depth2normal(depth, mask, camera):
    """utils/image_utils.depth2normal: sum of the four neighbour cross products, masked."""
    p = _cam_points(depth, camera)
    m = mask[0][..., None].float()
    p = F.pad(p[None].permute(0, 3, 1, 2), [1, 1, 1, 1], mode="replicate")[0].permute(1, 2, 0)
    m = F.pad(m[None].permute(0, 3, 1, 2), [1, 1, 1, 1], mode="replicate")[0].permute(1, 2, 0) > 0
    pc = p[1:-1, 1:-1] * m[1:-1, 1:-1]
    pu = (p[:-2, 1:-1] - pc) * m[:-2, 1:-1]
    pl = (p[1:-1, :-2] - pc) * m[1:-1, :-2]
    pb = (p[2:, 1:-1] - pc) * m[2:, 1:-1]
    pr = (p[1:-1, 2:] - pc) * m[1:-1, 2:]
    n = (torch.cross(pu, pl, dim=-1) + torch.cross(pr, pu, dim=-1)
         + torch.cross(pb, pr, dim=-1) + torch.cross(pl, pb, dim=-1))
    n = F.normalize(n, dim=-1)
    return (n * m[1:-1, 1:-1]).permute(2, 0, 1)


def normal2curv(normal, mask):
    """utils/image_utils.normal2curv: L1 norm over channels of sum_4nb (n_nb - n_c), masked."""
    n = F.pad(normal[None], [1, 1, 1, 1], mode="replicate")[0]
    m = F.pad(mask[None].float(), [1, 1, 1, 1], mode="replicate")[0] > 0
    nc = n[:, 1:-1, 1:-1] * m[:, 1:-1, 1:-1]
    nu = (n[:, :-2, 1:-1] - nc) * m[:, :-2, 1:-1]
    nl = (n[:, 1:-1, :-2] - nc) * m[:, 1:-1, :-2]
    nb = (n[:, 2:, 1:-1] - nc) * m[:, 2:, 1:-1]
    nr = (n[:, 1:-1, 2:] - nc) * m[:, 1:-1, 2:]
    curv = (nu + nl + nb + nr) * mask
    return curv.abs().sum(0, keepdim=True)


def prepared(outputs, target):
    """Per-step tensors shared by the loss terms (computed once, cached in outputs['gs'])."""
    gs = outputs["gs"]
    if "_prep" in gs:
        return gs["_prep"]
    view, bg = gs["view"], gs["background"]
    if view is not None:
        mask = view["mask"]
        base = view["image"] if view["image"] is not None else target.permute(2, 0, 1)
        if view["image"] is not None:
            base = base * mask  # get_gtImage: image * mask + bg * (1 - mask)
        gt = base + bg[:, None, None] * (1 - mask)
        mono = view["mono"] * mask  # train.py: mono *= mask_gt
    else:
        mask = torch.ones_like(gs["opac"])
        gt = target.permute(2, 0, 1)
        mono = None
    mask_vis = gs["opac"].detach() > 1e-5
    normal = F.normalize(gs["normal"], dim=0) * mask_vis
    gs["_prep"] = {"gt": gt, "mask": mask, "mono": mono, "mask_vis": mask_vis, "normal": normal}
    return gs["_prep"]


# ---------------------------------------------------------------- components
def create_photometric(spec):
    cfg = _check(spec.config, PHOTO_DEFAULTS, "photometric")
    lam = float(cfg["lambda_dssim"])

    def loss(params, outputs, target):
        gt = prepared(outputs, target)["gt"]
        img = outputs["gs"]["color"]
        return (1 - lam) * (img - gt).abs().mean() + lam * (1 - ssim(img, gt))

    return loss


def create_mask(spec):
    cfg = _check(spec.config, MASK_DEFAULTS, "mask")
    k = int(cfg["pool"])
    pool = torch.nn.MaxPool2d(k, stride=1, padding=k // 2)

    def loss(params, outputs, target):
        mask = prepared(outputs, target)["mask"]
        return float(cfg["weight"]) * (outputs["gs"]["opac"] * (1 - pool(mask[None])[0])).mean()

    return loss


def consistency_weight(step, cfg):
    return cfg["weight_start"] + cfg["weight_ramp"] * min(
        step / (cfg["ramp_fraction"] * cfg["total_steps"]), 1.0)


def create_consistency(spec):
    cfg = _check(spec.config, CONS_DEFAULTS, "consistency")

    def loss(params, outputs, target):
        p = prepared(outputs, target)
        gs = outputs["gs"]
        d2n = depth2normal(gs["depth"], p["mask_vis"], gs["camera"])
        return consistency_weight(outputs["step"], cfg) * cos_loss(p["normal"], d2n)

    return loss


def prior_weight(step, cfg):
    return cfg["weight_start"] - (step / cfg["total_steps"]) * cfg["weight_decay"]


def create_normal_prior(spec):
    cfg = _check(spec.config, PRIOR_DEFAULTS, "normal-prior")

    def loss(params, outputs, target):
        p = prepared(outputs, target)
        total = float(cfg["curvature_weight"]) * normal2curv(p["normal"], p["mask_vis"]).mean()
        if cfg["use_prior"] and p["mono"] is not None:
            total = total + prior_weight(outputs["step"], cfg) * cos_loss(
                p["normal"], p["mono"][:3], weight=p["mask"])
        return total

    return loss


def opacity_term(opac, cfg):
    band0 = (opac > 0.01) & (opac <= 0.5)
    band1 = (opac > 0.5) & (opac <= 0.99)
    w = band0.float() * cfg["low_band_weight"] + band1.float()
    return (torch.exp(-((opac - 0.5) ** 2) * cfg["sharpness"]) * w).mean()


def create_opacity(spec):
    cfg = _check(spec.config, OPAC_DEFAULTS, "opacity")

    def loss(params, outputs, target):
        return float(cfg["weight"]) * opacity_term(params["opacities"].sigmoid(), cfg)

    return loss
