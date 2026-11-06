"""GaussianPro losses (paper Eq. 8-10; official train.py:197-246 @ 86bb048).

* photometric: Eq. 10 image term with the official opacity mask — L1 over the pixels whose
  rendered opacity is > 0.999 before ``propagated_iteration_begin`` (when propagation is on),
  > 0 afterwards; the SSIM map is averaged over the same mask. The SSIM map uses the runtime's
  valid 11x11 convolution (our gsplat baseline convention), the mask cropped accordingly.
* normal (Eq. 8, β): lambda_l1 · mean_Q ||N̂ − N̄||_1 (channel sum) + lambda_cos · mean_Q (1 − N̂·N̄),
  Q = pixels whose propagated normal survived the geometric filter (≠ −10).
* flatten (Eq. 9, γ L_scale, NeuSG): lambda · mean_i clamp(min_j s_ij, 0, 30).
"""

import torch
import torch.nn.functional as F

from .geometry import INVALID_NORMAL

PHOTO_DEFAULTS = {"ssim_weight": 0.2, "mask_threshold_early": 0.999, "mask_threshold": 0.0,
                  "early_until": 1000, "masked": True}
NORMAL_DEFAULTS = {"lambda_l1_normal": 0.001, "lambda_cos_normal": 0.001}
FLATTEN_DEFAULTS = {"lambda_flatten": 100.0}


def _cfg(spec, defaults, label):
    unknown = set(spec.config) - set(defaults)
    if unknown:
        raise ValueError(f"Unknown {label} options: {sorted(unknown)}")
    return {**defaults, **spec.config}


def ssim_map(image, target):
    """SplatStudio ``losses.ssim`` without the final mean: [3, H-10, W-10]."""
    coords = torch.arange(11, device=image.device, dtype=image.dtype) - 5
    gauss = torch.exp(-(coords**2) / (2 * 1.5**2))
    gauss = gauss / gauss.sum()
    window = (gauss[:, None] * gauss[None, :])[None, None].expand(3, 1, 11, 11)
    x, y = (t.permute(2, 0, 1)[None] for t in (image, target))

    def conv(t):
        return F.conv2d(t, window, groups=3)

    mx, my = conv(x), conv(y)
    vx, vy, cov = conv(x * x) - mx * mx, conv(y * y) - my * my, conv(x * y) - mx * my
    return (((2 * mx * my + 0.01**2) * (2 * cov + 0.03**2))
            / ((mx * mx + my * my + 0.01**2) * (vx + vy + 0.03**2)))[0]


def opacity_mask(alpha, step, c):
    """train.py:197-203: [H,W] bool mask from rendered opacity."""
    thr = c["mask_threshold_early"] if step < c["early_until"] else c["mask_threshold"]
    return alpha[..., 0] > thr


def masked_photometric(rgb, target, alpha, step, c):
    w = float(c["ssim_weight"])
    if not c["masked"]:
        mask = torch.ones(rgb.shape[:2], dtype=torch.bool, device=rgb.device)
    else:
        mask = opacity_mask(alpha.detach(), step, c)
    m3 = mask[..., None].expand_as(rgb)
    if int(mask.sum()) == 0:
        # official: mean of an empty selection is NaN; keep the graph finite with a zero term
        l1 = (rgb - target).abs().sum() * 0.0
    else:
        l1 = (rgb[m3] - target[m3]).abs().mean()
    smap = ssim_map(rgb, target)
    sm = mask[5:-5, 5:-5][None].expand_as(smap)
    ssim = smap[sm].mean() if int(sm.sum()) else smap.mean()
    return (1 - w) * l1 + w * (1 - ssim)


def normal_loss(normal, target, c):
    """Eq. 8 as coded: both terms are means over valid pixels, weighted separately."""
    valid = target[..., 0] != INVALID_NORMAL
    if int(valid.sum()) == 0:
        return normal.sum() * 0.0
    l1 = (normal - target).abs().sum(dim=-1)[valid].mean()
    cos = (1.0 - (normal * target).sum(dim=-1))[valid].mean()
    return c["lambda_l1_normal"] * l1 + c["lambda_cos_normal"] * cos


def flatten_loss(log_scales, c):
    min_scale = log_scales.exp().min(dim=1).values.clamp(0, 30)
    return c["lambda_flatten"] * min_scale.abs().mean()


def create_photometric(spec):
    c = _cfg(spec, PHOTO_DEFAULTS, "GaussianPro photometric")
    if not 0 <= c["ssim_weight"] <= 1:
        raise ValueError("ssim_weight must be in [0, 1]")

    def loss(params, outputs, target):
        return masked_photometric(outputs["rgb"], target, outputs["alpha"], outputs["step"], c)

    return loss


def create_normal(spec):
    c = _cfg(spec, NORMAL_DEFAULTS, "GaussianPro normal")

    def loss(params, outputs, target):
        if "normal_cam" not in outputs:
            return params["quats"].sum() * 0.0  # view not propagated yet: no target (official)
        return normal_loss(outputs["normal_cam"], outputs["normal_target"], c)

    return loss


def create_flatten(spec):
    c = _cfg(spec, FLATTEN_DEFAULTS, "GaussianPro flatten")

    def loss(params, outputs, target):
        return flatten_loss(params["scales"], c)

    return loss
