"""WE-GS losses (paper §4.1 Eq. 13, §4.3 Eq. 19-22). Weights and reductions: paper.md §3/§7."""

import torch
import torch.nn.functional as F
from splatstudio.losses import ssim

# Eq. 19. ssim_detach_mask (DEVIATION, paper silent): M is detached inside the D-SSIM term only;
# D-SSIM of M(x)I^r vs M(x)I^gt -> 0 as M -> 0 and alone drives the literal form to M = 0
# (evidence/smoke-1, reviews/review-1.md table). The L1 term and L_regM still train M.
MASKED_DEFAULTS = {"l1_weight": 0.8, "ssim_weight": 0.2, "ssim_detach_mask": True}
# §5.1 lambda_1 (L_regM), literal Eq. 20: squared Frobenius norm over M_k at W' x H' = W/4 x H/4
# ("quarter"). "full" (on R(1 - M) at W x H, ~40x stronger) leaves the mask inert: ablation only.
MASK_REG_DEFAULTS = {"weight": 1e-5, "reduction": "sum", "resolution": "quarter"}
SH_REG_DEFAULTS = {"weight": 0.1, "reduction": "mean"}        # §5.1 lambda_2 (L_regSH)
TS_REG_DEFAULTS = {"weight": 1e-5, "eps": 0.1}  # §5.1 lambda_3 (L_regTS); eps: paper silent, paper.md §8


def _cfg(spec, defaults, what):
    unknown = set(spec.config) - set(defaults)
    if unknown:
        raise ValueError(f"Unknown {what} options: {sorted(unknown)}")
    return {**defaults, **spec.config}


def masked_photometric(rgb, target, mask_full, l1_weight=0.8, ssim_weight=0.2,
                       ssim_detach_mask=True):
    """Eq. 19 on M (x) I^r and M (x) I^gt (mask_full [H, W, 1]); same value either way, the
    D-SSIM term optionally sees a detached M (see MASKED_DEFAULTS)."""
    a, b = mask_full * rgb, mask_full * target
    ms = mask_full.detach() if ssim_detach_mask else mask_full
    return l1_weight * F.l1_loss(a, b) + ssim_weight * (1 - ssim(ms * rgb, ms * target))


def mask_regulariser(mask, reduction="sum"):
    """Eq. 20: ||1 - M_k||^2 (squared Frobenius norm for reduction "sum")."""
    v = (1 - mask).square()
    return v.sum() if reduction == "sum" else v.mean()


def sh_regulariser(delta, reduction="mean"):
    """Eq. 13 for the current image k: per-Gaussian L2 norm of Delta sh, reduced over Gaussians."""
    n = delta.reshape(len(delta), -1).norm(dim=1)
    return n.mean() if reduction == "mean" else n.sum()


def projected_mask_values(mask, means2d, width, height):
    """M(Pi(mu_i)) by bilinear lookup of the 1/4-res mask at pixel coordinates means2d [N, 2]."""
    x = 2 * means2d[:, 0] / float(width) - 1
    y = 2 * means2d[:, 1] / float(height) - 1
    grid = torch.stack([x, y], -1)[None, None]
    return F.grid_sample(mask, grid, mode="bilinear", padding_mode="border",
                         align_corners=False)[0, 0, 0]


def transient_opacity_regulariser(opacity, mask, info, width, height, eps=0.5):
    """Eq. 21: sum_i alpha_i * 1[M(Pi(mu_i)) < eps] over Gaussians visible in this view."""
    n = len(opacity)
    m2d = info["means2d"].reshape(-1, n, 2)[0].detach()
    r = info["radii"]
    if r.dim() >= 2 and r.shape[-1] == 2 and r.shape[-2] == n:  # gsplat >= 1.5: [..., N, 2]
        visible = (r.reshape(-1, n, 2)[0] > 0).all(-1)
    else:
        visible = r.reshape(-1, n)[0] > 0
    inside = ((m2d[:, 0] >= 0) & (m2d[:, 0] < width) & (m2d[:, 1] >= 0) & (m2d[:, 1] < height))
    with torch.no_grad():
        mv = projected_mask_values(mask.detach(), m2d, width, height)
    ind = (visible & inside & (mv < float(eps))).to(opacity.dtype)
    return (opacity * ind).sum(), ind


def create_masked_photometric(spec):
    c = _cfg(spec, MASKED_DEFAULTS, "wegs-masked-photometric")

    def loss(params, outputs, target):
        return masked_photometric(outputs["rgb"], target, outputs["static_mask_full"],
                                  float(c["l1_weight"]), float(c["ssim_weight"]),
                                  bool(c["ssim_detach_mask"]))

    return loss


def create_mask_reg(spec):
    c = _cfg(spec, MASK_REG_DEFAULTS, "wegs-mask-reg")

    if c["resolution"] not in ("full", "quarter"):
        raise ValueError("wegs-mask-reg resolution must be 'full' or 'quarter'")
    key = "static_mask_full" if c["resolution"] == "full" else "static_mask"

    def loss(params, outputs, target):
        return float(c["weight"]) * mask_regulariser(outputs[key], c["reduction"])

    return loss


def create_sh_reg(spec):
    c = _cfg(spec, SH_REG_DEFAULTS, "wegs-sh-reg")

    def loss(params, outputs, target):
        return float(c["weight"]) * sh_regulariser(outputs["sh_residual"], c["reduction"])

    return loss


def create_transient_reg(spec):
    c = _cfg(spec, TS_REG_DEFAULTS, "wegs-transient-opacity-reg")

    def loss(params, outputs, target):
        h, w = target.shape[:2]
        v, ind = transient_opacity_regulariser(outputs["opacity"], outputs["static_mask"],
                                               outputs["info"], w, h, float(c["eps"]))
        outputs["transient_gaussians"] = int(ind.sum())
        outputs["static_mask_mean"] = float(outputs["static_mask"].detach().mean())
        return float(c["weight"]) * v

    return loss
