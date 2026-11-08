"""MSAA-GS objective (paper Eq. 6-13): L = 0.8 L_w + 0.2 L_D-SSIM + 0.1 L_grad.

`aws-recon` replaces the base `photometric` loss (0.8 L1 + 0.2 D-SSIM): its L1 becomes the
error-adaptive weighted L1. `gdc` is a separate component so training.jsonl logs it separately.
Reductions (no official code; paper.md §7): w uses e = per-pixel channel-mean |error| (w only
depends on e / max e, so sum vs mean is irrelevant there) and is detached; L_w averages over
pixels and channels (same scale as 3DGS's l1_loss); L_grad = mean|d_x| + mean|d_y| over the
(H-1)x(W-1) domain of Eq. 12.
"""

import torch
from splatstudio.losses import ssim

AWS_DEFAULTS = {"lambda_w": 0.8, "lambda_ssim": 0.2, "alpha": 0.5, "eps": 1e-6,
                "normalize_mean": False}
GDC_DEFAULTS = {"weight": 0.1}


def aws_weights(rgb, target, alpha=0.5, eps=1e-6):
    """Eq. 7: w_ij = alpha + (1 - alpha) e_ij / (max e + eps); HWC inputs -> [H,W,1] (detached)."""
    with torch.no_grad():
        e = (rgb - target).abs().mean(dim=-1, keepdim=True)  # Eq. 6 (up to the factor C)
        return alpha + (1.0 - alpha) * e / (e.max() + eps)


def weighted_l1(rgb, target, alpha=0.5, eps=1e-6, normalize_mean=False):
    """Eq. 8: L_w = 1/(HW) sum_ij w_ij |I_pred - I_gt|_1 (per-channel mean).

    normalize_mean (iteration 1, post-hoc, NOT in the paper): divide w by its image mean so the
    relative emphasis of Eq. 7 is kept but the L1 term keeps 3DGS's scale. With alpha 0.5 the
    raw weights average ~0.52 on kitchen, which halves the photometric gradient and with it the
    3DGS densification statistic (-24% Gaussians, gap-analysis-1.md), whereas the paper keeps
    3DGS's densification hyper-parameters and "a similar number of Gaussians as in 3DGS".
    """
    w = aws_weights(rgb, target, alpha, eps)
    if normalize_mean:
        w = w / w.mean()
    return (w * (rgb - target).abs()).mean()


def gradient_difference(rgb, target):
    """Eq. 10-12: forward differences on the (H-1)x(W-1) domain, l1 on both directions."""
    def grads(img):
        core = img[:-1, :-1]
        return img[:-1, 1:] - core, img[1:, :-1] - core  # nabla_x (j+1), nabla_y (i+1)

    px, py = grads(rgb)
    gx, gy = grads(target)
    return (px - gx).abs().mean() + (py - gy).abs().mean()


def create_recon(spec):
    cfg = {**AWS_DEFAULTS, **spec.config}
    unknown = set(cfg) - set(AWS_DEFAULTS)
    if unknown:
        raise ValueError(f"Unknown AWS loss options: {sorted(unknown)}")
    lw, ls, a, eps = (float(cfg[k]) for k in ("lambda_w", "lambda_ssim", "alpha", "eps"))
    norm = bool(cfg["normalize_mean"])
    if not 0 <= a <= 1 or eps <= 0 or lw < 0 or ls < 0:
        raise ValueError("need alpha in [0,1], eps > 0, non-negative weights")

    def loss(params, outputs, target):  # Eq. 9
        rgb = outputs["rgb"]
        return lw * weighted_l1(rgb, target, a, eps, norm) + ls * (1.0 - ssim(rgb, target))

    return loss


def create_gdc(spec):
    cfg = {**GDC_DEFAULTS, **spec.config}
    unknown = set(cfg) - set(GDC_DEFAULTS)
    if unknown:
        raise ValueError(f"Unknown GDC loss options: {sorted(unknown)}")
    weight = float(cfg["weight"])
    if weight < 0:
        raise ValueError("weight must be non-negative")

    def loss(params, outputs, target):  # Eq. 13, lambda_3 term
        return weight * gradient_difference(outputs["rgb"], target)

    return loss
