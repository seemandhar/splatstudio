"""DeblurGS objective (official train.py:139-153; paper Eq. 8-10).

    loss = L1(blur, gamma^-1(gt)) + lambda_t(step) * L1(sub[1:], sub[:-1])
           + lambda_hinge * hinge_l2(opacity_raw)

* L1 = mean |.| (utils/loss_utils.l1_loss); the blur is the linear sub-frame average and the
  target the linearised observation gt^2.2 (scene.tone_mapping.inverse()). Paper Eq. 7/8 apply
  gamma to the render instead; the code's linear-space form is used.
* temporal smoothness (Eq. 9): code = mean absolute difference of adjacent sub-frames
  (batchwise_smoothness_loss), lambda_t log-linear 1e-3 -> 1e-5 over 150k (paper: L2, 0.05 ->
  0.01). Zero when only one sub-frame is rendered (step < curve_start_iter).
* hinge (code only): mean of x^2 (x <= 0), (x-1)^2 (x >= 1) over the raw opacities, which the
  renderer clamps to [0, 1].
No D-SSIM term (code has none).
"""

import torch

from .bank import expon
from .renderer import inverse_tonemap

RGB_DEFAULTS = {"gamma": 2.2, "tone_eps": 1e-8}
SMOOTH_DEFAULTS = {"lambda_t_smooth_init": 1e-3, "lambda_t_smooth_final": 1e-5,
                   "total_steps": 150000}
HINGE_DEFAULTS = {"lambda_hinge": 0.1}


def _cfg(spec, defaults):
    unknown = set(spec.config) - set(defaults)
    if unknown:
        raise ValueError(f"Unknown {spec.name} options: {sorted(unknown)}")
    return {**defaults, **spec.config}


def smoothness(subframes):
    """batchwise_smoothness_loss on [C,H,W,3] (identical mean to the official [C,3,H,W])."""
    if subframes.shape[0] == 1:
        return subframes.sum() * 0.0
    return (subframes[1:] - subframes[:-1]).abs().mean()


def hinge_l2(x):
    return torch.where(x <= 0.0, x * x, torch.where(x >= 1.0, (x - 1.0) ** 2,
                                                    torch.zeros_like(x))).mean()


def lambda_smooth(step, cfg):
    return expon(step, cfg["lambda_t_smooth_init"], cfg["lambda_t_smooth_final"],
                 cfg["total_steps"])


def create_rgb(spec):
    cfg = _cfg(spec, RGB_DEFAULTS)

    def loss(params, outputs, target):
        if "blur_linear" not in outputs:
            raise ValueError("deblurgs-rgb needs the training render of deblurgs-renderer")
        gt = inverse_tonemap(target, cfg["gamma"], cfg["tone_eps"])
        return (outputs["blur_linear"] - gt).abs().mean()  # Eq. 8 (code form)

    return loss


def create_smoothness(spec):
    cfg = _cfg(spec, SMOOTH_DEFAULTS)

    def loss(params, outputs, target):
        return lambda_smooth(outputs["step"], cfg) * smoothness(outputs["subframes"])  # Eq. 9

    return loss


def create_hinge(spec):
    cfg = _cfg(spec, HINGE_DEFAULTS)

    def loss(params, outputs, target):
        return cfg["lambda_hinge"] * hinge_l2(params["opacities"])

    return loss
