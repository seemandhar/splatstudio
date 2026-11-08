"""CoR-GS losses for the second field and pseudo-view co-regularisation (C2).

Official train.py:157-176 (gaussiansN = 2):
    loss_i  = photometric(I_i, I*)                                      (Eq. 6, both fields)
    loss_i += photometric(I'_i, sg(I'_j)) / (N - 1)   for 500 <= it <= 10000, every iteration
                                                                        (Eq. 5, 7, lambda_p = 1)
photometric = (1 - 0.2) * L1 + 0.2 * (1 - SSIM). Field 1's Eq. 6 term is the runtime's built-in
`photometric` loss; `corgs-field2` adds field 2's Eq. 6 term (same camera, background and SH
degree as field 1's render); `corgs-coreg` adds both fields' Eq. 5 terms on one pseudo view per
step, rendered with the same random background. The official `loss_scale` ramp is computed but
never applied (train.py:178), so the weight is constant.
"""

import torch.nn.functional as F
from splatstudio.losses import ssim

from .renderer import rasterize

FIELD2_DEFAULTS = {"ssim_weight": 0.2}
COREG_DEFAULTS = {"weight": 1.0, "ssim_weight": 0.2, "start": 500, "end": 10000, "interval": 1}


def photometric(image, target, ssim_weight=0.2):
    return (1 - ssim_weight) * F.l1_loss(image, target) + ssim_weight * (1 - ssim(image, target))


def coreg_active(step, start, end, interval):
    """Official gate: it % interval == 0 and start <= it <= end_sample_pseudo."""
    return step % interval == 0 and start <= step <= end


def coreg_terms(rgb1, rgb2, ssim_weight=0.2):
    """Eq. 5 for both fields: target from the other field detached (train.py:174)."""
    return (photometric(rgb1, rgb2.detach(), ssim_weight),
            photometric(rgb2, rgb1.detach(), ssim_weight))


def _check(cfg, defaults, name):
    unknown = set(cfg) - set(defaults)
    if unknown:
        raise ValueError(f"Unknown {name} options: {sorted(unknown)}")
    return {**defaults, **cfg}


def _density(outputs, name):
    density = outputs.get("corgs")
    if density is None or density.field2 is None:
        raise RuntimeError(f"{name} needs corgs-density with n_fields=2")
    return density


def create_field2(spec):
    cfg = _check(dict(spec.config), FIELD2_DEFAULTS, "corgs-field2")

    def loss(params, outputs, target):
        density = _density(outputs, "corgs-field2")
        f2 = density.field2
        out = rasterize(f2.params, outputs["camera"], outputs["sh_degree"],
                        outputs["background"], outputs["st_alpha_clamp"])
        out["info"]["means2d"].retain_grad()
        f2.info = out["info"]
        return photometric(out["rgb"], target, float(cfg["ssim_weight"]))

    return loss


def create_coreg(spec):
    cfg = _check(dict(spec.config), COREG_DEFAULTS, "corgs-coreg")
    stats = {"used": 0, "terms": []}

    def loss(params, outputs, target):
        step = int(outputs["step"])
        if not coreg_active(step, int(cfg["start"]), int(cfg["end"]), int(cfg["interval"])):
            return outputs["rgb"].sum() * 0.0
        density = _density(outputs, "corgs-coreg")
        camera = density.next_pseudo_camera()
        bg, deg, st = outputs["background"], outputs["sh_degree"], outputs["st_alpha_clamp"]
        rgb1 = rasterize(params, camera, deg, bg, st)["rgb"]
        rgb2 = rasterize(density.field2.params, camera, deg, bg, st)["rgb"]
        r1, r2 = coreg_terms(rgb1, rgb2, float(cfg["ssim_weight"]))
        stats["used"] += 1
        if step % 1000 == 0:
            stats["terms"].append((step, float(r1), float(r2)))
        # lambda_p / (N - 1) with N = 2
        return float(cfg["weight"]) * (r1 + r2)

    loss.stats = stats
    return loss
