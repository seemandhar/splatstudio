"""FSGS geometry guidance (C3) on training views and pseudo views (C2); official train.py:95-131.

Training view (every step):
    L_d = min(1 - corr(-D_dpt, D_ras), 1 - corr(1 / (D_dpt + 200), D_ras)),
    weight depth_weight = 0.05; official sets it to 0.001 *after* adding the loss at it > 9500,
    so 0.001 applies from step 9502. D_dpt = DPT(gt image), computed once per training view
    without gradient (official camera_utils.loadCam -> estimate_depth(gt_image)).
Pseudo view (step % interval == 0, 2000 < step < 9500):
    L_p = 1 - corr(D_ras_pseudo, -DPT(render_pseudo)) with gradient through DPT,
    weight min((step - 2000) / 500, 1) * depth_pseudo_weight; skipped if NaN.
corr = torchmetrics pearson_corrcoef (Eq. 6), clamped to [-1, 1].
"""

import torch

from .dpt import DEFAULT_PATHS, estimate_depth, load_dpt
from .renderer import rasterize

DEPTH_DEFAULTS = {"weight": 0.05, "late_weight": 0.001, "late_after": 9500, "shift": 200.0,
                  **DEFAULT_PATHS}
PSEUDO_DEFAULTS = {"weight": 0.5, "start": 2000, "end": 9500, "interval": 1, "ramp": 500,
                   **DEFAULT_PATHS}
PATH_KEYS = tuple(DEFAULT_PATHS)


def pearson(x, y):
    """Pearson correlation of two flattened tensors (torchmetrics pearson_corrcoef, clamped)."""
    x = x.reshape(-1)
    y = y.reshape(-1).to(x.dtype)
    xm, ym = x - x.mean(), y - y.mean()
    corr = (xm * ym).sum() / ((xm * xm).sum() * (ym * ym).sum()).sqrt()
    return corr.clamp(-1.0, 1.0)


def train_depth_loss(rendered, dpt, shift=200.0):
    return torch.minimum(1 - pearson(-dpt, rendered), 1 - pearson(1 / (dpt + shift), rendered))


def depth_weight(step, weight, late_weight, late_after):
    return late_weight if step > late_after + 1 else weight


def pseudo_scale(step, start, end, interval, ramp):
    """Official gate and ramp; 0 when inactive."""
    if step % interval != 0 or not (start < step < end):
        return 0.0
    return min((step - start) / float(ramp), 1.0)


def _check(cfg, defaults, name):
    unknown = set(cfg) - set(defaults)
    if unknown:
        raise ValueError(f"Unknown {name} options: {sorted(unknown)}")
    return {**defaults, **cfg}


def view_key(camera):
    return tuple(round(v, 6) for v in camera["viewmat"].detach().reshape(-1).tolist())


def create_depth(spec):
    cfg = _check(dict(spec.config), DEPTH_DEFAULTS, "fsgs-depth")
    cache = {}

    def loss(params, outputs, target):
        camera = outputs["camera"]
        key = view_key(camera)
        if key not in cache:
            model = load_dpt(target.device, {k: cfg[k] for k in PATH_KEYS})
            cache[key] = estimate_depth(model, target.permute(2, 0, 1).contiguous(), "test")
        w = depth_weight(int(outputs["step"]), float(cfg["weight"]), float(cfg["late_weight"]),
                         int(cfg["late_after"]))
        return w * train_depth_loss(outputs["depth"][..., 0], cache[key], float(cfg["shift"]))

    loss.cache = cache
    return loss


def create_pseudo(spec):
    cfg = _check(dict(spec.config), PSEUDO_DEFAULTS, "fsgs-pseudo")
    stats = {"used": 0, "skipped_nan": 0}

    def loss(params, outputs, target):
        step = int(outputs["step"])
        scale = pseudo_scale(step, int(cfg["start"]), int(cfg["end"]), int(cfg["interval"]),
                             int(cfg["ramp"]))
        zero = outputs["depth"].sum() * 0.0
        if scale == 0.0:
            return zero
        density = outputs.get("fsgs")
        if density is None:
            raise RuntimeError("fsgs-pseudo needs the fsgs-density component (pseudo cameras)")
        camera = density.next_pseudo_camera()
        out = rasterize(params, camera, outputs["sh_degree"], outputs["background"],
                        outputs["st_alpha_clamp"])
        model = load_dpt(out["rgb"].device, {k: cfg[k] for k in PATH_KEYS})
        dpt = estimate_depth(model, out["rgb"].permute(2, 0, 1), "train")
        term = 1 - pearson(out["depth"][..., 0], -dpt)
        if torch.isnan(term):
            stats["skipped_nan"] += 1
            return zero
        stats["used"] += 1
        return scale * float(cfg["weight"]) * term

    loss.stats = stats
    return loss
