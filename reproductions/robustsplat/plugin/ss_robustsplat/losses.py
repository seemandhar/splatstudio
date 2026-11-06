"""RobustSplat losses (arXiv 2506.02751, numerics of the official ``train.py`` @ a130281).

``rs-masked-photometric``  Eq. 2 with the transient mask: for step < mask_start plain
    0.8 L1 + 0.2 D-SSIM; afterwards M = MLP(up(f_fine)) (Eq. 3), m = erode7(M.detach() > 0.25),
    0.8 mean(m|I - I*|) + 0.2 mean(1 - SSIM_map(m I, m I*)). Replaces SplatStudio's photometric.
``rs-mask-mlp``  MLP supervision (Eq. 4-8, supp. S1) with scale-cascaded bootstrapping (C2):
    0.5 mean|M - M_cos| + 0.5 hinge(M, lower, upper) + 2.0 mean((1 - M) e^{-i/2000});
    for step < bootstrap_iter the residual masks come from a coarse (images_{4f}) render vs the
    coarse GT and the features from a 16x16 grid (224² input), afterwards from the training
    render and a 36x36 grid (504²). Not trained for step < mask_start nor for
    reset_start < i < reset_start + 300 once i >= reset_iter (reset_start = 3000 floor(i/3000)).
    It reads the step-local values of ``rs-masked-photometric`` and must follow it.

Both components return graph-attached scalars: the photometric loss reaches only the Gaussians
(mask detached), the MLP loss only the MLP (renders detached), so the runtime's single backward
of their sum equals the official two separate backward passes. The MLP's Adam steps at the
start of the next call.
"""

import torch

from . import dino
from .context import Shared
from .masks import (
    cosine_target,
    masked_photometric,
    reg_weight,
    residual_hinge,
    residual_masks,
    ssim,
    upsample,
)
from .renderer import rasterize

PHOTO_DEFAULTS = {
    "lambda_dssim": 0.2,
    "mask_start": 500,          # code mask_beginning
    "mask_threshold": 0.25,
    "erode_kernel": 7,
    "feature_dim": 384,
    "mlp_hidden": 16,
    "mlp_lr": 1e-3,
    "bins": 10000,
    "fine_grid": 36,            # code upper_feat_res (504 = 36 x 14)
    "coarse_grid": 16,          # code lower_feat_res (224 = 16 x 14)
    "dino_weights": dino.DEFAULT_WEIGHTS,
}
MLP_DEFAULTS = {
    "lambda_residual": 0.5,
    "lambda_cos": 0.5,
    "lambda_reg": 2.0,
    "beta_reg": 2000.0,
    "bootstrap_iter": 20000,    # code; paper text: switch at densification start (10000)
    "bootstrap": True,          # False: fine supervision throughout (ablation "w/o MB")
    "lower_bound": 0.6,
    "upper_bound": 0.8,
    "hist_decay": 0.95,
    "mask_start": 500,
    "reset_iter": 15000,
    "reset_every": 3000,
    "pause": 300,
}


def _check(cfg, defaults, name):
    unknown = set(cfg) - set(defaults)
    if unknown:
        raise ValueError(f"Unknown {name} options: {sorted(unknown)}")


def mlp_active(step, cfg):
    """Official: train the MLP iff i >= mask_beginning and not (reset_start < i < reset_start +
    300 and i >= reset_iter)."""
    i = int(step)
    if i < int(cfg["mask_start"]):
        return False
    reset_start = i // int(cfg["reset_every"]) * int(cfg["reset_every"])
    return not (reset_start < i < reset_start + int(cfg["pause"]) and i >= int(cfg["reset_iter"]))


# ----------------------------------------------------------------------------- photometric
class MaskedPhotometric:
    def __init__(self, spec):
        cfg = {**PHOTO_DEFAULTS, **spec.config}
        _check(cfg, PHOTO_DEFAULTS, "rs-masked-photometric")
        self.cfg = cfg

    def __call__(self, params, outputs, target):
        cfg = self.cfg
        step = int(outputs["step"])
        image = outputs["rgb"].permute(2, 0, 1)                 # [3,H,W], unclamped (official)
        gt = target.permute(2, 0, 1)
        device = image.device
        Shared.apply_pending_step()
        lam = float(cfg["lambda_dssim"])
        shared = {"step": step}
        outputs["robustsplat"] = shared
        if step < int(cfg["mask_start"]):
            l1 = (image - gt).abs().mean()
            return (1.0 - lam) * l1 + lam * (1.0 - ssim(image, gt))
        mlp = Shared.ensure_mlp(device, cfg)
        table = Shared.ensure_table(device, cfg)
        entry = table.lookup(outputs["camera"]["viewmat"])
        H, W = image.shape[1:]
        mask = mlp(entry["feat_fine"], H, W)                     # [1,H,W], graph -> MLP
        loss, m = masked_photometric(image, gt, mask, lam, float(cfg["mask_threshold"]),
                                     int(cfg["erode_kernel"]))
        shared.update({"mask": mask, "entry": entry, "image": image, "gt": gt,
                       "dino": dino.load_backbone(cfg["dino_weights"], device=device),
                       "grids": (int(cfg["fine_grid"]), int(cfg["coarse_grid"])),
                       "static_fraction": float(m.mean())})
        return loss


def create_photometric(spec):
    return MaskedPhotometric(spec)


# ----------------------------------------------------------------------------- mask MLP
class MaskMLPLoss:
    def __init__(self, spec):
        cfg = {**MLP_DEFAULTS, **spec.config}
        _check(cfg, MLP_DEFAULTS, "rs-mask-mlp")
        self.cfg = cfg
        self.last = {}

    def supervision(self, params, outputs, shared):
        """Residual masks [1,H,W] x2 and cosine target [1,H,W] for this step (C2)."""
        cfg = self.cfg
        step = int(outputs["step"])
        image, gt, entry = shared["image"], shared["gt"], shared["entry"]
        H, W = image.shape[1:]
        model = shared["dino"]
        fine_grid, coarse_grid = shared["grids"]
        coarse = bool(cfg["bootstrap"]) and step < int(cfg["bootstrap_iter"])
        if coarse:
            cam = entry["coarse_camera"]
            with torch.no_grad():
                c_img, _, _ = rasterize(params, cam, outputs["sh_degree"],
                                        outputs.get("near_plane", 0.2),
                                        outputs.get("far_plane", 1e10))
            res_gt, res_img = entry["coarse_image"], c_img.permute(2, 0, 1)
            gt_feat = entry["feat_coarse"]
            grid = coarse_grid
        else:
            res_gt, res_img = gt, image.detach()
            gt_feat = entry["feat_fine"]
            grid = fine_grid
        render_feat = dino.extract(model, image.detach(), grid)
        lo, up, Shared.hist = residual_masks(res_gt, res_img, Shared.hist,
                                             float(cfg["lower_bound"]), float(cfg["upper_bound"]),
                                             float(cfg["hist_decay"]))
        lo, up = upsample(lo, H, W), upsample(up, H, W)
        target = cosine_target(gt_feat, render_feat, H, W)
        return {"lower": lo, "upper": up, "cos_target": target, "coarse": coarse,
                "residual_size": tuple(res_img.shape[1:]), "grid": grid}

    def __call__(self, params, outputs, target):
        cfg = self.cfg
        shared = outputs.get("robustsplat")
        step = int(outputs["step"])
        if shared is None or shared.get("step") != step:
            raise RuntimeError("rs-mask-mlp must follow rs-masked-photometric in the components")
        if "mask" not in shared or not mlp_active(step, cfg):
            self.last = {"active": False}
            return _zero_attached(outputs["rgb"])
        mask = shared["mask"]
        sup = self.supervision(params, outputs, shared)
        l_cos = (mask - sup["cos_target"]).abs().mean()                       # Eq. 5
        l_res = residual_hinge(mask.flatten(), sup["lower"].flatten(), sup["upper"].flatten())
        l_reg = ((1.0 - mask) * reg_weight(step, cfg["beta_reg"])).mean()     # Eq. 7
        Shared.pending_step = True
        self.last = {"active": True, "coarse": sup["coarse"], "grid": sup["grid"],
                     "residual_size": sup["residual_size"], "l_cos": float(l_cos),
                     "l_res": float(l_res), "l_reg": float(l_reg)}
        return (float(cfg["lambda_cos"]) * l_cos + float(cfg["lambda_residual"]) * l_res
                + float(cfg["lambda_reg"]) * l_reg)                            # Eq. 8


def _zero_attached(rgb):
    """Graph-attached zero (no gradient anywhere) for steps where the official code does not
    train the MLP; ``pending_step`` stays False so Adam is not stepped."""
    return rgb.sum() * 0.0


def create_mask_mlp(spec):
    return MaskMLPLoss(spec)
