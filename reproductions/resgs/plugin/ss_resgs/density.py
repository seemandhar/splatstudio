"""ResGS density controller (AbsGS host, released 'Ours (AbsGS)' variant).

Official sources (github.com/yanzhelyu/ResGS): `train.py:training` (schedule, order of
operations), `scene/gaussian_model.py:adjust_gaussian` (Eq. 8 selection, pruning),
`densify_residual_split` (Eq. 4-5), `reduce_opacity` / `prune_opacity` (AbsGS periodic opacity
reduction), `arguments/__init__.py` (defaults).
"""

import torch
import torch.nn.functional as F
from gsplat.strategy import ops
from gsplat.strategy.ops import _update_param_with_optimizer
from gsplat.utils import normalized_quat_to_rotmat

from .link import LINK
from .schedule import ResGSSchedule

DEFAULTS = {
    "grad_threshold": 0.00067,  # tau, densify_grad_threshold (abs gradient)
    "stage_hr_factor": 2.0,  # alpha = stage_hr_factor ** (1 / stage_split) = 2^(1/3)
    "stage_split": 3,  # K substages per stage
    "residual_split_scale_div": 1.6,  # lambda_s (Eq. 4)
    "opacity_reduce_weight": 0.3,  # beta (Eq. 5)
    "min_opacity": 0.005,
    "start_stat": 500,
    "update_from": 700,
    "update_interval": 100,
    "update_until": 12000,
    "change_iter": [2500, 6000],
    "resolution_scales": [2, 1, 0],  # pyrDown count per stage, L = 3
    "warm_up_iter": 500,
    "use_opacity_reduce": True,
    "opacity_reduce_factor": 0.8,  # reduce_opacity: get_opacity * 0.8
    "opacity_reduce_interval": 500,
    "prune_until": 25000,
    # Learning rates of the official OptimizationParams (3DGS values in brackets).
    "position_lr_init": 0.000128,  # [1.6e-4]; x scene extent, decays x0.01 by the runtime
    "feature_lr": 0.002,  # sh0 [0.0025]; shN = feature_lr / 20
    "opacity_lr": 0.04,  # [0.05]
    "scaling_lr": 0.006,  # [0.005]
    "rotation_lr": 0.001,
    # Ablation switches (paper Table 4); True = full method.
    "image_pyramid": True,
    "varying_threshold": True,
}


def thresholds(levels, cur_stage, tau, alpha, varying=True):
    """Eq. 8 (code form): tau * alpha^(l - k) for l < k, tau otherwise."""
    thr = torch.full_like(levels, float(tau), dtype=torch.float32)
    if varying:
        low = levels < cur_stage
        thr[low] = thr[low] * torch.pow(torch.full_like(thr[low], alpha), levels[low] - cur_stage)
    return thr


@torch.no_grad()
def residual_split(params, optimizers, state, mask, scale_div, opacity_weight, levels_key="levels"):
    """Eq. 4-5: keep G_i, o_i <- beta o_i; append G_j with S_i / lambda_s, mu_j ~ N(mu_i, Sigma_i).

    Matches densify_residual_split: child opacity/rotation/SH copy the parent *before* the
    reduction; the parent's opacity Adam moments are zeroed; children start with zero moments.
    Running-state tensors are extended with the parent's values (stats are reset by the
    caller); `state[levels_key]` gets parent level + 1 for children.
    """
    sel = torch.where(mask)[0]
    n_new = len(sel)
    if n_new == 0:
        return 0
    device = mask.device
    scales = torch.exp(params["scales"][sel])
    rotmats = normalized_quat_to_rotmat(F.normalize(params["quats"][sel], dim=-1))
    eps = torch.randn(n_new, 3, device=device)
    new_means = torch.einsum("nij,nj->ni", rotmats, scales * eps) + params["means"][sel]
    new_log_scales = torch.log(scales / scale_div)
    child_opacity = params["opacities"][sel].clone()
    parent_opacity = torch.logit(torch.sigmoid(params["opacities"][sel]) * opacity_weight)

    def param_fn(name, p):
        base = p
        if name == "means":
            extra = new_means
        elif name == "scales":
            extra = new_log_scales
        elif name == "opacities":
            # Under no_grad a clone has requires_grad=False: keep the flag of the Parameter `p`
            # (an earlier version took it from the clone and froze all opacities after the first
            # residual split).
            base = p.detach().clone()
            extra = child_opacity
            base[sel] = parent_opacity
        else:
            extra = p[sel]
        return torch.nn.Parameter(torch.cat([base, extra]), requires_grad=p.requires_grad)

    def zeros_fn(key, v):
        return torch.cat([v, torch.zeros((n_new, *v.shape[1:]), device=device, dtype=v.dtype)])

    def opacity_moments_fn(key, v):
        v = v.clone()
        v[sel] = 0  # replace_tensor_to_optimizer_with_mask
        return zeros_fn(key, v)

    names = list(params.keys())
    _update_param_with_optimizer(param_fn, opacity_moments_fn, params, optimizers, ["opacities"])
    _update_param_with_optimizer(param_fn, zeros_fn, params, optimizers,
                                 [n for n in names if n != "opacities"])
    for k, v in state.items():
        if isinstance(v, torch.Tensor):
            extra = v[sel] + 1 if k == levels_key else v[sel]
            state[k] = torch.cat((v, extra))
    return n_new


@torch.no_grad()
def reduce_opacity(params, optimizers, factor):
    """AbsGS / ResGS reduce_opacity: o <- logit(sigmoid(o) * factor), opacity moments zeroed."""

    def param_fn(name, p):
        return torch.nn.Parameter(torch.logit(torch.sigmoid(p) * factor), requires_grad=p.requires_grad)

    def optimizer_fn(key, v):
        return torch.zeros_like(v)

    _update_param_with_optimizer(param_fn, optimizer_fn, params, optimizers, ["opacities"])


class ResGSDensity:
    requires = {"gradient.screen.absolute"}

    def __init__(self, spec):
        unknown = set(spec.config) - set(DEFAULTS)
        if unknown:
            raise ValueError(f"Unknown resgs-density options: {sorted(unknown)}")
        self.c = {**DEFAULTS, **spec.config}
        c = self.c
        for key in ("grad_threshold", "residual_split_scale_div", "stage_hr_factor"):
            if not c[key] > 0:
                raise ValueError(f"{key} must be positive")
        if not 0 < c["opacity_reduce_weight"] <= 1:
            raise ValueError("opacity_reduce_weight must be in (0, 1]")
        self.alpha = float(c["stage_hr_factor"]) ** (1.0 / int(c["stage_split"]))
        self.schedule = ResGSSchedule(c)
        self.active = False
        self.stats = None
        self.state = None
        self.history = []
        self.max_gaussian_level = 0
        LINK.attach(self)

    @property
    def pyramid_level(self):
        return self.schedule.pyramid_level if self.c["image_pyramid"] else 0

    # ---- set-up -------------------------------------------------------------------------
    def initialize(self, params, optimizers, scene_scale=1.0):
        for name in ("means", "scales", "quats", "opacities", "sh0", "shN"):
            if name not in params or name not in optimizers:
                raise ValueError(f"resgs-density requires the {name} field")
        c = self.c
        lrs = {
            "means": c["position_lr_init"] * float(scene_scale),
            "sh0": c["feature_lr"],
            "shN": c["feature_lr"] / 20.0,
            "opacities": c["opacity_lr"],
            "scales": c["scaling_lr"],
            "quats": c["rotation_lr"],
        }
        for name, lr in lrs.items():
            for group in optimizers[name].param_groups:
                group["lr"] = lr
        n, device = len(params["means"]), params["means"].device
        self.stats = {"grad2d": torch.zeros(n, device=device),
                      "count": torch.zeros(n, device=device),
                      "levels": torch.zeros(n, device=device)}
        self.active = True
        self._sync_state()

    # ---- runtime callbacks --------------------------------------------------------------
    def before_backward(self, params, optimizers, step, outputs):
        # Guard against renderer/schedule desync for renders this controller scheduled.
        if outputs.get("pyramid_owner") is self and outputs["pyramid_level"] != self.pyramid_level:
            raise RuntimeError("Rendered pyramid level disagrees with the ResGS schedule")
        outputs["info"]["means2d"].retain_grad()

    @torch.no_grad()
    def after_backward(self, params, optimizers, step, outputs):
        c, s, t = self.c, self.schedule, step
        if s.stats_active(t):
            self._accumulate(outputs["info"])
            if s.densify_step(t):
                self.densify(params, optimizers, t)
        s.change_stage(t)
        reduce, prune_first = s.opacity_reduce_step(t)
        if reduce:
            if prune_first:
                self._prune(params, optimizers)
            reduce_opacity(params, optimizers, c["opacity_reduce_factor"])
        s.end_iteration(t)
        if self.history and self.history[-1]["step"] == t or reduce:
            frozen = [k for k, p in params.items() if not p.requires_grad]
            if frozen:
                raise RuntimeError(f"resgs density op left non-trainable fields: {frozen}")
        self._sync_state()

    # ---- densification ------------------------------------------------------------------
    def _accumulate(self, info):
        grads = getattr(info["means2d"], "absgrad", None)
        if grads is None:
            raise RuntimeError("means2d has no absgrad; the renderer must use absgrad=True")
        grads = grads.clone()
        # 3DGS NDC gradient at the render size (gsplat DefaultStrategy._update_state).
        grads[..., 0] *= info["width"] / 2.0 * info["n_cameras"]
        grads[..., 1] *= info["height"] / 2.0 * info["n_cameras"]
        sel = (info["radii"] > 0.0).all(dim=-1)
        ids = torch.where(sel)[1]
        self.stats["grad2d"].index_add_(0, ids, grads[sel].norm(dim=-1))
        self.stats["count"].index_add_(0, ids, torch.ones_like(ids, dtype=torch.float32))

    def select(self):
        """adjust_gaussian: add_child_mask = mean abs grad >= tau_{k,i} (Eq. 8)."""
        grads = self.stats["grad2d"] / self.stats["count"]
        grads[grads.isnan()] = 0.0
        thr = thresholds(self.stats["levels"], self.schedule.cur_level, self.c["grad_threshold"],
                         self.alpha, self.c["varying_threshold"])
        return grads >= thr

    def densify(self, params, optimizers, step):
        c = self.c
        mask = self.select()
        n_split = residual_split(params, optimizers, self.stats, mask,
                                 c["residual_split_scale_div"], c["opacity_reduce_weight"])
        n_pruned = self._prune(params, optimizers)
        self.stats["grad2d"].zero_()
        self.stats["count"].zero_()
        self.max_gaussian_level = int(self.stats["levels"].max().item())
        self.history.append({"step": step, "level": self.schedule.cur_level, "split": n_split,
                             "pruned": n_pruned, "gaussians": len(params["means"]),
                             "pyramid_level": self.pyramid_level})

    def _prune(self, params, optimizers):
        prune = torch.sigmoid(params["opacities"].flatten()) < self.c["min_opacity"]
        n = int(prune.sum())
        if n:
            ops.remove(params, optimizers, self.stats, prune)
        return n

    def _sync_state(self):
        # Checkpointed with weights_only=True: plain numbers only. Levels are discarded after
        # training (paper Sec. 3.2.3).
        self.state = {
            **self.schedule.to_dict(),
            "pyramid_level": int(self.pyramid_level),
            "alpha": float(self.alpha),
            "max_gaussian_level": int(self.max_gaussian_level),
        }


def create_density(spec):
    return ResGSDensity(spec)
