"""Laplacian-GS density controller: frequency-staged training with archiving (Alg. 1).

Per stage (official train_lap.py, backbone `taming` = DashGaussian ADC with both schedulers
off, scene/gaussian_model.py:prune_and_densify_dash): 3DGS densification restricted to the
active field with stage-local schedules, position-LR restart per stage, and at the end of a
stage the field is archived and the next residual field is created by xyz-only inheritance
(extract_sub_gaussians_only_xyz + create_from_xyz) with a fresh optimizer.
"""

import math

import numpy as np
import torch
from gsplat.strategy import ops
from scipy.spatial import cKDTree

from .link import LINK
from .renderer import LEVEL_FIELD, check_hierarchy, stage_level

DEFAULTS = {
    "levels": 2,
    "iter_plan": [10000, 20000],  # --iter_plan 10000_20000
    "inherit_ratio": 1.0,  # --inherit_ratio 1.0, --inherit_state only_xyz
    "grow_grad2d": 0.0002,  # densify_grad_threshold
    "grow_scale3d": 0.01,  # --percent_dense 0.01 (Taming script)
    "prune_opa": 0.005,
    "refine_start": 500,  # densify_from_iter (strict >)
    "refine_every": 100,
    "densify_until": None,  # None: int(15000 * T_s / 30000) per stage (train_lap.py:446)
    "reference_densify_until": 15000,
    "reference_iterations": 30000,
    "reset_every": 3000,
    "reset_value": 0.01,
    "lr_final_factor": 0.01,  # position_lr_final / position_lr_init
    # arguments/__init__.py (training_setup): f_dc = lowfeature_lr, f_rest = highfeature_lr/20
    "lr": {"sh0": 0.0025, "shN": 0.005 / 20.0, "opacities": 0.025, "scales": 0.005,
           "quats": 0.001},
    "init_opacity": 0.1,
}


def equidistant_indices(n, k, offset=0):
    """train_lap.py:_equidistant_indices."""
    if k <= 0:
        return torch.empty(0, dtype=torch.long)
    stride = max(1, n // k)
    idx = (offset % stride) + torch.arange(k) * stride
    return torch.unique(idx.clamp(max=n - 1), sorted=True)


def knn_log_scales(points):
    """distCUDA2: mean squared distance to the 3 nearest other points; log sqrt, clamp 1e-7."""
    xyz = points.detach().cpu().double().numpy()
    k = min(4, len(xyz))
    dist, _ = cKDTree(xyz).query(xyz, k=k)
    dist = np.atleast_2d(dist)
    d2 = (dist[:, 1:] ** 2).mean(axis=1) if k > 1 else np.full(len(xyz), 1e-7)
    d2 = torch.from_numpy(np.maximum(d2, 1e-7)).float().to(points.device)
    return d2.sqrt().log()[:, None].expand(-1, 3).contiguous()


class LapDensity:
    requires = {"gradient.screen.signed"}

    def __init__(self, spec):
        unknown = set(spec.config) - set(DEFAULTS)
        if unknown:
            raise ValueError(f"Unknown lap-density options: {sorted(unknown)}")
        c = {**DEFAULTS, **spec.config, "lr": {**DEFAULTS["lr"], **spec.config.get("lr", {})}}
        self.levels, self.plan = check_hierarchy(c["levels"], c["iter_plan"])
        if not 0.0 < float(c["inherit_ratio"]) <= 1.0:
            raise ValueError("inherit_ratio must be in (0, 1]")
        until = c["densify_until"]
        if until is None:
            until = [int(c["reference_densify_until"] * (t / c["reference_iterations"]))
                     for t in self.plan]
        if len(until) != self.levels:
            raise ValueError("densify_until needs one value per stage")
        self.until = [int(u) for u in until]
        self.c = c
        self.stage = 0
        self.level = stage_level(0, self.levels)
        self.stage_start = 0  # global step after which the current stage began
        self.local_step = 0  # stage-local iterations completed
        self.stats = None
        self.history = []
        self.state = None
        LINK.attach(self)
        self._sync_state()

    # ---- set-up ---------------------------------------------------------------------------
    def initialize(self, params, optimizers, scene_scale=1.0):
        if LEVEL_FIELD not in params:
            raise ValueError(f"lap-density requires the {LEVEL_FIELD} field")
        if not bool((params[LEVEL_FIELD] == float(self.level)).all()):
            raise ValueError("All initial Gaussians must belong to the base level")
        self.scene_scale = float(scene_scale)
        for name, lr in self.c["lr"].items():
            optimizers[name].param_groups[0]["lr"] = float(lr)
        # The runtime has already multiplied the means LR by the scene scale.
        self.lr_init = optimizers["means"].param_groups[0]["lr"]
        self._reset_stats(params)
        self._sync_state()

    # ---- schedules ------------------------------------------------------------------------
    def position_lr(self, local_iteration):
        """get_expon_lr_func without delay, max_steps = stage iterations (restart per stage)."""
        t = min(max(local_iteration / self.plan[self.stage], 0.0), 1.0)
        return self.lr_init * math.exp(math.log(self.c["lr_final_factor"]) * t)

    def active_mask(self, params):
        return params[LEVEL_FIELD].detach() == float(self.level)

    # ---- runtime callbacks ----------------------------------------------------------------
    def before_backward(self, params, optimizers, step, outputs):
        if outputs.get("lap_level") != self.level:
            raise RuntimeError("Rendered level disagrees with the active Laplacian stage")
        if step - self.stage_start != self.local_step + 1:
            raise RuntimeError("laplacian-gs stage counter out of sync with the runtime step")
        outputs["info"]["means2d"].retain_grad()

    @torch.no_grad()
    def after_backward(self, params, optimizers, step, outputs):
        c = self.c
        self.local_step = i = step - self.stage_start
        until = self.until[self.stage]
        if i < until:
            self._accumulate(outputs)
            if i > c["refine_start"] and i % c["refine_every"] == 0:
                self.prune_and_densify(params, optimizers)
            if i % c["reset_every"] == 0:
                self.reset_opacity(params, optimizers)
        if i == self.plan[self.stage] and self.stage < self.levels - 1:
            self.archive_and_inherit(params, optimizers, step)
        optimizers["means"].param_groups[0]["lr"] = self.position_lr(self.local_step + 1)
        self._sync_state()

    # ---- densification (active field only) ------------------------------------------------
    def _reset_stats(self, params):
        n, device = len(params["means"]), params["means"].device
        self.stats = {"grad2d": torch.zeros(n, device=device),
                      "count": torch.zeros(n, device=device)}

    def _accumulate(self, outputs):
        info = outputs["info"]
        grads = info["means2d"].grad
        if grads is None:
            raise RuntimeError("means2d has no gradient; before_backward must retain it")
        grads = grads.clone()
        grads[..., 0] *= info["width"] / 2.0 * info["n_cameras"]
        grads[..., 1] *= info["height"] / 2.0 * info["n_cameras"]
        sel = (info["radii"] > 0.0).all(dim=-1)  # [1, N_active]
        local = torch.where(sel)[1]
        ids = outputs["active_ids"][local]
        self.stats["grad2d"].index_add_(0, ids, grads[sel].norm(dim=-1))
        self.stats["count"].index_add_(0, ids, torch.ones_like(ids, dtype=torch.float32))

    def prune_and_densify(self, params, optimizers):
        """prune_and_densify_dash with densify_rate 1.0, applied to the active field."""
        c = self.c
        active = self.active_mask(params)
        cur_n = int(active.sum())
        prune = active & (torch.sigmoid(params["opacities"].flatten()) < c["prune_opa"])
        if prune.any():
            ops.remove(params, optimizers, self.stats, prune)
        active = self.active_mask(params)
        n_left = int(active.sum())
        n_densify = min(int(cur_n * 2.0 - n_left), n_left)
        grads = self.stats["grad2d"] / self.stats["count"]
        grads[grads.isnan()] = 0.0
        grads = torch.where(active, grads, torch.full_like(grads, -1.0))
        qualified = active & (grads >= c["grow_grad2d"])
        top = torch.zeros_like(qualified)
        if n_densify > 0:
            top[torch.topk(grads, n_densify).indices] = True
        small = torch.exp(params["scales"]).max(dim=-1).values <= c["grow_scale3d"] * self.scene_scale
        clone = qualified & top & small
        split = qualified & top & ~small
        n_clone = int(clone.sum())
        if n_clone:
            ops.duplicate(params, optimizers, self.stats, clone)
        n_split = int(split.sum())
        if n_split:
            split = torch.cat([split, torch.zeros(n_clone, dtype=torch.bool, device=split.device)])
            ops.split(params, optimizers, self.stats, split)
        self._reset_stats(params)  # densification_postfix zeroes the accumulators
        self.last_densify = {"step_local": self.local_step, "clone": n_clone, "split": n_split,
                             "pruned": int(prune.sum()), "active": int(self.active_mask(params).sum())}
        self.history.append(self.last_densify)

    def reset_opacity(self, params, optimizers):
        """reset_opacity: sigma <- min(sigma, 0.01) for the active field, moments zeroed."""
        active = self.active_mask(params)
        cap = torch.logit(torch.tensor(self.c["reset_value"])).item()

        def param_fn(name, p):
            new = torch.where(active, p.clamp(max=cap), p)
            return torch.nn.Parameter(new, requires_grad=p.requires_grad)

        def optimizer_fn(key, v):
            return torch.where(active, torch.zeros_like(v), v)

        ops._update_param_with_optimizer(param_fn, optimizer_fn, params, optimizers,
                                         names=["opacities"])

    # ---- stage transition -----------------------------------------------------------------
    def archive_and_inherit(self, params, optimizers, step):
        """Archive the trained field; the next field inherits its xyz only (ratio rho)."""
        c = self.c
        source = torch.where(self.active_mask(params))[0]
        n = len(source)
        k = max(1, int(n * float(c["inherit_ratio"])))
        keep = source[equidistant_indices(n, k).to(source.device)]
        xyz = params["means"].detach()[keep].clone()
        new_level = self.level - 1
        m = len(xyz)
        device = xyz.device
        quats = torch.zeros(m, 4, device=device)
        quats[:, 0] = 1.0
        fresh = {
            "means": xyz,
            "scales": knn_log_scales(xyz),
            "quats": quats,
            "opacities": torch.full((m,), math.log(c["init_opacity"] / (1 - c["init_opacity"])),
                                    device=device),
            "sh0": torch.zeros(m, 1, 3, device=device),
            "shN": torch.zeros(m, *params["shN"].shape[1:], device=device),
            LEVEL_FIELD: torch.full((m,), float(new_level), device=device),
        }
        for name in params:
            if name not in fresh:
                fresh[name] = torch.zeros(m, *params[name].shape[1:], device=device)

        def param_fn(name, p):
            return torch.nn.Parameter(torch.cat([p.detach(), fresh[name].to(p.dtype)]),
                                      requires_grad=p.requires_grad)

        def optimizer_fn(key, v):
            # Fresh optimizer (training_setup): all moments start at zero; archived rows
            # thereafter receive zero gradients and hence exactly zero Adam updates.
            return torch.zeros((v.shape[0] + m, *v.shape[1:]), device=v.device, dtype=v.dtype)

        ops._update_param_with_optimizer(param_fn, optimizer_fn, params, optimizers)
        for opt in optimizers.values():
            for st in opt.state.values():
                if "step" in st:
                    st["step"] = torch.zeros_like(st["step"])
        self.history.append({"archive_step": step, "archived_level": self.level,
                             "archived": n, "inherited": m})
        self.stage += 1
        self.level = new_level
        self.stage_start = step
        self.local_step = 0
        self._reset_stats(params)

    def _sync_state(self):
        # Checkpointed with weights_only=True: plain numbers only.
        self.state = {"stage": int(self.stage), "level": int(self.level),
                      "stage_start": int(self.stage_start), "local_step": int(self.local_step),
                      "iter_plan": [int(t) for t in self.plan]}


def create_density(spec):
    return LapDensity(spec)
