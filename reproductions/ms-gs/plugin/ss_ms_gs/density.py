"""MS-GS density controller: 3DGS ADC on level-0 Gaussians, S range bookkeeping, aggregation
insertion (C2), multi-scale sampler (C4) and the 3DGS position-LR schedule.

Official reference: MS-GS @ 50e5040 train.py:119-418 and scene/gaussian_model.py. Per training
step (runtime order: render -> backward -> optimizer step -> after_backward):
  1. it >= 250 and > 250 its after the last opacity reset: Eq. 6 range update for visible
     Gaussians of the rendered level (train.py:244-245).
  2. it < 15000: gradient statistics from 1x renders only (slot reso_lvl=0, train.py:250);
     densify when the 1x-iteration count is a multiple of 100 and it > 500 (train.py:253-257):
     clone/split with |grad| >= 2e-4 (3DGS percent_dense 0.01 x extent), level-0 only; prune
     opacity < 0.005 or (it > 3000) max scale > 0.1 x extent, level-0 only. The 20 px screen-size
     prune is inert in the official code (max_radii2D is zeroed by densification_postfix right
     before it is read, as in 3DGS) and is therefore not applied.
  3. it % 3000 == 0 (it < 15000): opacity reset of level-0 Gaussians to min(sigma, 0.01).
  4. it in insert_at: aggregate-insert level r (train.py:269-387).
  5. position LR for the next step: 3DGS expon schedule over position_lr_max_steps, then hold.
"""

import torch
from gsplat import fully_fused_projection
from gsplat.strategy.ops import _update_param_with_optimizer, duplicate, remove, split

from .aggregate import RAW_FIELDS, voxel_aggregate
from .coverage import pixel_coverage, update_ranges
from .link import LINK
from .renderer import DEFAULTS as RENDER_DEFAULTS
from .renderer import coverage_and_mask, scaled_camera
from .schedule import MultiScaleSampler, insert_table, position_lr

FIELDS = ("ms_level", "ms_max_px", "ms_min_px")
DEFAULTS = {
    "max_scale": 7,  # train_all.py: Mip-NeRF 360 trains 1x..128x
    "p_full_res": 0.75,
    "grow_grad2d": 0.0002,
    "percent_dense": 0.01,
    "refine_start_iter": 500,
    "refine_stop_iter": 15000,
    "refine_every": 100,
    "reset_every": 3000,
    "reset_value": 0.01,
    "prune_opa": 0.005,
    "prune_scale3d": 0.1,
    "insert_threshold": 1.0,  # train.py:288 pixel_size_threshold
    "voxel_base": 0.02,
    "target_px": 2.0,
    "scale_clip": [0.25, 2.0],
    "decay_max": 0.95,
    "decay_min": 1.05,
    "px_update_from": 250,
    "px_skip_after_reset": 250,
    "position_lr_max_steps": 30000,
    "seed": 0,
}


def _scale_px_children(params, n_children):
    """Split children: S range / (0.8 * N) with N = 2 (gaussian_model.py:561-562)."""
    if n_children:
        for name in ("ms_max_px", "ms_min_px"):
            params[name].data[-n_children:] /= 1.6


class Density:
    requires = {"gradient.screen.signed"}

    def __init__(self, spec):
        cfg = dict(spec.config)
        render_cfg = cfg.pop("render", {})
        unknown = set(cfg) - set(DEFAULTS)
        if unknown:
            raise ValueError(f"Unknown ms-gs density options: {sorted(unknown)}")
        self.cfg = {**DEFAULTS, **cfg}
        self.render_cfg = {**RENDER_DEFAULTS, **render_cfg}
        self.train_lists, self.insert_levels, self.insert_at = insert_table(self.cfg["max_scale"])
        self.sampler = MultiScaleSampler(self.cfg["max_scale"], self.cfg["p_full_res"],
                                         self.cfg["seed"])
        self.n_levels = self.cfg["max_scale"] + 1
        self.cameras = []
        self.step = 0
        self.state = None
        LINK.attach(self)

    # ---- link API -------------------------------------------------------------------------
    def next_training_level(self):
        """Called by the renderer for every training render; returns the resolution index."""
        return self.sampler.next_level(self.step + 1)

    def set_training_cameras(self, cameras):
        self.cameras = list(cameras)
        self.sampler.n_train = len(self.cameras)

    # ---- density API ----------------------------------------------------------------------
    def initialize(self, params, optimizers, scene_scale=1.0):
        for f in FIELDS:
            if f not in params:
                raise ValueError(f"ms-gs density needs field {f}")
        n = len(params["means"])
        dev = params["means"].device
        self.state = {
            "scene_scale": float(scene_scale),
            "stats": {"grad2d": torch.zeros(n, device=dev), "count": torch.zeros(n, device=dev)},
            "initial_means_lr": optimizers["means"].param_groups[0]["lr"],
            "last_reset": None,
            "inserted": {},
            "n_full_res": 0,
        }
        self.step = 0

    def before_backward(self, params, optimizers, step, outputs):
        self.step = step
        outputs["info"]["means2d"].retain_grad()

    @torch.no_grad()
    def after_backward(self, params, optimizers, step, outputs):
        cfg, st = self.cfg, self.state
        info = outputs["info"]
        level = int(info.get("ms_level", 0))
        if level == 0:
            st["n_full_res"] += 1
        n = len(params["means"])
        visible = self._visible_full(info, n)

        last = st["last_reset"]
        if step >= cfg["px_update_from"] and (last is None or step - last > cfg["px_skip_after_reset"]):
            update_ranges(params["ms_max_px"].data, params["ms_min_px"].data,
                          params["ms_level"].data, visible, info["ms_px"], level, self.n_levels,
                          cfg["decay_max"], cfg["decay_min"])

        if step < cfg["refine_stop_iter"]:
            if level == 0:
                self._accumulate(info, n)
                if step > cfg["refine_start_iter"] and st["n_full_res"] % cfg["refine_every"] == 0:
                    self.densify_and_prune(params, optimizers, step)
            if step % cfg["reset_every"] == 0:
                self.reset_opacity(params, optimizers)
                st["last_reset"] = step

        if step in self.insert_at:
            r = self.insert_levels[self.insert_at.index(step)]
            st["inserted"][r] = self.insert_large(params, optimizers, r)

        optimizers["means"].param_groups[0]["lr"] = position_lr(
            st["initial_means_lr"], step + 1, cfg["position_lr_max_steps"])

    # ---- helpers --------------------------------------------------------------------------
    def _visible_full(self, info, n):
        return (info["radii"][0] > 0).all(dim=-1)

    def _accumulate(self, info, n):
        stats = self.state["stats"]
        grad = info["means2d"].grad
        if grad is None:
            return
        g = grad[0].clone()
        g[..., 0] *= info["width"] / 2.0
        g[..., 1] *= info["height"] / 2.0
        vis = self._visible_full(info, n)
        stats["grad2d"][vis] += g[vis].norm(dim=-1)
        stats["count"][vis] += 1

    def _reset_stats(self, n, device):
        self.state["stats"] = {"grad2d": torch.zeros(n, device=device),
                               "count": torch.zeros(n, device=device)}

    def densify_and_prune(self, params, optimizers, step):
        cfg, st = self.cfg, self.state
        stats = st["stats"]
        extent = st["scene_scale"]
        grads = stats["grad2d"] / stats["count"]
        grads = torch.nan_to_num(grads, nan=0.0)
        grads[params["ms_level"] != 0] = 0.0
        high = grads >= cfg["grow_grad2d"]
        max_scale = params["scales"].exp().max(dim=-1).values
        is_clone = high & (max_scale <= cfg["percent_dense"] * extent)
        is_split = high & (max_scale > cfg["percent_dense"] * extent)
        n_clone, n_split = int(is_clone.sum()), int(is_split.sum())
        if n_clone:
            duplicate(params, optimizers, stats, is_clone)
        is_split = torch.cat([is_split, torch.zeros(n_clone, dtype=torch.bool,
                                                     device=is_split.device)])
        if n_split:
            split(params, optimizers, stats, is_split)
            _scale_px_children(params, 2 * n_split)
        prune = params["opacities"].sigmoid() < cfg["prune_opa"]
        if step > cfg["reset_every"]:
            prune |= params["scales"].exp().max(dim=-1).values > cfg["prune_scale3d"] * extent
        prune &= params["ms_level"] == 0
        n_prune = int(prune.sum())
        if n_prune:
            remove(params, optimizers, stats, prune)
        self._reset_stats(len(params["means"]), params["means"].device)
        return {"clone": n_clone, "split": n_split, "prune": n_prune}

    def reset_opacity(self, params, optimizers):
        """gaussian_model.py:346 reset only level 0; Adam moments of the opacity tensor zeroed."""
        cap = torch.logit(torch.tensor(self.cfg["reset_value"])).item()
        level0 = params["ms_level"] == 0

        def param_fn(name, p):
            return torch.nn.Parameter(torch.where(level0, p.clamp(max=cap), p),
                                      requires_grad=p.requires_grad)

        def optimizer_fn(key, v):
            return torch.zeros_like(v)

        _update_param_with_optimizer(param_fn, optimizer_fn, params, optimizers,
                                     names=["opacities"])

    def coverage_and_visibility(self, params, camera):
        """Official render() side products: pixel_sizes and visibility_filter (radii > 0)."""
        px, keep = coverage_and_mask(params, camera, self.render_cfg)
        opac = params["opacities"].sigmoid() * keep.float()
        radii = fully_fused_projection(
            params["means"], None, params["quats"], params["scales"].exp(),
            camera["viewmat"][None], camera["K"][None], int(camera["width"]),
            int(camera["height"]), packed=False, opacities=opac,
        )[0]
        return px, (radii[0] > 0).all(dim=-1)

    def insert_large(self, params, optimizers, level):
        """Aggregate small level-0 Gaussians into level `level` (train.py:269-387)."""
        cfg, st = self.cfg, self.state
        if not self.cameras:
            return {"selected": 0, "inserted": 0}
        thr = cfg["insert_threshold"]
        min_px = torch.full_like(params["ms_min_px"], thr)
        for cam in self.cameras:
            _, vis1 = self.coverage_and_visibility(params, cam)
            px_r = pixel_coverage(params, scaled_camera(cam, level),
                                  self.render_cfg["opacity_level"])
            ok = (px_r > 0) & vis1
            min_px = torch.where(ok, torch.minimum(px_r, min_px), min_px)
        chosen = (min_px < thr) & (params["ms_level"] == 0)
        raw = {k: params[k][chosen] for k in RAW_FIELDS}
        new = voxel_aggregate(raw, min_px[chosen], st["scene_scale"], level,
                              voxel_base=cfg["voxel_base"], target_px=cfg["target_px"],
                              scale_clip=tuple(cfg["scale_clip"]))
        n_new = 0 if new is None else len(new["means"])
        if n_new:
            self.append(params, optimizers, new)
            for cam in self.cameras:
                px, vis = self.coverage_and_visibility(params, scaled_camera(cam, level))
                update_ranges(params["ms_max_px"].data, params["ms_min_px"].data,
                              params["ms_level"].data, vis, px, level, self.n_levels,
                              cfg["decay_max"], cfg["decay_min"])
        return {"selected": int(chosen.sum()), "inserted": n_new}

    def append(self, params, optimizers, new):
        n_new = len(new["means"])
        dev = params["means"].device

        def param_fn(name, p):
            return torch.nn.Parameter(torch.cat([p, new[name].to(p.dtype)]),
                                      requires_grad=p.requires_grad)

        def optimizer_fn(key, v):
            return torch.cat([v, torch.zeros((n_new, *v.shape[1:]), device=dev, dtype=v.dtype)])

        _update_param_with_optimizer(param_fn, optimizer_fn, params, optimizers)
        stats = self.state["stats"]
        for k in stats:
            stats[k] = torch.cat([stats[k], torch.zeros(n_new, device=dev)])


def create_density(spec):
    return Density(spec)
