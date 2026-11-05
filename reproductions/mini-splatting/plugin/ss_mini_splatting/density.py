"""Mini-Splatting density control (densification + simplification), after the official
`ms/train.py` / `ms_d/train.py` and `scene/gaussian_model.py` of
https://github.com/fatPeter/mini-splatting.

Per training step s (the runtime calls `after_backward` after the optimizer step):
  s < densify_until: 3DGS gradient statistics + blur mask (Eq. 1) from the step's own render;
      densify (clone + split ∪ blur, prune) every 100 steps; depth reinit every 5000 steps.
  s == restart_step: Mini-Splatting: intersection preserving (Eq. 2) + importance-weighted
      sampling, then reinit from the surviving centres; Mini-Splatting-D: reinit from centres.
  s == simp_iteration2 (Mini-Splatting): intersection + 1% CDF importance prune.
  every step: means LR and active SH degree of the official schedule for step s + 1.
"""

import torch
from gsplat.strategy.ops import duplicate, remove, split
from torch import nn

from . import core, kernels
from .renderer import TARGET_CAPABILITY

DEFAULTS = {
    # 3DGS densification (official OptimizationParams)
    "densify_from": 500,
    "densify_until": 15000,
    "densify_every": 100,
    "grad_threshold": 0.0002,
    "percent_dense": 0.01,
    "min_opacity": 0.005,
    "big_world_from": 3000,  # size_threshold is set when iteration > opacity_reset_interval
    "big_world": 0.1,
    # C1 blur split (Eq. 1)
    "theta_blur": 2e-4,
    # C2 depth reinitialization
    "reinit_every": 5000,
    "num_depth": 3_500_000,
    "depth_with_replacement": False,  # Mini-Splatting-D: True (choice with replacement + unique)
    "reinit_opacity": 0.1,
    "num_max": 4_500_000,  # Mini-Splatting only; None disables the cap (Mini-Splatting-D)
    # C3/C4 simplification
    "simplify": True,
    "restart_step": 15000,  # simp_iteration1 (MS) / densify_until_iter (MS-D)
    "simp_iteration2": 20000,
    "sampling_factor": 0.5,
    "prune_cdf": 0.99,
    "imp_metric": "outdoor",
    # C5 schedules
    "lr_max_steps": 30000,
    "lr_final_ratio": 0.01,
    "lr_restart_offset": 5000,
    "sh_restart": True,
}

STAT_KEYS = ("grad2d", "count", "blur")


class MiniSplattingDensity:
    def __init__(self, spec):
        unknown = set(spec.config) - set(DEFAULTS)
        if unknown:
            raise ValueError(f"Unknown Mini-Splatting density options: {sorted(unknown)}")
        self.cfg = {**DEFAULTS, **spec.config}
        if self.cfg["imp_metric"] not in ("outdoor", "indoor"):
            raise ValueError("imp_metric must be 'outdoor' or 'indoor'")
        self.requires = {"gradient.screen.signed", TARGET_CAPABILITY}
        self.state = None
        self.cameras = None
        self.images = None
        self.stats = {}

    # ------------------------------------------------------------------ runtime hooks
    def initialize(self, params, optimizers, scene_scale=1.0):
        self.extent = float(scene_scale)
        self.lr_init = float(optimizers["means"].param_groups[0]["lr"])
        self._reset_stats(len(params["means"]), params["means"].device)
        # Small JSON-like event log; saved in the checkpoint by the runtime.
        self.state = {"events": [], "depth_views_short": 0}

    def set_training_cameras(self, cameras):
        self.cameras = list(cameras)
        self.viewmats = torch.stack([c["viewmat"] for c in self.cameras]).flatten(1)
        self.images = [None] * len(self.cameras)
        self._cached = 0

    def before_backward(self, params, optimizers, step, outputs):
        if step < self.cfg["densify_until"]:
            outputs["info"]["means2d"].retain_grad()

    @torch.no_grad()
    def after_backward(self, params, optimizers, step, outputs):
        cfg = self.cfg
        self._cache_target(outputs, step)
        if step < cfg["densify_until"]:
            self._accumulate(params, outputs["info"])
            n = len(params["means"])
            if (
                step > cfg["densify_from"]
                and step % cfg["densify_every"] == 0
                and step % cfg["reinit_every"] != 0
                and (cfg["num_max"] is None or n < cfg["num_max"])
            ):
                self._densify(params, optimizers, step)
            if step % cfg["reinit_every"] == 0:
                self._depth_reinit(params, optimizers, step)
        if step == cfg["restart_step"]:
            if cfg["simplify"]:
                self._simplify_sample(params, optimizers, step)
            else:
                self._log(step, "restart", n_before=len(params["means"]))
                self._reinit(params, optimizers, params["means"].detach().clone(),
                             params["sh0"].detach().clone())
        if cfg["simplify"] and step == cfg["simp_iteration2"]:
            self._simplify_prune(params, optimizers, step)
        self._schedule(params, optimizers, step + 1)

    # ------------------------------------------------------------------ statistics
    def _reset_stats(self, n, device):
        self.stats = {
            "grad2d": torch.zeros(n, device=device),
            "count": torch.zeros(n, device=device),
            "blur": torch.zeros(n, dtype=torch.bool, device=device),
        }

    def _accumulate(self, params, info):
        grad = info["means2d"].grad
        if grad is None:
            raise RuntimeError("means2d gradient missing; before_backward must retain it")
        w, h = info["width"], info["height"]
        grads = grad[0].clone()
        grads[:, 0] *= w / 2.0  # 3DGS NDC-space viewspace gradient
        grads[:, 1] *= h / 2.0
        visible = (info["radii"][0] > 0).all(dim=-1)
        self.stats["grad2d"][visible] += grads[visible].norm(dim=-1)
        self.stats["count"][visible] += 1
        # Eq. 1: S_i > theta_blur * H * W from the step's own render (official render_imp)
        area_max = kernels.blend_stats(info, weights=False)["area_max"]
        self.stats["blur"] |= area_max > self.cfg["theta_blur"] * h * w

    # ------------------------------------------------------------------ C1 densification
    def _densify(self, params, optimizers, step):
        cfg = self.cfg
        n0 = len(params["means"])
        grads = self.stats["grad2d"] / self.stats["count"]
        grads[grads.isnan()] = 0.0
        max_scale = params["scales"].exp().max(dim=-1).values
        high = grads >= cfg["grad_threshold"]
        small = max_scale <= cfg["percent_dense"] * self.extent
        clone = high & small
        # densify_and_split_mask: gradient split OR blur split (Eq. 1)
        split_mask = (high & ~small) | self.stats["blur"]
        n_blur_only = int((self.stats["blur"] & ~(high & ~small)).sum())
        empty = {}
        duplicate(params, optimizers, empty, clone)
        n_clone = int(clone.sum())
        split_mask = torch.cat([split_mask, torch.zeros(n_clone, dtype=torch.bool,
                                                        device=split_mask.device)])
        n_split = int(split_mask.sum())
        split(params, optimizers, empty, split_mask)
        prune = params["opacities"].sigmoid() < cfg["min_opacity"]
        if step > cfg["big_world_from"]:
            # The 3DGS screen-size test (max_radii2D > 20) is inert: densification_postfix
            # zeroes max_radii2D before the prune test (official gaussian_model.py).
            prune |= params["scales"].exp().max(dim=-1).values > cfg["big_world"] * self.extent
        remove(params, optimizers, empty, prune)
        self._reset_stats(len(params["means"]), params["means"].device)
        if step % 1000 == 0:
            self._log(step, "densify", n_before=n0, clone=n_clone, split=n_split,
                      blur_only=n_blur_only, pruned=int(prune.sum()), n_after=len(params["means"]))

    # ------------------------------------------------------------------ C2 depth reinit
    def _cache_target(self, outputs, step):
        """Keep training-view images (uint8, lossless for 8-bit sources) for depth reinit."""
        if self.cameras is None or self._cached == len(self.cameras):
            return
        if step > self._last_reinit_step() or outputs.get("target_image") is None:
            return
        d = (self.viewmats - outputs["viewmat"].flatten()[None]).abs().sum(dim=1)
        k = int(d.argmin())
        if float(d[k]) > 1e-5:
            raise RuntimeError("Rendered frame is not one of the training cameras")
        if self.images[k] is None:
            self.images[k] = (outputs["target_image"].clamp(0, 1) * 255).round().to(torch.uint8)
            self._cached += 1

    def _last_reinit_step(self):
        e, u = self.cfg["reinit_every"], self.cfg["densify_until"]
        return ((u - 1) // e) * e

    def _depth_reinit(self, params, optimizers, step):
        if self.cameras is None:
            raise RuntimeError("Depth reinitialization needs set_training_cameras()")
        missing = [k for k, img in enumerate(self.images) if img is None]
        if missing:
            raise RuntimeError(
                f"{len(missing)} training views were never rendered before step {step}; "
                "their ground-truth colours are unavailable for depth reinitialization"
            )
        cfg = self.cfg
        n0 = len(params["means"])
        pts_list, rgb_list = [], []
        for cam, img in zip(self.cameras, self.images):
            meta = kernels.project(params, cam)
            out = kernels.depth_points(meta, params, cam)
            prob = (1 - out["T"]).flatten()
            h, w = int(cam["height"]), int(cam["width"])
            want = core.depth_samples_per_view(h, w, len(self.cameras), cfg["num_depth"])
            if cfg["depth_with_replacement"]:
                idx = core.weighted_sample(prob, want, replace=True)
            else:
                if int((prob > 0).sum()) < want:
                    self.state["depth_views_short"] += 1
                idx = core.weighted_sample(prob, want, replace=False)
            pts_list.append(out["points"].reshape(-1, 3)[idx])
            rgb_list.append(img.reshape(-1, 3)[idx].float() / 255.0)
        pts, rgb = torch.cat(pts_list), torch.cat(rgb_list)
        self._reinit(params, optimizers, pts, ((rgb - 0.5) / core.SH_C0)[:, None, :])
        self._log(step, "depth_reinit", n_before=n0, n_after=len(params["means"]))
        torch.cuda.empty_cache()

    def _reinit(self, params, optimizers, means, sh0):
        """Official reinitial_pts + training_setup: k-NN scales, identity rotation, opacity 0.1,
        zero higher SH, fresh Adam state for every field."""
        n = len(means)
        dist2 = kernels.knn_dist2(means).clamp_min(1e-7)
        quats = torch.zeros(n, 4, device=means.device)
        quats[:, 0] = 1
        p = torch.tensor(self.cfg["reinit_opacity"])
        values = {
            "means": means.float(),
            "scales": dist2.sqrt().log()[:, None].repeat(1, 3),
            "quats": quats,
            "opacities": torch.full((n,), float(torch.log(p / (1 - p))), device=means.device),
            "sh0": sh0.float().reshape(n, 1, 3),
            "shN": torch.zeros(n, 15, 3, device=means.device),
        }
        extra = set(params) - set(values)
        if extra:
            raise ValueError(f"Mini-Splatting reinit does not know fields {sorted(extra)}")
        self._replace_all(params, optimizers, values)

    def _replace_all(self, params, optimizers, values):
        for name, value in values.items():
            p = nn.Parameter(value.contiguous())
            opt = optimizers[name]
            opt.state.clear()
            opt.param_groups[0]["params"] = [p]
            params[name] = p
        self._reset_stats(len(params["means"]), params["means"].device)

    # ------------------------------------------------------------------ C3/C4 simplification
    def _importance(self, params):
        n = len(params["means"])
        device = params["means"].device
        imp = torch.zeros(n, device=device)
        accum_area_max = torch.zeros(n, device=device)
        for cam in self.cameras:
            view = kernels.blend_stats(kernels.project(params, cam), weights=True)
            imp, accum_area_max = core.accumulate_importance(
                imp, accum_area_max, view, self.cfg["imp_metric"]
            )
        imp[accum_area_max == 0] = 0  # Eq. 2: intersection preserving
        return imp

    def _simplify_sample(self, params, optimizers, step):
        if self.cameras is None:
            raise RuntimeError("Simplification needs set_training_cameras()")
        n0 = len(params["means"])
        imp = self._importance(params)
        keep_n = core.sampling_count(imp, self.cfg["sampling_factor"])
        idx = core.weighted_sample(imp, keep_n, replace=False)
        keep = torch.zeros(n0, dtype=torch.bool, device=imp.device)
        keep[idx] = True
        self._reinit(params, optimizers, params["means"].detach()[keep].clone(),
                     params["sh0"].detach()[keep].clone())
        self._log(step, "simplify_sample", n_before=n0, intersected=int((imp > 0).sum()),
                  n_after=len(params["means"]))
        torch.cuda.empty_cache()

    def _simplify_prune(self, params, optimizers, step):
        n0 = len(params["means"])
        imp = self._importance(params)
        keep = core.cdf_keep_mask(imp, self.cfg["prune_cdf"])
        self._replace_all(params, optimizers, {k: v.detach()[keep] for k, v in params.items()})
        self._log(step, "simplify_prune", n_before=n0, n_after=len(params["means"]))
        torch.cuda.empty_cache()

    # ------------------------------------------------------------------ C5 schedules
    def _schedule(self, params, optimizers, next_step):
        cfg = self.cfg
        optimizers["means"].param_groups[0]["lr"] = core.means_lr(
            next_step, self.lr_init, restart_step=cfg["restart_step"],
            restart_offset=cfg["lr_restart_offset"], lr_final_ratio=cfg["lr_final_ratio"],
            max_steps=cfg["lr_max_steps"],
        )
        if cfg["sh_restart"]:
            k = core.active_rest_coeffs(core.sh_degree(next_step, cfg["restart_step"]))
            if k < 15:
                p = params["shN"]
                p.data[:, k:] = 0
                st = optimizers["shN"].state.get(p, {})
                for key in ("exp_avg", "exp_avg_sq"):
                    if key in st:
                        st[key][:, k:] = 0

    def _log(self, step, kind, **values):
        row = {"step": int(step), "event": kind, **{k: int(v) for k, v in values.items()}}
        self.state["events"].append(row)
        print({"mini_splatting": row}, flush=True)


def create_density(spec):
    return MiniSplattingDensity(spec)
