"""Mini-Splatting2 density control, after the official `msv2/train.py`, `msv2_d/train.py` and
`scene/gaussian_model.py` of https://github.com/fatPeter/mini-splatting2 (config/fast).

Per training step s (the runtime calls `after_backward` after the optimizer step):
  s < densify_until (3000): 3DGS gradient statistics (x factor_culling on culled views) and the
      Mini-Splatting blur mask from the step's own render; every 100 steps (s > 500, s != 2000)
      clone / split-or-blur / prune; s == 2000: intersection preserving + depth reinit with
      num_depth = N; every 250 steps from 500 (s != 2000): culling_with_clone = per-view
      visibility masks (C4) + prune + critical Gaussian identification (C1) + aggressive clone (C2).
  s == 3000: visibility + importance sampling (MSv2) / importance pruning (MSv2-D); SH enabled;
      fresh Adam.   s == 8000: visibility + intersection preserving / importance pruning.
  s == 13000: visibility culling switched off.
  every step: means LR lr(s+1) (30000-step horizon), SH bands of degree(s+1), warm-up flag.
"""

import torch
from gsplat import rasterization
from gsplat.strategy.ops import duplicate, remove, split
from torch import nn

from . import core, kernels, link
from .optim import SparseGaussianAdam, sparsify
from .renderer import CULLING_CAPABILITY, TARGET_CAPABILITY, full_index

DEFAULTS = {
    "total_steps": 18000,  # config/fast iterations; culling ends at (simp2 + total) // 2
    # 3DGS densification kept by Mini-Splatting(2) (official OptimizationParams / config/fast)
    "densify_from": 500,
    "densify_every": 100,
    "densify_until": 3000,
    "grad_threshold": 0.0002,
    "percent_dense": 0.01,
    "min_opacity": 0.005,
    "theta_blur": 2e-4,  # mask_blur: area_max > H*W/5000
    # C1/C2 aggressive clone (culling_with_clone)
    "clone_from": 500,
    "clone_every": 250,
    "critical_cdf": 0.99,
    "clone_prune_cdf": 0.999,
    "clone_min_opacity": 0.0051,
    # C4 visibility culling
    "culling": True,
    "visibility_cdf": 0.99,
    # C3 depth reinitialization at 2000 with num_depth = N * factor
    "depth_reinit_iter": 2000,
    "num_depth_factor": 1.0,
    "preserve_before_reinit": True,  # interesction_preserving(0.99) before depth_reinit
    "reinit_opacity": 0.1,
    # simplification
    "simplify": "sampling",  # MSv2: sampling at 3k + preserving at 8k; MSv2-D: "importance"
    "simp_iteration1": 3000,
    "simp_iteration2": 8000,
    "sampling_factor": 0.6,
    "prune_cdf": 0.99,
    "imp_metric": "outdoor",
    # accelerated training
    "warmup_until": 3000,  # 0.5x resolution while iteration <= 3000
    "sh_from": 3000,  # SH degree 0 until simp_iteration1, then +1 per 1000 steps
    "lr_max_steps": 30000,
    "lr_final_ratio": 0.01,
    "opacity_lr": 0.025,
    "sparse_adam": True,
}

STAT_KEYS = ("grad2d", "count", "blur")


def visible_mask(radii):
    """gsplat 1.5 radii are [N,2] (x/y extents); older versions [N]."""
    return (radii > 0).all(dim=-1) if radii.dim() == 2 else radii > 0


class MiniSplatting2Density:
    def __init__(self, spec):
        unknown = set(spec.config) - set(DEFAULTS)
        if unknown:
            raise ValueError(f"Unknown Mini-Splatting2 density options: {sorted(unknown)}")
        self.cfg = {**DEFAULTS, **spec.config}
        if self.cfg["imp_metric"] not in ("outdoor", "indoor"):
            raise ValueError("imp_metric must be 'outdoor' or 'indoor'")
        if self.cfg["simplify"] not in ("sampling", "importance"):
            raise ValueError("simplify must be 'sampling' or 'importance'")
        self.requires = {"gradient.screen.signed", TARGET_CAPABILITY, CULLING_CAPABILITY}
        self.state = None
        self.table = link.LinkTable()
        self.cameras = None
        self.half_cameras = None
        self.images = None
        self.buf = {}

    # ------------------------------------------------------------------ runtime hooks
    def initialize(self, params, optimizers, scene_scale=1.0):
        cfg = self.cfg
        self.extent = float(scene_scale)
        self.lr_init = float(optimizers["means"].param_groups[0]["lr"])
        optimizers["opacities"].param_groups[0]["lr"] = float(cfg["opacity_lr"])
        if cfg["sparse_adam"]:
            sparsify(optimizers)
        link.register(params, self.table)
        self.table.half_res = 1 <= cfg["warmup_until"]
        n, device = len(params["means"]), params["means"].device
        self.buf = {"factor": torch.ones(n, device=device)}
        self._reset_stats(n, device)
        self.state = {"events": [], "depth_views_fallback": 0}

    def set_training_cameras(self, cameras):
        from .renderer import half_res_camera

        self.cameras = list(cameras)
        self.half_cameras = [half_res_camera(c) for c in self.cameras]
        self.table.viewmats = torch.stack([c["viewmat"] for c in self.cameras]).flatten(1)
        self.images = [None] * len(self.cameras)

    def before_backward(self, params, optimizers, step, outputs):
        info = outputs["info"]
        if step < self.cfg["densify_until"]:
            info["means2d"].retain_grad()
        if self.cfg["sparse_adam"]:
            n = len(params["means"])
            idx = full_index(outputs, n, params["means"].device)
            vis = torch.zeros(n, dtype=torch.bool, device=params["means"].device)
            vis[idx[visible_mask(info["radii"][0])]] = True
            for opt in optimizers.values():
                if not isinstance(opt, SparseGaussianAdam):
                    raise RuntimeError("sparse_adam expects the optimizers set in initialize()")
                opt.visibility = vis

    @torch.no_grad()
    def after_backward(self, params, optimizers, step, outputs):
        cfg = self.cfg
        self._cache_target(outputs, step)
        if step < cfg["densify_until"]:
            self._accumulate(params, outputs)
        for event in core.events(step, cfg):
            n0 = len(params["means"])
            getattr(self, "_" + event)(params, optimizers, step)
            if event != "densify" or step % 500 == 0:
                self._log(step, event, n_before=n0, n_after=len(params["means"]))
        self._schedule(params, optimizers, step + 1)

    # ------------------------------------------------------------------ buffers
    def _reset_stats(self, n, device):
        self.buf["grad2d"] = torch.zeros(n, device=device)
        self.buf["count"] = torch.zeros(n, device=device)
        self.buf["blur"] = torch.zeros(n, dtype=torch.bool, device=device)

    def _set_culling(self, cull, factor):
        if self.cfg["culling"]:
            self.buf["cull"] = cull
            self.buf["factor"] = factor
        self._sync()

    def _culling_off(self, params, optimizers, step):
        self.buf.pop("cull", None)
        self.buf["factor"] = torch.ones(len(params["means"]), device=params["means"].device)
        self._sync()

    def _sync(self):
        self.table.cull = self.buf.get("cull")

    def _remove(self, params, optimizers, mask):
        remove(params, optimizers, self.buf, mask)
        self._sync()

    # ------------------------------------------------------------------ statistics
    def _accumulate(self, params, outputs):
        info = outputs["info"]
        grad = info["means2d"].grad
        if grad is None:
            raise RuntimeError("means2d gradient missing; before_backward must retain it")
        n, device = len(params["means"]), params["means"].device
        idx = full_index(outputs, n, device)
        w, h = info["width"], info["height"]
        grads = grad[0].clone()
        grads[:, 0] *= w / 2.0  # 3DGS NDC-space viewspace gradient
        grads[:, 1] *= h / 2.0
        vis = visible_mask(info["radii"][0])
        sel = idx[vis]
        norms = grads[vis].norm(dim=-1)
        if outputs.get("subset_index") is not None:
            # add_densification_stats_culling: normalize the xy gradient after culling
            norms = norms * self.buf["factor"][sel]
        self.buf["grad2d"][sel] += norms
        self.buf["count"][sel] += 1
        area_max = kernels.blend_stats(info, weights=False)["area_max"]
        self.buf["blur"][idx] |= area_max > self.cfg["theta_blur"] * h * w

    def _view_stats(self, params, cam, cull_col=None):
        """Official render_simp statistics of one view at full Gaussian length (culled = 0)."""
        n, device = len(params["means"]), params["means"].device
        sub = None
        if cull_col is not None and bool(cull_col.any()):
            sub = (~cull_col).nonzero().squeeze(1)
        p = params if sub is None else {k: params[k][sub] for k in kernels_fields(params)}
        meta = kernels.project(p, cam)
        st = kernels.blend_stats(meta, weights=True)
        st["visible"] = visible_mask(meta["radii"][0])
        if sub is None:
            return st
        full = {}
        for key in ("accum_weights", "area_proj", "area_max", "visible"):
            v = st[key]
            out = torch.zeros(n, dtype=v.dtype, device=device)
            out[sub] = v
            full[key] = out
        return full

    def _cams(self, step):
        if self.cameras is None:
            raise RuntimeError("Mini-Splatting2 needs set_training_cameras()")
        return self.half_cameras if step <= self.cfg["warmup_until"] else self.cameras

    def _visibility_pass(self, params, step):
        """Shared loop of the official culling_with_* functions (culling reset beforehand)."""
        cams = self._cams(step)
        n, v, device = len(params["means"]), len(cams), params["means"].device
        out = {
            "imp_w": torch.zeros(n, device=device),
            "imp_m": torch.zeros(n, device=device),
            "area": torch.zeros(n, device=device),
            "cull": torch.zeros(n, v, dtype=torch.bool, device=device),
        }
        count_rad = torch.zeros(n, device=device)
        count_vis = torch.zeros(n, device=device)
        for k, cam in enumerate(cams):
            st = self._view_stats(params, cam)
            out["imp_w"] += st["accum_weights"]
            core.accumulate_importance(out["imp_m"], out["area"], st, self.cfg["imp_metric"])
            vis = core.cdf_keep_mask(st["accum_weights"], self.cfg["visibility_cdf"])
            out["cull"][:, k] = ~vis
            count_rad += st["visible"].float()
            count_vis += vis.float()
        out["count_vis"] = count_vis
        out["factor"] = core.culling_factor(count_vis, count_rad)
        return out

    def _importance_culled(self, params, step):
        """Importance (imp_metric) with the current culling masks, intersection preserved."""
        cams = self._cams(step)
        n, device = len(params["means"]), params["means"].device
        imp = torch.zeros(n, device=device)
        area = torch.zeros(n, device=device)
        cull = self.buf.get("cull")
        for k, cam in enumerate(cams):
            st = self._view_stats(params, cam, None if cull is None else cull[:, k])
            core.accumulate_importance(imp, area, st, self.cfg["imp_metric"])
        imp[area == 0] = 0
        return imp

    # ------------------------------------------------------------------ 3DGS + blur densify
    def _densify(self, params, optimizers, step):
        cfg = self.cfg
        grads = self.buf["grad2d"] / self.buf["count"]
        grads[grads.isnan()] = 0.0
        max_scale = params["scales"].exp().max(dim=-1).values
        high = grads >= cfg["grad_threshold"]
        small = max_scale <= cfg["percent_dense"] * self.extent
        clone = high & small
        split_mask = (high & ~small) | self.buf["blur"]  # densify_and_split_mask
        duplicate(params, optimizers, self.buf, clone)
        split_mask = torch.cat([split_mask, torch.zeros(int(clone.sum()), dtype=torch.bool,
                                                        device=split_mask.device)])
        split(params, optimizers, self.buf, split_mask)
        # size_threshold is only set for iteration > 3000: inert while densifying until 3000
        self._remove(params, optimizers, params["opacities"].sigmoid() < cfg["min_opacity"])
        self._reset_stats(len(params["means"]), params["means"].device)

    # ------------------------------------------------------------------ C1 + C2 + C4
    def _culling_with_clone(self, params, optimizers, step):
        cfg = self.cfg
        vp = self._visibility_pass(params, step)
        keep = core.cdf_keep_mask(vp["imp_w"], cfg["clone_prune_cdf"])
        prune = (vp["count_vis"] <= 1) | ~keep
        # C1 critical Gaussians: max-weight Gaussians of some pixel, 0.99 CDF of sum w
        imp = vp["imp_w"].clone()
        imp[vp["area"] == 0] = 0
        critical = core.cdf_keep_mask(imp, cfg["critical_cdf"])[~prune]
        self._set_culling(vp["cull"], vp["factor"])
        self._remove(params, optimizers, prune)
        self._clone(params, optimizers, critical)
        self._reset_stats(len(params["means"]), params["means"].device)
        self.state["last_clone"] = {"step": int(step), "critical": int(critical.sum()),
                                    "pruned": int(prune.sum())}

    def _clone(self, params, optimizers, mask):
        """C2 official GaussianModel.clone: source overwritten, one copy with zero moments."""
        a_new, s_new = core.clone_values(params["opacities"][mask].sigmoid(),
                                         params["scales"][mask].exp(),
                                         self.cfg["clone_min_opacity"])
        params["opacities"].data[mask] = torch.logit(a_new)
        params["scales"].data[mask] = s_new.log()
        duplicate(params, optimizers, self.buf, mask)
        self._sync()

    # ------------------------------------------------------------------ C3 depth reinit
    def _cache_target(self, outputs, step):
        """Training-view targets at the reinit resolution (uint8, lossless for 8-bit data)."""
        if self.images is None or step > self.cfg["depth_reinit_iter"]:
            return
        k, img = outputs.get("view_index"), outputs.get("target_image")
        if k is None or img is None or self.images[k] is not None:
            return
        self.images[k] = (img.clamp(0, 1) * 255).round().to(torch.uint8)

    def _depth_reinit(self, params, optimizers, step):
        cfg = self.cfg
        num_depth = len(params["means"]) * cfg["num_depth_factor"]
        if cfg["preserve_before_reinit"]:
            # official interesction_preserving (better early point clouds), with culling
            imp = self._importance_culled(params, step)
            self._remove(params, optimizers, ~core.cdf_keep_mask(imp, cfg["prune_cdf"]))
        cams = self._cams(step)
        cull = self.buf.get("cull")
        pts_list, rgb_list = [], []
        for k, cam in enumerate(cams):
            col = None if cull is None else cull[:, k]
            p = params
            if col is not None and bool(col.any()):
                sub = (~col).nonzero().squeeze(1)
                p = {f: params[f][sub] for f in kernels_fields(params)}
            meta = kernels.project(p, cam)
            out = kernels.depth_points(meta, p, cam)
            prob = (1 - out["T"]).flatten()
            h, w = int(cam["height"]), int(cam["width"])
            want = core.depth_samples_per_view(h, w, len(cams), num_depth)
            idx = core.weighted_sample(prob, want, replace=False)
            pts_list.append(out["points"].reshape(-1, 3)[idx])
            if self.images[k] is not None:
                rgb = self.images[k].reshape(-1, 3)[idx].float() / 255.0
            else:
                # view never drawn before the reinit (random sampling): rendered colours
                self.state["depth_views_fallback"] += 1
                rgb = rendered_colors(p, cam).reshape(-1, 3)[idx]
            rgb_list.append(rgb)
        pts, rgb = torch.cat(pts_list), torch.cat(rgb_list)
        self._reinit(params, optimizers, pts, ((rgb - 0.5) / core.SH_C0)[:, None, :])
        self._culling_off(params, optimizers, step)
        torch.cuda.empty_cache()

    def _reinit(self, params, optimizers, means, sh0):
        """Official reinitial_pts + training_setup: simple-knn scales, identity rotation,
        opacity 0.1, zero SH rest, fresh optimizer state."""
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
            raise ValueError(f"Mini-Splatting2 reinit does not know fields {sorted(extra)}")
        for name, value in values.items():
            param = nn.Parameter(value.contiguous())
            opt = optimizers[name]
            opt.state.clear()
            opt.param_groups[0]["params"] = [param]
            params[name] = param
        self.buf = {"factor": torch.ones(n, device=means.device)}
        self._reset_stats(n, means.device)
        self._sync()

    # ------------------------------------------------------------------ simplification
    def _simplify1(self, params, optimizers, step):
        cfg = self.cfg
        vp = self._visibility_pass(params, step)
        if cfg["simplify"] == "sampling":
            # culling_with_interesction_sampling
            imp = vp["imp_m"]
            imp[vp["area"] == 0] = 0
            idx = core.weighted_sample(imp, core.sampling_count(imp, cfg["sampling_factor"]))
            keep = torch.zeros_like(imp, dtype=torch.bool)
            keep[idx] = True
        else:
            # culling_with_importance_pruning (MSv2-D)
            keep = core.cdf_keep_mask(vp["imp_w"], cfg["prune_cdf"])
        self._set_culling(vp["cull"], vp["factor"])
        self._remove(params, optimizers, (vp["count_vis"] <= 1) | ~keep)
        # max_sh_degree = 3, extend_features_rest (zeros), training_setup: fresh Adam
        params["shN"].data.zero_()
        for opt in optimizers.values():
            opt.state.clear()
        self._reset_stats(len(params["means"]), params["means"].device)
        torch.cuda.empty_cache()

    def _simplify2(self, params, optimizers, step):
        cfg = self.cfg
        vp = self._visibility_pass(params, step)
        if cfg["simplify"] == "sampling":
            # culling_with_interesction_preserving
            imp = vp["imp_m"]
            imp[vp["area"] == 0] = 0
            keep = core.cdf_keep_mask(imp, cfg["prune_cdf"])
        else:
            keep = core.cdf_keep_mask(vp["imp_w"], cfg["prune_cdf"])
        self._set_culling(vp["cull"], vp["factor"])
        self._remove(params, optimizers, (vp["count_vis"] <= 1) | ~keep)
        self._reset_stats(len(params["means"]), params["means"].device)
        torch.cuda.empty_cache()

    # ------------------------------------------------------------------ schedules
    def _schedule(self, params, optimizers, next_step):
        cfg = self.cfg
        optimizers["means"].param_groups[0]["lr"] = core.expon_lr(
            next_step, self.lr_init, cfg["lr_final_ratio"], cfg["lr_max_steps"])
        k = core.active_rest_coeffs(core.sh_degree(next_step, cfg["sh_from"]))
        if k < 15:
            p = params["shN"]
            p.data[:, k:] = 0
            st = optimizers["shN"].state.get(p, {})
            for key in ("exp_avg", "exp_avg_sq"):
                if key in st:
                    st[key][:, k:] = 0
        self.table.half_res = next_step <= cfg["warmup_until"]

    def _log(self, step, kind, **values):
        row = {"step": int(step), "event": kind, **{k: int(v) for k, v in values.items()}}
        self.state["events"].append(row)
        print({"mini_splatting2": row}, flush=True)


def kernels_fields(params):
    return ("means", "scales", "quats", "opacities", "sh0", "shN")


@torch.no_grad()
def rendered_colors(params, cam):
    """Background-free rendered colour C / (1 - T) of the SH-degree-0 model, [H,W,3]."""
    rgb, alpha, _ = rasterization(
        means=params["means"], quats=params["quats"], scales=params["scales"].exp(),
        opacities=params["opacities"].sigmoid(), colors=params["sh0"],
        viewmats=cam["viewmat"][None], Ks=cam["K"][None], width=int(cam["width"]),
        height=int(cam["height"]), sh_degree=0, packed=False)
    return (rgb[0] / alpha[0].clamp_min(1e-6)).clamp(0, 1)


def create_density(spec):
    return MiniSplatting2Density(spec)
