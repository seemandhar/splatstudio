"""Perceptual-GS density control (C3 guided densification, C4 scene-adaptive depth reinit,
C5 opacity decline), after the official `train.py` and `scene/gaussian_model.py` of
https://github.com/eezkni/Perceptual-GS (Inria 3DGS licence; see csrc/LICENSE-3DGS.md).

Per training step s (the runtime calls `after_backward` after the optimizer step; the official
code also steps the optimizer before computing the weight statistics):
  s < 15000: 3DGS viewspace-gradient statistics from the RGB render.
  s <= 15000, s % 1000 == 0 or s % 1500 == 0: omega_max = max over training views of
      sum_pixels alpha T (Eq. 14; Mini-Splatting `render_imp` kernel).
  s == 600: split_only = beta > 0.85 (Eqs. 15-16); sparse = gamma > 0.55 (Eqs. 17-18).
  500 < s < 15000, s % 100 == 0 (not when sparse and s % 5000 == 0): densify_and_prune with
      sensitivity-guided masks (Eqs. 11-13) and opacity decline on clones (Eqs. 19-20).
  s < 15000, sparse, s % 5000 == 0: depth reinitialization (Sec. 3.6, after Mini-Splatting).
  s < 15000, not sparse, s % 3000 == 0: 3DGS opacity reset (min(alpha, 0.01)).
  s == 15000, sparse: re-initialize from the current centres and DC colours.
"""

import torch
from gsplat.strategy.ops import duplicate, remove, split
from torch import nn

from . import core, kernels, sidecar
from .renderer import TARGET_CAPABILITY

DEFAULTS = {
    # 3DGS (official arguments/__init__.py OptimizationParams)
    "densify_from": 500,
    "densify_until": 15000,
    "densify_every": 100,
    "grad_threshold": 0.0002,
    "percent_dense": 0.01,
    "min_opacity": 0.005,
    "big_world_from": 3000,  # size_threshold set when iteration > opacity_reset_interval
    "big_world": 0.1,
    "opacity_reset_every": 3000,
    "opacity_reset_value": 0.01,
    # C3 (Table 8)
    "hd_interval": 1000,
    "md_interval": 1500,
    "guided_from": 1000,
    "threshold_high": 0.9,
    "threshold_low": 0.3,
    "complex_weight": 25.0,
    "mix_weight": 10.0,
    "beta_threshold": 0.85,
    # C4
    "decision_step": 600,
    "gamma_threshold": 0.55,
    "reinit_every": 5000,
    "num_depth": 3_500_000,
    "reinit_opacity": 0.1,
    "reinit_sensitivity": 0.1,
    "adaptive_reinit": True,  # False: never sparse (ablation)
    # C5
    "od_k": 1.2,  # None disables opacity decline
    "sidecar_root": sidecar.DEFAULT_SIDECAR_ROOT,
}


class PerceptualDensity:
    def __init__(self, spec):
        unknown = set(spec.config) - set(DEFAULTS)
        if unknown:
            raise ValueError(f"Unknown Perceptual-GS density options: {sorted(unknown)}")
        self.cfg = {**DEFAULTS, **spec.config}
        self.requires = {"gradient.screen.signed", TARGET_CAPABILITY}
        self.state = None
        self.cameras = None
        self.images = None
        self.table = None  # sidecar.Table (set by probes, else loaded from --data at decision)
        self.split_only = False
        self.sparse = False

    # ------------------------------------------------------------------ runtime hooks
    def initialize(self, params, optimizers, scene_scale=1.0):
        if "sensitivity" not in params:
            raise ValueError("Perceptual-GS needs the per-Gaussian 'sensitivity' field")
        self.extent = float(scene_scale)
        self._reset_stats(params)
        self.state = {"events": [], "split_only": False, "sparse": False}

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
        until = cfg["densify_until"]
        self._cache_target(outputs, step)
        if step < until:
            self._accumulate(outputs["info"])
        if step == cfg["decision_step"]:
            self._decide(params, step)
        reinit_now = self.sparse and step % cfg["reinit_every"] == 0
        densify_now = step > cfg["densify_from"] and step % cfg["densify_every"] == 0
        if step < until and densify_now and not reinit_now:
            if self._guided(step, "hd") or self._guided(step, "md"):
                self._weights(params)
            self._densify(params, optimizers, step)
        if step < until and reinit_now:
            self._depth_reinit(params, optimizers, step)
        if step < until and not self.sparse and step % cfg["opacity_reset_every"] == 0:
            self._reset_opacity(params, optimizers)
        if self.sparse and step == until:
            self._log(step, "self_reinit", n_before=len(params["means"]))
            self._reinit(params, optimizers, params["means"].detach().clone(),
                         params["sh0"].detach().clone())

    # ------------------------------------------------------------------ statistics
    def _reset_stats(self, params):
        n, device = len(params["means"]), params["means"].device
        self.grad2d = torch.zeros(n, device=device)
        self.count = torch.zeros(n, device=device)
        self.trans = torch.zeros(n, device=device)

    def _accumulate(self, info):
        grad = info["means2d"].grad
        if grad is None:
            raise RuntimeError("means2d gradient missing; before_backward must retain it")
        grads = grad[0].clone()
        grads[:, 0] *= info["width"] / 2.0  # 3DGS NDC-space viewspace gradient
        grads[:, 1] *= info["height"] / 2.0
        visible = (info["radii"][0] > 0).all(dim=-1)
        self.grad2d[visible] += grads[visible].norm(dim=-1)
        self.count[visible] += 1

    def _guided(self, step, kind):
        interval = self.cfg["hd_interval" if kind == "hd" else "md_interval"]
        return step >= self.cfg["guided_from"] and step % interval == 0

    def _weights(self, params):
        """Eq. 14: omega_max_i = max_v sum_pixels w_i^v (official `trans`)."""
        if self.cameras is None:
            raise RuntimeError("Guided densification needs set_training_cameras()")
        trans = torch.zeros(len(params["means"]), device=params["means"].device)
        for cam in self.cameras:
            w = kernels.blend_stats(kernels.project(params, cam), weights=True)["accum_weights"]
            trans = torch.maximum(trans, w.float())
        self.trans = trans

    # ------------------------------------------------------------------ C4 decision
    def _gt_maps(self):
        if self.table is None:
            self.table = sidecar.table(sidecar_root=self.cfg["sidecar_root"])
        return [self.table.sens(c["viewmat"]) for c in self.cameras]

    def _decide(self, params, step):
        cfg = self.cfg
        sens = params["sensitivity"].sigmoid()
        max_scale = params["scales"].exp().max(dim=-1).values
        gamma = core.sparse_indicator(max_scale, sens, cfg["threshold_low"], cfg["threshold_high"])
        beta = core.scene_sensitivity(self._gt_maps())
        self.sparse = bool(cfg["adaptive_reinit"]) and gamma > cfg["gamma_threshold"]
        self.split_only = beta > cfg["beta_threshold"]
        self.state.update(split_only=self.split_only, sparse=self.sparse, gamma=gamma, beta=beta)
        self._log(step, "decision", split_only=self.split_only, sparse=self.sparse,
                  gamma_x1e4=round(gamma * 1e4), beta_x1e4=round(beta * 1e4))

    # ------------------------------------------------------------------ C3 + C5 densification
    def masks(self, params, step):
        """Official densify_and_clone / densify_and_split selection masks (Eqs. 11-13)."""
        cfg = self.cfg
        grads = self.grad2d / self.count
        grads[grads.isnan()] = 0.0
        max_scale = params["scales"].exp().max(dim=-1).values
        sens = params["sensitivity"].sigmoid()
        high_grad = grads >= cfg["grad_threshold"]
        small = max_scale <= cfg["percent_dense"] * self.extent
        complex_ = (sens >= cfg["threshold_high"]) & (self.trans > cfg["complex_weight"])
        clone = high_grad & small
        split_mask = high_grad & ~small
        if self._guided(step, "hd"):
            if not self.split_only:
                clone |= complex_ & small
                split_mask |= complex_ & ~small
            else:
                split_mask |= complex_
        if self._guided(step, "md"):
            mixed = (sens < cfg["threshold_high"]) & (sens > cfg["threshold_low"])
            split_mask |= mixed & (self.trans > cfg["mix_weight"])
        return clone, split_mask

    def _densify(self, params, optimizers, step):
        cfg = self.cfg
        n0 = len(params["means"])
        clone, split_mask = self.masks(params, step)
        if cfg["od_k"] is not None and bool(clone.any()):
            # Eqs. 19-20 applied in place to the selected originals; clones copy it.
            alpha = params["opacities"][clone].sigmoid()
            declined = core.opacity_decline(alpha, cfg["od_k"]).clamp(1e-7, 1 - 1e-7)
            params["opacities"].data[clone] = torch.logit(declined)
        empty = {}
        duplicate(params, optimizers, empty, clone)
        n_clone = int(clone.sum())
        split_mask = torch.cat([split_mask, torch.zeros(n_clone, dtype=torch.bool,
                                                        device=split_mask.device)])
        n_split = int(split_mask.sum())
        split(params, optimizers, empty, split_mask)
        prune = params["opacities"].sigmoid() < cfg["min_opacity"]
        if step > cfg["big_world_from"]:
            # The 3DGS screen-size test (max_radii2D > 20) is inert in the official code:
            # densification_postfix zeroes max_radii2D before the prune test.
            prune |= params["scales"].exp().max(dim=-1).values > cfg["big_world"] * self.extent
        remove(params, optimizers, empty, prune)
        self._reset_stats(params)
        _check_trainable(params)
        if step % 500 == 0:
            self._log(step, "densify", n_before=n0, clone=n_clone, split=n_split,
                      pruned=int(prune.sum()), n_after=len(params["means"]))

    def _reset_opacity(self, params, optimizers):
        """Official reset_opacity: logit(min(alpha, 0.01)) with zeroed Adam moments."""
        p = params["opacities"]
        value = core.logit(self.cfg["opacity_reset_value"])
        new = torch.minimum(p.detach(), torch.full_like(p, value))
        self._replace(params, optimizers, "opacities", new, reset_moments=True)

    # ------------------------------------------------------------------ C4 depth reinit
    def _cache_target(self, outputs, step):
        """Keep training-view images (uint8, lossless for 8-bit sources) for depth reinit."""
        if self.cameras is None or self._cached == len(self.cameras):
            return
        if step >= self.cfg["densify_until"] or outputs.get("target_image") is None:
            return
        d = (self.viewmats - outputs["viewmat"].flatten()[None]).abs().sum(dim=1)
        k = int(d.argmin())
        if float(d[k]) > 1e-5:
            raise RuntimeError("Rendered frame is not one of the training cameras")
        if self.images[k] is None:
            self.images[k] = (outputs["target_image"].clamp(0, 1) * 255).round().to(torch.uint8)
            self._cached += 1

    def _depth_reinit(self, params, optimizers, step):
        if self.cameras is None:
            raise RuntimeError("Depth reinitialization needs set_training_cameras()")
        missing = [k for k, img in enumerate(self.images) if img is None]
        if missing:
            raise RuntimeError(
                f"{len(missing)} training views were never rendered before step {step}; "
                "their ground-truth colours are unavailable for depth reinitialization"
            )
        n0 = len(params["means"])
        pts_list, rgb_list = [], []
        for cam, img in zip(self.cameras, self.images):
            meta = kernels.project(params, cam)
            out = kernels.depth_points(meta, params, cam)
            prob = (1 - out["T"]).flatten().clamp_min(0)
            h, w = int(cam["height"]), int(cam["width"])
            want = core.depth_samples_per_view(h, w, len(self.cameras), self.cfg["num_depth"])
            idx = core.weighted_sample_unique(prob, want)
            pts_list.append(out["points"].reshape(-1, 3)[idx])
            rgb_list.append(img.reshape(-1, 3)[idx].float() / 255.0)
        pts, rgb = torch.cat(pts_list), torch.cat(rgb_list)
        self._reinit(params, optimizers, pts, ((rgb - 0.5) / core.SH_C0)[:, None, :])
        self._log(step, "depth_reinit", n_before=n0, n_after=len(params["means"]))
        torch.cuda.empty_cache()

    def _reinit(self, params, optimizers, means, sh0):
        """Official reinitial_pts + training_setup: simple-knn scales, identity rotation,
        opacity 0.1, sensitivity 0.1, zero higher SH, fresh Adam state for every field."""
        n = len(means)
        dist2 = kernels.knn_dist2(means).clamp_min(1e-7)
        quats = torch.zeros(n, 4, device=means.device)
        quats[:, 0] = 1
        values = {
            "means": means.float(),
            "scales": dist2.sqrt().log()[:, None].repeat(1, 3),
            "quats": quats,
            "opacities": torch.full((n,), core.logit(self.cfg["reinit_opacity"]),
                                    device=means.device),
            "sh0": sh0.float().reshape(n, 1, 3),
            "shN": torch.zeros(n, 15, 3, device=means.device),
            "sensitivity": torch.full((n,), core.logit(self.cfg["reinit_sensitivity"]),
                                      device=means.device),
        }
        if set(params) != set(values):
            raise ValueError(f"Perceptual-GS reinit does not know fields {sorted(set(params))}")
        for name, value in values.items():
            self._replace(params, optimizers, name, value, reset_moments=True, clear=True)
        self._reset_stats(params)
        _check_trainable(params)

    @staticmethod
    def _replace(params, optimizers, name, value, reset_moments, clear=False):
        """New trainable leaf Parameter (requires_grad copied from the old one, which is
        always True here) and matching optimizer state."""
        old = params[name]
        p = nn.Parameter(value.detach().clone().contiguous(), requires_grad=old.requires_grad)
        opt = optimizers[name]
        st = opt.state.pop(old, None)
        if clear or st is None:
            st = {}
        elif reset_moments:
            for key in ("exp_avg", "exp_avg_sq"):
                if key in st:
                    st[key] = torch.zeros_like(st[key])
        if st:
            opt.state[p] = st
        opt.param_groups[0]["params"] = [p]
        params[name] = p

    def _log(self, step, kind, **values):
        row = {"step": int(step), "event": kind, **{k: int(v) for k, v in values.items()}}
        self.state["events"].append(row)
        print({"perceptual_gs": row}, flush=True)


def _check_trainable(params):
    """Guard against the ResGS pitfall: every field must remain a trainable leaf."""
    for name, p in params.items():
        if not (p.is_leaf and p.requires_grad):
            raise RuntimeError(f"Perceptual-GS density left field {name!r} untrainable")


def create_density(spec):
    return PerceptualDensity(spec)
