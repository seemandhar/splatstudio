"""EfficientGS density control, after the official `train.py` / `scene/gaussian_model.py` of
https://github.com/wenkaier/EfficientGS (train.sh flags: --use_grad_norm --grad_norm_way s1
--densify_grad_threshold 0.0007 --prune_by_max_weight --prune_way p4
--oneupSHdegree_from_iter 15500 --shs_lr_rate 5 --use_sparse_shs --sparse_rate 0.2).

The runtime calls `after_backward(step)` after the optimizer step of `step`:
  step < densify_until: accumulate S_g (Eq. 6, CUDA) and visibility counts; densify every
      100 steps after 500 (3DGS clone + split + prune); opacity reset every 3000 steps.
  step in prune_steps: top-K dominant-Gaussian pruning over all training views (Eq. 7/8).
  step + 1 == sh_alloc_step: fresh Adam state for shN (official set_shs_memory).
  step + 1 in sparse steps: Eq. 9 over all training views, +1 SH degree for the top r_s.
"""

import torch
from gsplat.strategy.ops import duplicate, remove, reset_opa, split

from . import core, kernels

DEFAULTS = {
    # 3DGS densification (official OptimizationParams)
    "densify_from": 500,
    "densify_until": 15000,
    "densify_every": 100,
    "percent_dense": 0.01,
    "min_opacity": 0.005,
    "big_world_from": 3000,  # size_threshold is active when iteration > opacity_reset_interval
    "big_world": 0.1,
    "reset_every": 3000,
    "reset_value": 0.01,
    # C1 selective densification (Eq. 6)
    "selective": True,  # False: 3DGS statistic E_g (Eq. 5) for ablation
    "grad_threshold": 0.0007,
    # C2 Gaussian pruning (Eq. 7/8)
    "prune": True,
    "top_k": 1,
    "view_num": 1,
    "prune_steps": [15000, 18000, 21000],  # official prune_way p4 (paper text: once after 15500)
    "weight_after_T": True,  # official forward.cu: weight = alpha * T after the T update
    # C3 sparse order increment (Eq. 9)
    "sparse_sh": True,
    "sh_alloc_step": 15500,
    "sh_interval": 1000,
    "sparse_count": 3,
    "sparse_rate": 0.2,
    "sparse_way": "pix_loss",  # or "random" (Table 6 ablation)
    "max_sh_degree": 3,
    "shN_lr": 0.0025 / 5,  # feature_lr / shs_lr_rate
}

REQUIRES = ("render.pixel_gradients", "training.target_image")


class EfficientGSDensity:
    def __init__(self, spec):
        unknown = set(spec.config) - set(DEFAULTS)
        if unknown:
            raise ValueError(f"Unknown EfficientGS density options: {sorted(unknown)}")
        self.cfg = {**DEFAULTS, **spec.config}
        if self.cfg["sparse_way"] not in ("pix_loss", "random"):
            raise ValueError("sparse_way must be 'pix_loss' or 'random'")
        if not 1 <= int(self.cfg["top_k"]) <= 10:
            raise ValueError("top_k must be in [1, 10] (official forward storage limit)")
        self.requires = set(REQUIRES)
        self.state = None
        self.cameras = None
        self.images = None
        self.renderer = None
        self.sparse_steps = core.sparse_steps(
            self.cfg["sh_alloc_step"], self.cfg["sh_interval"], self.cfg["sparse_count"]
        )

    # ------------------------------------------------------------------ runtime hooks
    def initialize(self, params, optimizers, scene_scale=1.0):
        if "sh_order" not in params:
            raise ValueError("EfficientGS needs the per-Gaussian 'sh_order' field")
        self.extent = float(scene_scale)
        optimizers["shN"].param_groups[0]["lr"] = float(self.cfg["shN_lr"])
        self._reset_stats(len(params["means"]), params["means"].device)
        self.state = {"events": []}

    def set_training_cameras(self, cameras):
        self.cameras = list(cameras)
        self.viewmats = torch.stack([c["viewmat"] for c in self.cameras]).flatten(1)
        self.images = [None] * len(self.cameras)
        self._cached = 0

    def before_backward(self, params, optimizers, step, outputs):
        if step < self.cfg["densify_until"]:
            outputs["info"]["means2d"].retain_grad()
            if self.cfg["selective"]:
                outputs["raw_rgb"].retain_grad()
                outputs["raw_alpha"].retain_grad()

    @torch.no_grad()
    def after_backward(self, params, optimizers, step, outputs):
        cfg = self.cfg
        self._cache_target(outputs)
        if step < cfg["densify_until"]:
            self._accumulate(outputs)
            if step > cfg["densify_from"] and step % cfg["densify_every"] == 0:
                self._densify(params, optimizers, step)
            if step % cfg["reset_every"] == 0:
                reset_opa(params, optimizers, {}, value=cfg["reset_value"])
        if cfg["prune"] and step in cfg["prune_steps"]:
            self._prune(params, optimizers, step)
        nxt = step + 1
        if cfg["sparse_sh"]:
            if nxt == cfg["sh_alloc_step"]:
                # official set_shs_memory: f_rest re-created as zeros with fresh Adam state
                params["shN"].data.zero_()
                optimizers["shN"].state.clear()
                params["sh_order"].data.zero_()
                self._log(step, "sh_alloc", n=len(params["means"]))
            if nxt in self.sparse_steps:
                self._sparse_increment(params, step)
        else:
            # Ablation without SOI: 3DGS global schedule from the allocation step on.
            if nxt >= cfg["sh_alloc_step"] and nxt % cfg["sh_interval"] == 0:
                params["sh_order"].data.add_(1).clamp_(max=cfg["max_sh_degree"])

    # ------------------------------------------------------------------ C1 statistics
    def _reset_stats(self, n, device):
        self.stats = {"grad": torch.zeros(n, device=device), "count": torch.zeros(n, device=device)}

    def _accumulate(self, outputs):
        info = outputs["info"]
        visible = (info["radii"][0] > 0).all(dim=-1)
        if self.cfg["selective"]:
            v_rgb, v_alpha = outputs["raw_rgb"].grad, outputs["raw_alpha"].grad
            if v_rgb is None or v_alpha is None:
                raise RuntimeError("raw render gradients missing; before_backward must retain them")
            norm, _ = kernels.gradnorm(info, outputs["colors"], v_rgb[0], v_alpha[0])
            stat = norm  # Eq. 6: sum over pixels of |per-pixel gradient| (NDC units)
        else:
            grad = info["means2d"].grad[0].clone()
            grad[:, 0] *= info["width"] / 2.0
            grad[:, 1] *= info["height"] / 2.0
            stat = grad.norm(dim=-1)  # Eq. 5 (3DGS)
        self.stats["grad"][visible] += stat[visible]
        self.stats["count"][visible] += 1

    def _densify(self, params, optimizers, step):
        cfg = self.cfg
        n0 = len(params["means"])
        max_scale = params["scales"].exp().max(dim=-1).values
        _, clone, split_mask = core.densify_select(
            self.stats["grad"], self.stats["count"], max_scale, cfg["grad_threshold"],
            cfg["percent_dense"], self.extent,
        )
        duplicate(params, optimizers, {}, clone)
        n_clone = int(clone.sum())
        split_mask = torch.cat(
            [split_mask, torch.zeros(n_clone, dtype=torch.bool, device=split_mask.device)]
        )
        n_split = int(split_mask.sum())
        split(params, optimizers, {}, split_mask)
        prune = params["opacities"].sigmoid() < cfg["min_opacity"]
        if step > cfg["big_world_from"]:
            # The 3DGS screen-size test (max_radii2D > 20) is inert: densification_postfix
            # zeroes max_radii2D before the prune test (official gaussian_model.py).
            prune |= params["scales"].exp().max(dim=-1).values > cfg["big_world"] * self.extent
        remove(params, optimizers, {}, prune)
        self._reset_stats(len(params["means"]), params["means"].device)
        if step % 1000 == 0:
            self._log(step, "densify", n_before=n0, clone=n_clone, split=n_split,
                      pruned=int(prune.sum()), n_after=len(params["means"]))

    # ------------------------------------------------------------------ helpers
    def _render_views(self, params):
        if self.cameras is None:
            raise RuntimeError("EfficientGS needs set_training_cameras()")
        if self.renderer is None:
            from .renderer import EfficientGSRenderer

            self.renderer = EfficientGSRenderer(None)
        for k, cam in enumerate(self.cameras):
            yield k, cam, self.renderer.render(params, cam, sh_degree=self.cfg["max_sh_degree"])

    def _cache_target(self, outputs):
        """Keep training-view images (uint8, lossless for 8-bit sources) for Eq. 9."""
        if not self.cfg["sparse_sh"] or self.cfg["sparse_way"] != "pix_loss":
            return
        if self.cameras is None or self._cached == len(self.cameras):
            return
        if outputs.get("target_image") is None:
            return
        d = (self.viewmats - outputs["viewmat"].flatten()[None]).abs().sum(dim=1)
        k = int(d.argmin())
        if float(d[k]) > 1e-5:
            raise RuntimeError("Rendered frame is not one of the training cameras")
        if self.images[k] is None:
            self.images[k] = (outputs["target_image"].clamp(0, 1) * 255).round().to(torch.uint8)
            self._cached += 1

    # ------------------------------------------------------------------ C2 pruning
    def _prune(self, params, optimizers, step):
        n0 = len(params["means"])
        count = torch.zeros(n0, dtype=torch.int32, device=params["means"].device)
        for _, _, out in self._render_views(params):
            count += kernels.blend_stats(
                out["info"], self.cfg["top_k"], self.cfg["weight_after_T"]
            )["dominant"]
        mask = count < self.cfg["view_num"]
        remove(params, optimizers, {}, mask)
        self._reset_stats(len(params["means"]), params["means"].device)
        self._log(step, "topk_prune", n_before=n0, pruned=int(mask.sum()),
                  n_after=len(params["means"]))
        torch.cuda.empty_cache()

    # ------------------------------------------------------------------ C3 sparse SH
    def _sparse_increment(self, params, step):
        cfg = self.cfg
        order = params["sh_order"].data[:, 0]
        n = len(order)
        if cfg["sparse_way"] == "random":
            mask = torch.zeros(n, dtype=torch.bool, device=order.device)
            mask[torch.randperm(n, device=order.device)[: int(n * cfg["sparse_rate"])]] = True
        else:
            missing = [k for k, img in enumerate(self.images) if img is None]
            if missing:
                raise RuntimeError(
                    f"{len(missing)} training views were never rendered before step {step}; "
                    "their ground-truth images are unavailable for Eq. 9"
                )
            lw = torch.zeros(n, device=order.device)
            for k, _, out in self._render_views(params):
                gt = self.images[k].float() / 255.0
                pix_loss = (out["rgb"] - gt).abs()  # official: |render - gt|, unclamped
                lw += kernels.blend_stats(
                    out["info"], 1, cfg["weight_after_T"], pix_loss=pix_loss
                )["loss_weight"]
            mask = core.sparse_increment_mask(lw, cfg["sparse_rate"])
        params["sh_order"].data[:, 0] = core.increment_orders(order, mask, cfg["max_sh_degree"])
        hist = torch.bincount(params["sh_order"].data[:, 0].long(), minlength=4).tolist()
        self._log(step + 1, "sparse_sh", selected=int(mask.sum()), n=n,
                  deg0=hist[0], deg1=hist[1], deg2=hist[2], deg3=hist[3])
        torch.cuda.empty_cache()

    def _log(self, step, kind, **values):
        row = {"step": int(step), "event": kind, **{k: int(v) for k, v in values.items()}}
        self.state["events"].append(row)
        print({"efficientgs": row}, flush=True)


def create_density(spec):
    return EfficientGSDensity(spec)
