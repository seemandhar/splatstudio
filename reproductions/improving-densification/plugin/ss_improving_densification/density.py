"""ImprovedGS density controller (EAS + LAS + RAP + GC + MU + hyper-parameter changes).

Mirrors, per iteration and in the official order (train.py: optimizer step -> densification
stage -> pruning stage):
- scene/methods/densification_stage.py  (stats, EAS scores, budget, densify dispatch)
- scene/gaussian_model_densification.py (densify_and_prune_improved, long_axis_split, only_prune)
- scene/methods/pruning_methods.py       (early prune, RAP reset + delayed percentile prune)
- scene/training_loop.py                 (MU-gated optimizer steps, lr schedule)
"""

import sys
import time
from pathlib import Path

import torch
from gsplat.strategy.ops import _update_param_with_optimizer, remove, reset_opa, split

from . import core
from .optim import Gate, gate_optimizers
from .raster import blend_weights

REQUIRES = {"gradient.screen.inria_abs"}
DEFAULTS = {
    # Densification window / cadence (arguments/__init__.py OptimizationBaseParams)
    "densify_from_iter": 500,
    "densify_until_iter": 15000,
    "densification_interval": 100,
    "densify_grad_threshold": 0.0003,  # abs-gradient threshold (App. A.2)
    "min_opacity": 0.005,
    "opacity_reset_interval": 3000,
    # ImprovedGSParams
    "edge_sample_cams": 10,
    "eas_full_every": 3000,  # it % 3000 == 400 and it < 9000 -> all views
    "eas_full_offset": 400,
    "eas_full_until": 9000,
    "reset_opacity": 0.05,  # improvedgs_reset_min_opacity
    "split_distance": 0.45,
    "opacity_reduction": 0.6,
    "budget": None,  # None: per-scene table (paper Sec. 4.2) keyed by the --data basename
    "budget_warmup_until_offset": 500,
    "rap_prune_ratio": 0.2,
    "rap_prune_offset": 300,
    "rap_rounds": 2,
    "early_prune_iter": 300,
    "early_prune_opacity": 0.02,
    "mu_start_iter": 20000,
    "mu_interval": 5,
    "mu_second_start_iter": 24000,
    "mu_second_interval": 20,
    # Optimisation hyper-parameters the runtime does not set itself
    "position_lr_init": 4e-5,
    "position_lr_final": 2e-6,
    "position_lr_max_steps": 30000,
    "opacity_lr": 0.025,
    "total_steps": 30000,  # official: no parameter update at iteration == iterations
    # Component switches (official use_eas / use_las / use_rap / use_mu; GC = budget curve)
    "use_eas": True,
    "use_las": True,
    "use_rap": True,
    "use_mu": True,
    "use_gc": True,
    "seed": 0,
    # Training-image source for the edge maps; default: ss-train's --data / --downscale.
    "data": None,
    "downscale": None,
}


def _argv_value(flag):
    argv = sys.argv
    for i, arg in enumerate(argv):
        if arg == flag and i + 1 < len(argv):
            return argv[i + 1]
        if arg.startswith(flag + "="):
            return arg.split("=", 1)[1]
    return None


class ImprovedGSDensity:
    requires = REQUIRES

    def __init__(self, spec):
        unknown = set(spec.config) - set(DEFAULTS)
        if unknown:
            raise ValueError(f"Unknown improvedgs-density options: {sorted(unknown)}")
        self.c = {**DEFAULTS, **spec.config}
        if not 0.0 < float(self.c["split_distance"]) <= 0.5:
            raise ValueError("split_distance must be in (0, 0.5] (paper: d <= 0.5 L0)")
        self.requires = set(REQUIRES)
        c = self.c
        self.gate = Gate(lambda s: core.mu_update(
            s, c["total_steps"], c["mu_start_iter"], c["mu_interval"],
            c["mu_second_start_iter"], c["mu_second_interval"], bool(c["use_mu"])))
        self.cameras = None
        self.edges = None  # list of [H,W] float32 edge maps (CPU, pinned when CUDA)
        self.pool = None
        self.n_max = None if c["budget"] is None else int(c["budget"])
        self.stats = None
        self.rap_triggers = set()
        self.rap_resets = 0
        self.extent = 1.0
        self.history = []
        self.state = {"history": self.history}
        self.gen = None

    # ---- runtime hooks ------------------------------------------------------------------
    def initialize(self, params, optimizers, scene_scale=1.0):
        for name in ("means", "scales", "quats", "opacities", "sh0", "shN"):
            if name not in params or name not in optimizers:
                raise ValueError(f"improvedgs-density requires the {name} field")
        self.extent = float(scene_scale)
        gate_optimizers(optimizers, self.gate)
        optimizers["opacities"].param_groups[0]["lr"] = float(self.c["opacity_lr"])
        optimizers["means"].param_groups[0]["lr"] = self.xyz_lr(1)
        self._reset_stats(len(params["means"]), params["means"].device)
        self.gen = torch.Generator(device=params["means"].device)
        self.gen.manual_seed(int(self.c["seed"]))
        if self.n_max is None:
            root = self.c["data"] or _argv_value("--data")
            scene = Path(root).name if root else None
            if scene in core.SCENE_BUDGETS:
                self.n_max = core.SCENE_BUDGETS[scene]
        self.state.update({"extent": self.extent, "budget": self.n_max})

    def set_training_cameras(self, cameras):
        """Runtime hook: reload the training images (not exposed to components) for the edge
        maps (mvgs precedent)."""
        if not self.c["use_eas"]:
            self.cameras = [dict(c) for c in cameras]
            return
        root = self.c["data"] or _argv_value("--data")
        if root is None:
            raise RuntimeError("improvedgs-density needs the training images for EAS: set "
                               "density config 'data' or run through ss-train --data")
        downscale = int(self.c["downscale"] or _argv_value("--downscale") or 1)
        from splatstudio.data import load_dataset

        t0 = time.perf_counter()
        dataset = load_dataset(Path(root), downscale)
        names = dataset["splits"]["train"]
        if len(names) != len(cameras):
            raise RuntimeError("Reloaded training split does not match the runtime cameras")
        images = []
        for name, cam in zip(names, cameras):
            frame = dataset["frames"][name]
            same = (frame["width"] == cam["width"] and frame["height"] == cam["height"]
                    and torch.allclose(frame["K"].to(cam["K"].device), cam["K"])
                    and torch.allclose(frame["viewmat"].to(cam["K"].device), cam["viewmat"]))
            if not same:
                raise RuntimeError(f"Reloaded training view {name} differs from runtime camera")
            images.append(frame["image"])
        del dataset
        self.set_training_frames(cameras, images)
        self.state["edge_load_seconds"] = time.perf_counter() - t0

    def set_training_frames(self, cameras, images):
        """cameras: native camera dicts; images: HWC float [0,1] aligned with cameras."""
        if len(cameras) != len(images) or not cameras:
            raise ValueError("cameras and images must be non-empty and aligned")
        self.cameras = [dict(c) for c in cameras]
        pin = torch.cuda.is_available()
        self.edges = []
        for im in images:
            e = core.edge_map(im.float()).cpu()
            self.edges.append(e.pin_memory() if pin else e)
        self.pool = core.EdgeViewPool(len(cameras), int(self.c["edge_sample_cams"]),
                                      seed=int(self.c["seed"]))

    def before_backward(self, params, optimizers, step, outputs):
        self.gate.step = int(step)
        outputs["info"]["means2d"].retain_grad()

    @torch.no_grad()
    def after_backward(self, params, optimizers, step, outputs):
        c = self.c
        if c["densify_from_iter"] < step < c["densify_until_iter"]:
            self._accumulate(outputs["info"])
            if step % c["densification_interval"] == 0:
                self._densify(params, optimizers, step)
        if c["use_rap"]:
            self._rap(params, optimizers, step)
        optimizers["means"].param_groups[0]["lr"] = self.xyz_lr(step + 1)

    # ---- pieces -------------------------------------------------------------------------
    def xyz_lr(self, step):
        c = self.c
        return core.expon_lr(step, c["position_lr_init"] * self.extent,
                             c["position_lr_final"] * self.extent, c["position_lr_max_steps"])

    def _reset_stats(self, n, device):
        self.stats = {"accum_abs": torch.zeros(n, device=device),
                      "denom": torch.zeros(n, device=device)}

    def _accumulate(self, info):
        """add_densification_stats_abs: ||(sum|dx|, sum|dy|)|| for radii > 0."""
        grad = info["means2d"].grad
        if grad is None:
            return
        vis = info["radii"] > 0
        self.stats["accum_abs"][vis] += torch.linalg.vector_norm(grad[vis, 2:4], dim=-1)
        self.stats["denom"][vis] += 1

    def current_budget(self, step):
        if self.n_max is None:
            raise RuntimeError("No Gaussian budget: unknown scene; set density config 'budget'")
        c = self.c
        if not c["use_gc"]:
            return self.n_max
        return core.budget(step, self.n_max, c["densify_from_iter"], c["densify_until_iter"],
                           c["budget_warmup_until_offset"])

    def eas_views(self, step):
        c = self.c
        n = len(self.cameras)
        k = int(c["edge_sample_cams"])
        if k == -1 or (step % c["eas_full_every"] == c["eas_full_offset"]
                       and step < c["eas_full_until"] and k < n):
            return self.pool.all_views()
        return self.pool.sample()

    def _eas_scores(self, params, step):
        """_compute_improvedgs_scores (Eq. 5-6 + per-view min-max normalisation)."""
        if not self.c["use_eas"]:
            return None
        if self.edges is None:
            raise RuntimeError("EAS is enabled but no training images were provided")
        views = self.eas_views(step)
        dev = params["means"].device
        score = torch.zeros(len(params["means"]), device=dev)
        any_vis = torch.zeros(len(params["means"]), dtype=torch.bool, device=dev)
        for i in views:
            accum, radii = blend_weights(params, self.cameras[i],
                                         self.edges[i].to(dev, non_blocking=True))
            if accum.numel() == 0:
                continue
            vis = radii > 0
            score[vis] += core.minmax01(accum)[vis] / max(len(views), 1)
            any_vis |= vis
        self.state["eas_views_last"] = len(views)
        return score if bool(any_vis.any()) else None

    def _densify(self, params, optimizers, step):
        """densify_and_prune_improved."""
        c = self.c
        scores = self._eas_scores(params, step)
        g = self.stats["accum_abs"] / self.stats["denom"]
        g[g.isnan()] = 0.0
        min_grad = float(c["densify_grad_threshold"])
        late = max(int(c["densify_until_iter"]) - 100, int(c["densify_from_iter"]))
        n_budget = self.current_budget(step)
        if scores is None or step > late:
            scores = g.clone()
            if len(params["means"]) < n_budget and step > late:
                min_grad = min_grad / 1.5
        qual = g >= min_grad
        n = len(params["means"])
        current = min(n_budget, int(qual.sum()) + n)
        k = current - n
        n_split = 0
        if k > 0:
            if c["use_las"]:
                n_split = self._long_axis_split(params, optimizers, scores, k, qual)
            else:
                n_split = self._split_3dgs(params, optimizers, qual)
        n_pruned = 0
        if step < late:
            mask = params["opacities"].sigmoid() < float(c["min_opacity"])
            n_pruned = int(mask.sum())
            if n_pruned:
                remove(params, optimizers, self.stats, mask)
        self.history.append({"step": step, "budget": n_budget, "candidates": int(qual.sum()),
                             "split": n_split, "pruned": n_pruned,
                             "gaussians": len(params["means"])})

    def _long_axis_split(self, params, optimizers, scores, k, qual):
        """long_axis_split: multinomial(score * qualifier, k, no replacement) parents."""
        w = scores.detach().float().clamp_min(0)
        w[~qual] = 0
        positive = int((w > 0).sum())
        if positive == 0:
            return 0
        k = min(int(k), positive)
        idx = torch.multinomial(w, k, replacement=False, generator=self.gen)
        sel = torch.zeros(len(w), dtype=torch.bool, device=w.device)
        sel[idx] = True
        rest = ~sel
        means, log_scales, opac = core.long_axis_children(
            params["means"][sel], params["scales"][sel], params["quats"][sel],
            params["opacities"][sel], self.c["split_distance"], self.c["opacity_reduction"])
        children = {"means": means, "scales": log_scales, "opacities": opac}

        def param_fn(name, p):
            new = children[name] if name in children else p[sel].repeat(2, *[1] * (p.dim() - 1))
            return torch.nn.Parameter(torch.cat([p[rest], new]), requires_grad=p.requires_grad)

        def optimizer_fn(key, v):
            return torch.cat([v[rest], torch.zeros((2 * k, *v.shape[1:]), device=v.device)])

        _update_param_with_optimizer(param_fn, optimizer_fn, params, optimizers)
        # densification_postfix(reset_params=True): accumulators restart from zero.
        self._reset_stats(len(params["means"]), params["means"].device)
        return k

    def _split_3dgs(self, params, optimizers, qual):
        """Ablation 'w/o LAS' (official densify_and_split: grad >= tau and
        max scale > 0.01 extent, sampled children, scale / 1.6, no budget)."""
        big = params["scales"].exp().max(dim=1).values > 0.01 * self.extent
        sel = qual & big
        n = int(sel.sum())
        if n:
            split(params, optimizers, self.stats, sel)
            self._reset_stats(len(params["means"]), params["means"].device)
        return n

    def _rap(self, params, optimizers, step):
        """run_3dgs_pruning_method (RAP branch)."""
        c = self.c
        if step == int(c["early_prune_iter"]):
            self._prune_below(params, optimizers, float(c["early_prune_opacity"]))
        if step in self.rap_triggers:
            opac = params["opacities"].sigmoid().detach()
            thr = torch.quantile(opac, float(c["rap_prune_ratio"]))
            self._prune_below(params, optimizers, thr)
            self.rap_triggers.discard(step)
            self.history.append({"step": step, "rap_prune_threshold": float(thr),
                                 "gaussians": len(params["means"])})
        if (c["densify_from_iter"] < step < c["densify_until_iter"]
                and step % c["opacity_reset_interval"] == 0):
            reset_opa(params, optimizers, {}, value=float(c["reset_opacity"]))
            if float(c["rap_prune_ratio"]) > 0 and self.rap_resets < int(c["rap_rounds"]):
                self.rap_triggers.add(step + int(c["rap_prune_offset"]))
            self.rap_resets += 1
            self.history.append({"step": step, "opacity_reset": float(c["reset_opacity"])})

    def _prune_below(self, params, optimizers, threshold):
        mask = params["opacities"].sigmoid() < threshold
        if bool(mask.any()) and not bool(mask.all()):
            remove(params, optimizers, self.stats, mask)


def create_density(spec):
    return ImprovedGSDensity(spec)

