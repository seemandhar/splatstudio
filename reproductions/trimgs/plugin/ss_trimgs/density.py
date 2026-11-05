"""Trim3DGS density controller: 30k flat-3DGS pre-training, then the 7k Trim fine-tune.

Steps 1..pretrain_steps: SplatStudio's gsplat Density (DefaultStrategy + 3DGS opacity reset),
i.e. the `baseline3dgs` recipe, on flat Gaussians (official train.py with FlattenGaussianModel).
After pretrain_steps (official tune.py, scripts/run_Mipnerf360.py command line):
  * fresh Adam state, constant means LR = final 3DGS LR (position_lr_init 1.6e-6);
  * per step: max 2D radius (official 3-sigma radius) of visible Gaussians;
  * at t % densify_interval == 0, 0 < t < finetune_steps (t = step - pretrain_steps):
      scale-driven split (Sec. 4.2) -> opacity prune -> contribution trimming (Sec. 4.1)
      -> radii reset.
"""

import random
from dataclasses import replace

import torch
from splatstudio.backends.gsplat import Density as BaseDensity

from . import core, kernels

DEFAULTS = {
    "pretrain_steps": 30000,
    "finetune_steps": 7000,
    "final_lr_ratio": 0.01,
    "reset_optimizer": True,
    "flat_eps": 1e-8,
    # scale-driven densification (densify_and_scale_split, split="scale")
    "densify_interval": 1000,
    "percent_dense": 0.01,
    "scale_factor": 1.0,
    "max_screen_size": 100,
    "split_n": 3,
    "min_opacity": 0.005,
    "cull_expansion": 2.0,
    "scale_split": True,
    # contribution-based trimming
    "trim": True,
    "trim_interval": 1000,
    "trim_from": 0,
    "prune_ratio": 0.1,
    "top_k": 5,
    "gamma": 0.5,
    "shuffle_views_seed": 0,
}


class TrimDensity:
    def __init__(self, spec):
        config = dict(spec.config)
        self.c = {k: config.pop(k, v) for k, v in DEFAULTS.items()}
        self.base = BaseDensity(replace(spec, config=config))
        self.requires = set(self.base.requires)
        self.state = None
        self.cameras = None
        self.lr0 = None
        self.extent = None
        self.max_radii = None
        self.events = []

    # ------------------------------------------------------------------ hooks
    def initialize(self, params, optimizers, scene_scale=1.0):
        self.base.initialize(params, optimizers, scene_scale)
        self.state = {"base": self.base.state, "events": self.events}
        self.extent = float(scene_scale)
        # runtime already multiplied the means LR by scene_scale (spatial_lr_scale)
        self.lr0 = optimizers["means"].param_groups[0]["lr"]
        core.pin_flat_axis(params, self.c["flat_eps"])

    def set_training_cameras(self, cameras):
        cams = list(cameras)
        # official Scene shuffles the train cameras once (random.shuffle, seed 0); the order
        # matters for the record-insertion "top-K" of the trimming pass.
        if self.c["shuffle_views_seed"] is not None:
            random.Random(self.c["shuffle_views_seed"]).shuffle(cams)
        self.cameras = cams

    def before_backward(self, params, optimizers, step, outputs):
        if step <= self.c["pretrain_steps"]:
            self.base.before_backward(params, optimizers, step, outputs)

    @torch.no_grad()
    def after_backward(self, params, optimizers, step, outputs):
        c = self.c
        P = c["pretrain_steps"]
        if step <= P:
            n0 = len(params["means"])
            self.base.after_backward(params, optimizers, step, outputs)
            if len(params["means"]) != n0 or step % 100 == 0:
                core.pin_flat_axis(params, c["flat_eps"])
            if step == P:
                self._start_finetune(params, optimizers)
        else:
            t = step - P
            self._track_radii(params, outputs)
            if t % c["densify_interval"] == 0 and 0 < t < c["finetune_steps"]:
                self._densify(params, optimizers, t)
        optimizers["means"].param_groups[0]["lr"] = core.pretrain_means_lr(
            self.lr0, step, P, c["final_lr_ratio"]
        )

    # ------------------------------------------------------------------ fine-tune
    def _start_finetune(self, params, optimizers):
        if self.c["reset_optimizer"]:
            for opt in optimizers.values():
                opt.state.clear()  # tune.py builds a new Adam from the loaded ply
        core.pin_flat_axis(params, self.c["flat_eps"])
        self.max_radii = torch.zeros(len(params["means"]), device=params["means"].device)
        self.events.append({"step": self.c["pretrain_steps"], "event": "finetune_start",
                            "gaussians": len(params["means"])})

    def _track_radii(self, params, outputs):
        n = len(params["means"])
        if self.max_radii is None or len(self.max_radii) != n:
            self.max_radii = torch.zeros(n, device=params["means"].device)
        info = outputs["info"]
        radii = info["radii"][0]
        visible = (radii > 0).all(dim=-1) if radii.dim() == 2 else radii > 0
        r = core.official_radii(info["conics"][0].detach())
        self.max_radii[visible] = torch.maximum(self.max_radii[visible], r[visible])

    def _densify(self, params, optimizers, t):
        c = self.c
        eps = c["flat_eps"]
        event = {"step": c["pretrain_steps"] + t, "before": len(params["means"])}
        if c["scale_split"]:
            centers = torch.stack([torch.linalg.inv(cam["viewmat"])[:3, 3] for cam in self.cameras])
            box = core.culling_mask(params["means"], centers.to(params["means"]),
                                    c["cull_expansion"])
            mask = core.scale_split_mask(
                core.flat_scales(params["scales"], eps), self.max_radii, box, c["percent_dense"],
                self.extent, c["scale_factor"], c["max_screen_size"],
            )
            event["split"] = core.split_n(params, optimizers, mask, c["split_n"], eps)
            low = params["opacities"].sigmoid() < c["min_opacity"]
            event["opacity_pruned"] = core.remove(params, optimizers, low) if low.any() else 0
        if c["trim"] and t > c["trim_from"] and t % c["trim_interval"] == 0:
            contribution = self.contribution(params)
            mask = core.trim_mask(contribution, c["prune_ratio"])
            event["trimmed"] = core.remove(params, optimizers, mask)
        core.pin_flat_axis(params, eps)
        self.max_radii = torch.zeros(len(params["means"]), device=params["means"].device)
        event["after"] = len(params["means"])
        self.events.append(event)
        torch.cuda.empty_cache()

    def contribution(self, params):
        """Eq. 6 over all training views with the official record-insertion list."""
        top = [None] * self.c["top_k"]
        for cam in self.cameras:
            r = kernels.contribution(params, cam, self.c["flat_eps"], self.c["gamma"])
            top = core.official_topk_insert(top, core.normalized_contribution(r["sum"],
                                                                              r["covered"]))
        return torch.stack(top, dim=-1).mean(-1)


def create_density(spec):
    return TrimDensity(spec)
