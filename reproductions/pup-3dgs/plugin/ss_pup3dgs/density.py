"""PUP 3D-GS density component: 3DGS pretraining, then multi-round Fisher prune-refine.

One native run of `prune_steps[-1] + finetune` steps (40000 for the paper) expresses the
official pipeline (`scripts/full_pruning_pipeline.sh` + `prune_finetune.py`):

* steps 1..base_steps: the unchanged SplatStudio 3DGS density (gsplat DefaultStrategy with the
  opacity-reset fix); the means LR follows the schedule of a `--steps base_steps` run.
* after step prune_steps[r]: per-Gaussian Fisher over all training views (paper Eq. U_i),
  prune the prune_ratios[r] lowest-scoring Gaussians, fresh Adam (official reloads the .ply).
* refinement: official fine-tune LRs (xyz exp decay with max_steps 35000 from lr_iter 30001,
  ExponentialLR gamma 0.95 every 400 iterations), no densification.
"""

import json
import time
from dataclasses import replace

import torch
from gsplat.strategy import ops
from splatstudio.backends.gsplat import Density

from . import fisher, schedule

DEFAULTS = {
    "base": {"absgrad": False, "grow_grad2d": 0.0002},
    "base_steps": 30000,
    "prune_steps": [30000, 35000],
    "prune_ratios": [0.8, 0.5],
    "fisher_resolution": 4,
    "finetune_lr_start": 30000,
    "finetune_lr_max_steps": 35000,
    "lr_final_ratio": 0.01,
    "exp_gamma": 0.95,
    "exp_every": 400,
    "reset_optimizer": True,
    "background": [1.0, 1.0, 1.0],
}


class PUPDensity:
    def __init__(self, spec):
        unknown = set(spec.config) - set(DEFAULTS)
        if unknown:
            raise ValueError(f"Unknown pup-density options: {sorted(unknown)}")
        self.cfg = {**DEFAULTS, **spec.config}
        steps, ratios = list(self.cfg["prune_steps"]), list(self.cfg["prune_ratios"])
        if len(steps) != len(ratios) or not steps or sorted(steps) != steps:
            raise ValueError("prune_steps must be increasing and match prune_ratios")
        if steps[0] < self.cfg["base_steps"] or not all(0 < r < 1 for r in ratios):
            raise ValueError("pruning must start after pretraining with ratios in (0, 1)")
        self.base = Density(replace(spec, config=dict(self.cfg["base"])))
        self.requires = self.base.requires
        self.cameras = None
        self.lr0 = None
        self.round_start = None
        self.history = []

    @property
    def state(self):
        return {"strategy": self.base.state, "pup_rounds": self.history}

    def set_training_cameras(self, cameras):
        self.cameras = cameras

    def initialize(self, params, optimizers, scene_scale=1.0):
        self.base.initialize(params, optimizers, scene_scale)
        # Runtime has already multiplied the means LR by the scene scale (spatial_lr_scale).
        self.lr0 = {name: opt.param_groups[0]["lr"] for name, opt in optimizers.items()}

    def before_backward(self, params, optimizers, step, outputs):
        self.base.before_backward(params, optimizers, step, outputs)

    def after_backward(self, params, optimizers, step, outputs):
        self.base.after_backward(params, optimizers, step, outputs)
        if step in self.cfg["prune_steps"]:
            self.prune(params, optimizers, step)
        self.set_learning_rates(optimizers, step)

    def set_learning_rates(self, optimizers, step):
        """Learning rates for step + 1 (the runtime set its own just before this call)."""
        c = self.cfg
        if self.round_start is None:
            optimizers["means"].param_groups[0]["lr"] = schedule.pretrain_means_lr(
                self.lr0["means"], step, c["base_steps"], c["lr_final_ratio"]
            )
            return
        lr_iter = schedule.finetune_lr_iter(step + 1, self.round_start, c["finetune_lr_start"])
        for name, opt in optimizers.items():
            if name == "means":
                lr = schedule.finetune_means_lr(
                    self.lr0["means"], lr_iter, c["finetune_lr_max_steps"], c["lr_final_ratio"],
                    c["exp_gamma"], c["exp_every"],
                )
            else:
                lr = schedule.finetune_other_lr(
                    self.lr0[name], lr_iter, c["finetune_lr_start"], c["exp_gamma"], c["exp_every"]
                )
            opt.param_groups[0]["lr"] = lr

    @torch.no_grad()
    def prune(self, params, optimizers, step):
        if not self.cameras:
            raise RuntimeError("pup-density needs set_training_cameras() before pruning")
        ratio = self.cfg["prune_ratios"][self.cfg["prune_steps"].index(step)]
        start = time.perf_counter()
        sym = fisher.scene_fisher(
            params, self.cameras, self.cfg["fisher_resolution"], tuple(self.cfg["background"])
        )
        scores = fisher.log_det_scores(sym)
        del sym
        mask = fisher.prune_mask(scores, ratio)
        before = len(params["means"])
        ops.remove(params, optimizers, self.base.state, mask)
        if self.cfg["reset_optimizer"]:
            for opt in optimizers.values():
                opt.state.clear()
        torch.cuda.empty_cache()
        self.round_start = step
        record = {
            "pup_prune_step": step,
            "ratio": ratio,
            "before": before,
            "after": len(params["means"]),
            "zero_fisher": int(torch.isneginf(scores).sum()),
            "seconds": round(time.perf_counter() - start, 2),
        }
        self.history.append(record)
        print(json.dumps(record), flush=True)


def create_density(spec):
    return PUPDensity(spec)
