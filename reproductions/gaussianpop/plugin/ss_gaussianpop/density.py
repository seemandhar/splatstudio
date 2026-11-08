"""GaussianPOP post-training density: 3DGS pretraining, iterative error-based prune, fine-tune.

One native run of `prune_step + finetune` steps (35000 for the paper's post-training setting):

* steps 1..base_steps: the unchanged SplatStudio 3DGS density (gsplat DefaultStrategy with the
  opacity-reset fix); the means LR follows a `--steps base_steps` run (the runtime would decay
  it over the full run otherwise), so the model at base_steps is the 30k 3DGS model.
* end of step `prune_step`: `cycles` cycles of (Δ_SE over all training views, Alg. 1 / Eq. 4 →
  remove the lowest-error Gaussians down to the cycle's keep count) (Sec. 3.4, 4.3).
* fine-tune: plain 3DGS continuation, no densification (gsplat refine stops at 15k), means LR
  held at its final 3DGS value, fresh Adam state (pre-trained model reloaded).
"""

import json
import time
from dataclasses import replace

import torch
from gsplat.strategy import ops
from splatstudio.backends.gsplat import Density

from . import kernels, schedule

DEFAULTS = {
    "base": {"absgrad": False, "grow_grad2d": 0.0002},
    "base_steps": 30000,
    "prune_step": 30000,
    "prune_ratio": 0.8,
    "cycles": 8,
    "split": "compound",
    "n_max": 64,
    "eps": 1e-9,
    "background": [1.0, 1.0, 1.0],
    "lr_final_ratio": 0.01,
    "reset_optimizer": True,
}


class POPDensity:
    def __init__(self, spec):
        unknown = set(spec.config) - set(DEFAULTS)
        if unknown:
            raise ValueError(f"Unknown pop-density options: {sorted(unknown)}")
        self.cfg = {**DEFAULTS, **spec.config}
        c = self.cfg
        if c["prune_step"] < c["base_steps"]:
            raise ValueError("post-training pruning must start after pretraining")
        schedule.keep_counts(100, c["prune_ratio"], c["cycles"], c["split"])  # validates
        self.base = Density(replace(spec, config=dict(c["base"])))
        self.requires = self.base.requires
        self.cameras = None
        self.lr0 = None
        self.history = []
        self.quantify_calls = 0

    @property
    def state(self):
        return {"strategy": self.base.state, "pop_cycles": self.history}

    def set_training_cameras(self, cameras):
        self.cameras = cameras

    def initialize(self, params, optimizers, scene_scale=1.0):
        self.base.initialize(params, optimizers, scene_scale)
        # The runtime has already multiplied the means LR by the scene scale.
        self.lr0 = optimizers["means"].param_groups[0]["lr"]

    def before_backward(self, params, optimizers, step, outputs):
        self.base.before_backward(params, optimizers, step, outputs)

    def after_backward(self, params, optimizers, step, outputs):
        self.base.after_backward(params, optimizers, step, outputs)
        if step == self.cfg["prune_step"]:
            self.prune(params, optimizers, step)
        optimizers["means"].param_groups[0]["lr"] = schedule.pretrain_means_lr(
            self.lr0, step, self.cfg["base_steps"], self.cfg["lr_final_ratio"]
        )

    def quantify(self, params):
        self.quantify_calls += 1
        c = self.cfg
        return kernels.scene_error(
            params, self.cameras, background=tuple(c["background"]), n_max=c["n_max"],
            eps=c["eps"],
        )

    @torch.no_grad()
    def prune(self, params, optimizers, step):
        if not self.cameras:
            raise RuntimeError("pop-density needs set_training_cameras() before pruning")
        c = self.cfg
        n0 = len(params["means"])
        for cycle, keep in enumerate(
            schedule.keep_counts(n0, c["prune_ratio"], c["cycles"], c["split"]), 1
        ):
            start = time.perf_counter()
            scores = self.quantify(params)
            before = len(scores)
            mask = schedule.lowest_mask(scores, before - keep)
            ops.remove(params, optimizers, self.base.state, mask)
            record = {
                "pop_prune_step": step,
                "cycle": cycle,
                "before": before,
                "after": len(params["means"]),
                "zero_error": int((scores == 0).sum()),
                "max_removed_error": float(scores[mask].max()) if mask.any() else 0.0,
                "total_removed_error": float(scores[mask].sum()),
                "seconds": round(time.perf_counter() - start, 2),
            }
            self.history.append(record)
            print(json.dumps(record), flush=True)
            del scores, mask
        if c["reset_optimizer"]:
            for opt in optimizers.values():
                opt.state.clear()
        for name, p in params.items():
            if not (p.is_leaf and p.requires_grad):
                raise RuntimeError(f"field {name} is no longer a trainable leaf after pruning")
        torch.cuda.empty_cache()


def create_density(spec):
    return POPDensity(spec)
