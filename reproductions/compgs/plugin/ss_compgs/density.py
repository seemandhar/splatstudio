"""CompGS density component: 3DGS density + opacity-regularised pruning + k-means VQ schedule.

Runtime order per step: render(step) -> backward -> optimizer step -> after_backward(step). The
official loop quantises at the top of iteration `it` on the post-step parameters, so
`after_backward(step)` prepares the codebooks for render(it = step + 1) (train_kmeans.py:129-146):
* it % freq == 1 (vq_start < it <= total, freq 100; 5000 for it > total - 5000): k-means
  assignment (`kmeans_iters` Lloyd iterations + final assignment); otherwise the centroids are
  re-averaged from the current parameters with the cached assignments.
* opacity pruning (sigma < 0.005) every 1000 steps for 15000 < step <= 20000 runs first
  (train_kmeans.py:203-208), so a prune at step s is followed by an assignment for s + 1.
* at total_steps: the decoded model C[idx] (centroids used for render(total_steps), saved as-is
  by the official code) is copied into sh0/shN/scales/quats, the VQ state is deactivated and the
  paper-formula size is recorded, so validation and ss-eval measure the decoded model.
Until 15000 (and in fact for the whole run) the unchanged SplatStudio 3DGS density runs.
"""

import json
import time
from dataclasses import replace

import torch
from gsplat.strategy import ops
from splatstudio.backends.gsplat import Density

from . import codec, schedule
from . import vq as vq_mod

DEFAULTS = {
    "base": {"absgrad": False, "grow_grad2d": 0.0002},
    "total_steps": 30000,
    "vq_start": 20000,
    "assign_freq": 100,
    "final_freq": 5000,
    "freeze_window": 5000,
    "kmeans_iters": 5,
    "codebooks": {"dc": 4096, "sh": 4096, "scale": 32768, "rot": 32768},
    "chunk": 10000,
    "opacity_prune": True,
    "prune_start": 15000,
    "prune_stop": 20000,
    "prune_every": 1000,
    "prune_opacity": 0.005,
    "seed": 0,
}


def check_trainable(params):
    for name, p in params.items():
        if not (isinstance(p, torch.nn.Parameter) and p.requires_grad and p.is_leaf):
            raise RuntimeError(f"compgs-density: field {name} is no longer a trainable leaf")


class CompGSDensity:
    def __init__(self, spec):
        unknown = set(spec.config) - set(DEFAULTS)
        if unknown:
            raise ValueError(f"Unknown compgs-density options: {sorted(unknown)}")
        self.cfg = {**DEFAULTS, **spec.config}
        c = self.cfg
        if set(c["codebooks"]) - set(vq_mod.FIELDS):
            raise ValueError(f"codebooks must be a subset of {sorted(vq_mod.FIELDS)}")
        if not 0 <= c["vq_start"] < c["total_steps"]:
            raise ValueError("compgs needs 0 <= vq_start < total_steps")
        self.base = Density(replace(spec, config=dict(c["base"])))
        self.requires = self.base.requires
        self.vq = None
        self.lr0 = None
        self.history = []
        self.encoded = None
        self.generator = None
        self.stats = {"assignments": 0, "updates": 0, "kmeans_seconds": 0.0}

    @property
    def state(self):
        out = {"events": self.history, "encoded": self.encoded, "stats": dict(self.stats)}
        if self.encoded is not None and self.vq is not None:
            out["vq"] = self.vq.to_state()
        return {"strategy": self.base.state, "compgs": out}

    def initialize(self, params, optimizers, scene_scale=1.0):
        self.base.initialize(params, optimizers, scene_scale)
        self.lr0 = optimizers["means"].param_groups[0]["lr"]
        self.vq = vq_mod.VQState(self.cfg["codebooks"], self.cfg["kmeans_iters"], self.cfg["chunk"])
        vq_mod.ACTIVE["vq"] = self.vq
        self.generator = torch.Generator(device=params["means"].device)
        self.generator.manual_seed(int(self.cfg["seed"]))

    def check_run_length(self, step, lr):
        if step != 1 or self.lr0 is None:
            return
        steps = schedule.inferred_run_length(self.lr0, lr)
        if steps is None:  # LR untouched (probes driving the density directly)
            print("compgs-density: WARNING cannot infer the run length from the means LR; the "
                  f"decoded model is baked at step {self.cfg['total_steps']}", flush=True)
            return
        if abs(steps - self.cfg["total_steps"]) > 0.5:
            raise ValueError(
                f"run has ~{round(steps)} steps but compgs-density total_steps="
                f"{self.cfg['total_steps']}: the decoded model would never be baked. Train for "
                "total_steps (30000 for the paper) or set the density config.")

    def before_backward(self, params, optimizers, step, outputs):
        self.base.before_backward(params, optimizers, step, outputs)

    def after_backward(self, params, optimizers, step, outputs):
        self.check_run_length(step, optimizers["means"].param_groups[0]["lr"])
        self.base.after_backward(params, optimizers, step, outputs)
        self.compgs_step(params, optimizers, step)

    def compgs_step(self, params, optimizers, step):
        c = self.cfg
        if schedule.prunes(step, c):
            self.prune(params, optimizers, step)
        if step == c["total_steps"]:
            self.bake(params, step)
            return
        event = schedule.vq_event(step + 1, c)
        if event is None:
            return
        t0 = time.perf_counter()
        if event == "assign":
            self.vq.assign(params, self.generator)
            self.stats["assignments"] += 1
            if self.stats["assignments"] == 1:
                self._log({"compgs": "vq_start", "it": step + 1, "gaussians": len(params["means"])})
        else:
            if self.vq.n != len(params["means"]):
                raise RuntimeError(
                    f"compgs: Gaussian count changed at step {step} without a k-means "
                    "assignment (densification or pruning inside the VQ phase)")
            self.vq.update(params)
            self.stats["updates"] += 1
        torch.cuda.synchronize() if params["means"].is_cuda else None
        self.stats["kmeans_seconds"] += time.perf_counter() - t0

    @torch.no_grad()
    def prune(self, params, optimizers, step):
        c = self.cfg
        mask = params["opacities"].sigmoid() < c["prune_opacity"]
        n0 = len(mask)
        if bool(mask.any()):
            ops.remove(params, optimizers, self.base.state, mask)
            if self.vq is not None and self.vq.ready:
                self.vq.keep(~mask)
        check_trainable(params)
        self._log({"compgs": "prune", "step": step, "before": n0, "after": len(params["means"])})

    @torch.no_grad()
    def bake(self, params, step):
        vq = self.vq
        if not vq.ready or vq.n != len(params["means"]):
            raise RuntimeError("compgs: no aligned VQ state at total_steps")
        for f, q in vq.quantizers.items():
            p = params[f]
            p.copy_(q.decoded(p.shape))
        vq.decoded = True
        if vq_mod.ACTIVE["vq"] is vq:
            vq_mod.ACTIVE["vq"] = None
        check_trainable(params)
        n = len(params["means"])
        cbs = {f: tuple(q.centers.shape) for f, q in vq.quantizers.items()}
        self.encoded = {
            "gaussians": n,
            "codebooks": {f: list(s) for f, s in cbs.items()},
            "used_codes": {f: int(torch.unique(q.idx).numel()) for f, q in vq.quantizers.items()},
            "paper_size": codec.paper_size(n, cbs),
            "official_disk_size": codec.official_disk_size(n, cbs),
            "uncompressed_3dgs": codec.uncompressed_3dgs_size(n),
            "stats": {**self.stats, "kmeans_seconds": round(self.stats["kmeans_seconds"], 1)},
        }
        self._log({"compgs": "bake", "step": step, "gaussians": n,
                   "paper_MB": round(self.encoded["paper_size"]["MB"], 3),
                   "official_MB": round(self.encoded["official_disk_size"]["MB"], 3),
                   "assignments": self.stats["assignments"]})

    def _log(self, record):
        self.history.append(record)
        print(json.dumps(record), flush=True)


def create_density(spec):
    return CompGSDensity(spec)
