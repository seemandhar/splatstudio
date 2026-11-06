"""RobustSplat density control = Inria 3DGS ``densify_and_prune`` with the delayed schedule of
the official ``train.py`` / ``arguments/__init__.py`` (C1, "delayed Gaussian growth", §4.3, supp.
"Training details"):

* gradient statistics (‖∂L/∂μ_2D‖ in Inria NDC-pixel units, visibility radii > 0) accumulated for
  every step < densify_until_iter = 20000, starting at step 1;
* densify (clone if max scale <= percent_dense·extent, split otherwise; split scale / 1.6 with two
  N(0, Σ) samples — ``gsplat.strategy.ops``) and prune opacity < 0.005 when
  densify_from_iter = 10000 < step < 20000 and step % 100 == 0 (3DGS: 500 / 15000);
  no screen-size or world-size pruning (official passes max_screen_size=None);
* opacity reset to min(σ, 0.01) when step >= reset_iter = 15000, step % 3000 == 0, step < 20000
  (i.e. 15000 and 18000; 3DGS: every 3000 from 3000);
* opacity learning rate 0.025 (official / Inria 2024 default; SplatStudio's default is 0.05).

Step convention: the runtime's 1-based step == the official 1-based iteration.
"""

import torch
from gsplat.strategy import ops

DEFAULTS = {
    "refine_start": 10000,
    "refine_stop": 20000,
    "refine_every": 100,
    "grow_grad2d": 0.0002,
    "percent_dense": 0.01,
    "prune_opa": 0.005,
    "reset_start": 15000,
    "reset_every": 3000,
    "reset_value": 0.01,
    "opacity_lr": 0.025,
}
REQUIRES = frozenset({"gradient.screen.signed"})


def refine_steps(cfg, steps):
    return [s for s in range(1, steps + 1) if s < cfg["refine_stop"] and s > cfg["refine_start"]
            and s % cfg["refine_every"] == 0]


def reset_steps(cfg, steps):
    return [s for s in range(1, steps + 1) if s < cfg["refine_stop"] and s >= cfg["reset_start"]
            and s % cfg["reset_every"] == 0]


class Density:
    def __init__(self, spec):
        cfg = {**DEFAULTS, **spec.config}
        unknown = set(cfg) - set(DEFAULTS)
        if unknown:
            raise ValueError(f"Unknown rs-density options: {sorted(unknown)}")
        self.cfg = cfg
        self.requires = set(REQUIRES)
        self.state = None
        self.scene_scale = 1.0
        self.last = {}
        self.events = []

    def initialize(self, params, optimizers, scene_scale=1.0):
        n = len(params["means"])
        dev = params["means"].device
        self.scene_scale = float(scene_scale)
        self.state = {"grad2d": torch.zeros(n, device=dev),
                      "count": torch.zeros(n, device=dev, dtype=torch.int)}
        if self.cfg["opacity_lr"] is not None:
            for group in optimizers["opacities"].param_groups:
                group["lr"] = float(self.cfg["opacity_lr"])

    def before_backward(self, params, optimizers, step, outputs):
        outputs["info"]["means2d"].retain_grad()

    @torch.no_grad()
    def update_stats(self, outputs):
        info = outputs["info"]
        grad = info["means2d"].grad
        if grad is None:
            return
        grads = grad.clone()
        grads[..., 0] *= info["width"] / 2.0
        grads[..., 1] *= info["height"] / 2.0
        radii = info["radii"]
        sel = (radii > 0).all(dim=-1) if radii.dim() == 3 else radii > 0   # [1, N]
        gs_ids = torch.where(sel)[1]
        self.state["grad2d"].index_add_(0, gs_ids, grads[sel].norm(dim=-1))
        self.state["count"].index_add_(0, gs_ids, torch.ones_like(gs_ids, dtype=torch.int))

    @torch.no_grad()
    def refine(self, params, optimizers):
        cfg, st = self.cfg, self.state
        grads = st["grad2d"] / st["count"].clamp_min(1)   # Inria: accum / denom, NaN -> 0
        is_grad_high = grads >= cfg["grow_grad2d"]
        is_small = (torch.exp(params["scales"]).max(dim=-1).values
                    <= cfg["percent_dense"] * self.scene_scale)
        is_dupli = is_grad_high & is_small
        n_dupli = int(is_dupli.sum())
        ops.duplicate(params, optimizers, st, is_dupli)
        is_split = is_grad_high & ~is_small
        is_split = torch.cat([is_split, torch.zeros(n_dupli, device=is_split.device,
                                                    dtype=torch.bool)])
        n_split = int(is_split.sum())
        ops.split(params, optimizers, st, is_split)
        is_prune = torch.sigmoid(params["opacities"]) < cfg["prune_opa"]
        n_prune = int(is_prune.sum())
        if n_prune:
            ops.remove(params, optimizers, st, is_prune)
        st["grad2d"].zero_()
        st["count"].zero_()
        self.last = {"duplicated": n_dupli, "split": n_split, "pruned": n_prune}

    def after_backward(self, params, optimizers, step, outputs):
        cfg = self.cfg
        step = int(step)
        if step >= cfg["refine_stop"]:
            return
        self.update_stats(outputs)
        if step > cfg["refine_start"] and step % cfg["refine_every"] == 0:
            self.refine(params, optimizers)
            self.events.append(("refine", step))
        if step >= cfg["reset_start"] and step % cfg["reset_every"] == 0:
            ops.reset_opa(params, optimizers, self.state, value=float(cfg["reset_value"]))
            self.events.append(("reset", step))


def create_density(spec):
    return Density(spec)
