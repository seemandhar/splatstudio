"""SpotLessSplats density control: the official ``spotless_trainer.py`` refine loop.

* 3DGS duplicate / split (grad2d >= 2e-4, grow_scale3d 0.01) every 100 steps for 500 < t < 15000
  (``gsplat.strategy.ops`` has the official ``refine_duplicate`` / ``refine_split`` numerics:
  zero Adam moments for new Gaussians, split scale / 1.6, two N(0, Sigma) samples).
* Prune opacity < 0.005. No opacity reset (§4.2.3: replaced by UBP) and no scale prune (the
  official ``step > reset_every = 300000`` guard never fires), no screen-size prune.
* Utilisation-based pruning (Eq. 10, official CUDA ``sqrgrad``): accumulate per Gaussian the
  per-pixel squared screen-mean gradient between refinements, prune ``sqrgrad < 1e-14``;
  duplicated / split Gaussians start with utilisation 1.
* t = 8002: ``shN = clamp(shN, max=0.001)`` with zeroed Adam moments (§5 "reset the non-diffuse
  SH coefficients to 0.001 at the 8000th step"; code ``reset_sh = 8002``).

Step convention: runtime step k (1-based) = official t + 1.
"""

import torch
from gsplat.strategy import ops

from .ubp import utilization

DEFAULTS = {
    "refine_start": 500,
    "refine_stop": 15000,
    "refine_every": 100,
    "grow_grad2d": 0.0002,
    "grow_scale3d": 0.01,
    "prune_opa": 0.005,
    "ubp": True,
    "ubp_thresh": 1e-14,
    "reset_sh_step": 8002,
    "reset_sh_value": 0.001,
}


def requirements(ubp):
    req = {"gradient.screen.signed"}
    if ubp:
        req |= {"render.rgb_grad", "render.sh_colors"}
    return req


@torch.no_grad()
def reset_sh(params, optimizers, value):
    """Official ``Runner.reset_sh``: clamp(shN, max=|value|), Adam moments zeroed (step kept)."""
    def param_fn(name, p):
        return torch.nn.Parameter(torch.clamp(p, max=abs(value)), requires_grad=p.requires_grad)

    def optimizer_fn(key, v):
        return torch.zeros_like(v)

    ops._update_param_with_optimizer(param_fn, optimizer_fn, params, optimizers, names=["shN"])


class Density:
    def __init__(self, spec):
        cfg = {**DEFAULTS, **spec.config}
        unknown = set(cfg) - set(DEFAULTS)
        if unknown:
            raise ValueError(f"Unknown sls-density options: {sorted(unknown)}")
        self.cfg = cfg
        self.ubp = bool(cfg["ubp"])
        self.requires = requirements(self.ubp)
        self.state = None
        self.scene_scale = 1.0
        self.last = {}

    def initialize(self, params, optimizers, scene_scale=1.0):
        n = len(params["means"])
        dev = params["means"].device
        self.scene_scale = float(scene_scale)
        self.state = {
            "grad2d": torch.zeros(n, device=dev),
            "count": torch.zeros(n, device=dev, dtype=torch.int),
            "sqrgrad": torch.zeros(n, device=dev),
        }

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
        if self.ubp:
            v_render = outputs.get("grads", {}).get("rgb")
            if v_render is None:
                raise RuntimeError("UBP needs dL/dI: the renderer hook did not fire")
            sqr = utilization(info, outputs["sh_colors"], v_render)
            self.state["sqrgrad"].index_add_(0, gs_ids, sqr[gs_ids])

    @torch.no_grad()
    def refine(self, params, optimizers):
        cfg, st = self.cfg, self.state
        grads = st["grad2d"] / st["count"].clamp_min(1)
        is_grad_high = grads >= cfg["grow_grad2d"]
        is_small = (torch.exp(params["scales"]).max(dim=-1).values
                    <= cfg["grow_scale3d"] * self.scene_scale)
        is_dupli = is_grad_high & is_small
        n_dupli = int(is_dupli.sum())
        ops.duplicate(params, optimizers, st, is_dupli)
        if n_dupli:
            st["sqrgrad"][-n_dupli:] = 1.0   # new Gaussians assumed highly utilised (official)
        is_split = is_grad_high & ~is_small
        is_split = torch.cat([is_split, torch.zeros(n_dupli, device=is_split.device,
                                                    dtype=torch.bool)])
        n_split = int(is_split.sum())
        ops.split(params, optimizers, st, is_split)
        if n_split:
            st["sqrgrad"][-2 * n_split:] = 1.0
        is_prune = torch.sigmoid(params["opacities"]) < cfg["prune_opa"]
        n_unused = 0
        if self.ubp:
            not_utilized = st["sqrgrad"] < cfg["ubp_thresh"]
            n_unused = int((not_utilized & ~is_prune).sum())
            is_prune = is_prune | not_utilized
        n_prune = int(is_prune.sum())
        if n_prune:
            ops.remove(params, optimizers, st, is_prune)
        st["grad2d"].zero_()
        st["count"].zero_()
        st["sqrgrad"].zero_()
        self.last = {"duplicated": n_dupli, "split": n_split, "pruned": n_prune,
                     "pruned_unutilized_only": n_unused}

    def after_backward(self, params, optimizers, step, outputs):
        cfg = self.cfg
        t = int(step) - 1
        if t < cfg["refine_stop"]:
            self.update_stats(outputs)
            if t > cfg["refine_start"] and t % cfg["refine_every"] == 0:
                self.refine(params, optimizers)
            if t == cfg["reset_sh_step"]:
                reset_sh(params, optimizers, float(cfg["reset_sh_value"]))


def create_density(spec):
    return Density(spec)
