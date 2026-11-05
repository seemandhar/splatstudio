"""Official DropGaussian / 3DGS density control (source/code/train.py:106-117,
scene/gaussian_model.py:350-409).

Every `densification_interval` steps with densify_from_iter < step < densify_until_iter:
clone (grad >= tau and max scale <= percent_dense * extent), then split the post-clone set
(clones carry a zero padded gradient; grad >= tau and max scale > percent_dense * extent; two
samples from N(mu, Sigma), scale / 1.6), then prune opacity < min_opacity. `size_threshold` is
None in the official call, which disables both screen-size and world-size pruning. Opacity is
reset to min(o, 0.01) every `opacity_reset_interval` steps while densifying (3000/6000/9000).

Gradient statistics follow add_densification_stats: |dL/d(NDC xy)| summed over steps in which
the Gaussian is visible (radius > 0, computed without the dropout mask, see renderer.py).
"""

import torch
from gsplat.strategy.ops import duplicate, remove, reset_opa, split

DEFAULTS = {
    "grad_threshold": 0.0005,
    "densify_from_iter": 500,
    "densify_until_iter": 10000,
    "densification_interval": 100,
    "percent_dense": 0.01,
    "min_opacity": 0.005,
    "opacity_reset_interval": 3000,
    "opacity_reset_value": 0.01,
}


class DropGaussianDensity:
    requires = frozenset({"gradient.screen.signed"})

    def __init__(self, spec):
        unknown = set(spec.config) - set(DEFAULTS)
        if unknown:
            raise ValueError(f"Unknown DropGaussian density options: {sorted(unknown)}")
        self.config = {**DEFAULTS, **spec.config}
        self.state = {"grad2d": None, "count": None}
        self.extent = 1.0
        self.log = []

    def initialize(self, params, optimizers, scene_scale=1.0):
        self.extent = float(scene_scale)
        self._reset_stats(len(params["means"]), params["means"].device)

    def _reset_stats(self, n, device):
        self.state["grad2d"] = torch.zeros(n, device=device)
        self.state["count"] = torch.zeros(n, device=device)

    def before_backward(self, params, optimizers, step, outputs):
        drop = outputs.get("drop")
        if drop is not None and drop["step"] != step:
            raise RuntimeError(
                f"DropGaussian schedule out of sync: renderer step {drop['step']} != {step}"
            )
        outputs["info"]["means2d"].retain_grad()

    def accumulate(self, info):
        means2d = info["means2d"]
        if means2d.grad is None:
            return
        n = means2d.shape[-2]
        if self.state["grad2d"] is None or len(self.state["grad2d"]) != n:
            self._reset_stats(n, means2d.device)
        grads = means2d.grad.clone()
        # gsplat pixel gradients -> 3DGS NDC gradients (x W/2, y H/2).
        grads[..., 0] *= info["width"] / 2.0 * info["n_cameras"]
        grads[..., 1] *= info["height"] / 2.0 * info["n_cameras"]
        sel = (info["visibility_radii"] > 0).all(dim=-1)  # [C, N]
        ids = torch.where(sel)[1]
        self.state["grad2d"].index_add_(0, ids, grads[sel].norm(dim=-1))
        self.state["count"].index_add_(0, ids, torch.ones_like(ids, dtype=torch.float32))

    @torch.no_grad()
    def densify_and_prune(self, params, optimizers):
        cfg, st = self.config, self.state
        tau, pd = float(cfg["grad_threshold"]), float(cfg["percent_dense"])
        grads = st["grad2d"] / st["count"]
        grads[grads.isnan()] = 0.0
        is_dup = (grads >= tau) & (params["scales"].exp().max(dim=1).values <= pd * self.extent)
        n_dup = int(is_dup.sum())
        if n_dup:
            duplicate(params, optimizers, st, is_dup)
        padded = torch.cat([grads, torch.zeros(n_dup, device=grads.device)])
        is_split = (padded >= tau) & (
            params["scales"].exp().max(dim=1).values > pd * self.extent
        )
        n_split = int(is_split.sum())
        if n_split:
            split(params, optimizers, st, is_split)
        is_prune = params["opacities"].sigmoid() < float(cfg["min_opacity"])
        n_prune = int(is_prune.sum())
        if n_prune:
            remove(params, optimizers, st, is_prune)
        self._reset_stats(len(params["means"]), params["means"].device)
        return {"cloned": n_dup, "split": n_split, "pruned": n_prune,
                "gaussians": len(params["means"])}

    def resets_opacity(self, step):
        cfg = self.config
        return step < int(cfg["densify_until_iter"]) and step % int(
            cfg["opacity_reset_interval"]
        ) == 0

    def after_backward(self, params, optimizers, step, outputs):
        cfg = self.config
        if step >= int(cfg["densify_until_iter"]):
            return
        self.accumulate(outputs["info"])
        if step > int(cfg["densify_from_iter"]) and step % int(cfg["densification_interval"]) == 0:
            self.log.append({"step": step, **self.densify_and_prune(params, optimizers)})
        if self.resets_opacity(step):
            reset_opa(params, optimizers, self.state, value=float(cfg["opacity_reset_value"]))


def create_density(spec):
    return DropGaussianDensity(spec)
