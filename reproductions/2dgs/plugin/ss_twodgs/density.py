"""2DGS adaptive density control (official train.py:121-131 and scene/gaussian_model.py of
github.com/hbb1/2d-gaussian-splatting @f3e3b9f).

  init     rots = torch.rand(N, 4) (uniform [0,1), normalised by the getter); 2 scales =
           log sqrt(mean 3-NN dist^2) (runtime --recipe 3dgs), opacity 0.1
  stats    step < densify_until: radii > 0 -> accum += ||g_means2D[:, :2]||, denom += 1
           (g_means2D = dL/dT[:, 2] * depth * W/2, the projected 3D-centre gradient)
  densify  step > densify_from and step % 100 == 0 (grads = accum/denom, nan -> 0):
           clone  grads >= thr & max(s_u, s_v) <= percent_dense * extent: exact copies
           split  (post-clone population, clone grads padded with 0) grads >= thr &
                  max(s) > percent_dense * extent: N=2 samples ~ R [N(0, s_u), N(0, s_v), 0],
                  scales s / (0.8 N), originals removed
           prune  sigmoid(o) < 0.05; when step > opacity_reset_interval also
                  max(s) > 0.1 * extent. The official screen-size test (max_radii2D > 20)
                  never fires because densification_postfix zeroes max_radii2D first:
                  replicated by not applying it.
  reset    step < densify_until and step % 3000 == 0: opacity = min(o, 0.01), moments zeroed
The runtime applies the optimiser step before this controller (official: densify first).
The stored ``scales`` keep the runtime's three columns; only the first two are used, the third
is copied unchanged by every operation.
"""

import torch
from gsplat.strategy.ops import _update_param_with_optimizer, duplicate, remove, reset_opa
from gsplat.utils import normalized_quat_to_rotmat

DEFAULTS = {
    "densify_from_iter": 500,
    "densify_until_iter": 15000,
    "densification_interval": 100,
    "densify_grad_threshold": 0.0002,
    "percent_dense": 0.01,
    "opacity_cull": 0.05,
    "opacity_reset_interval": 3000,
    "opacity_reset_value": 0.01,
    "big_world_factor": 0.1,
    "split_n": 2,
    "uniform_quat_init": True,
}
REQUIRES = {"gradient.screen.projected_center"}


def check_trainable(params):
    """Guard against the frozen-Parameter pitfall (CLAUDE.md, ResGS)."""
    for name, p in params.items():
        if not (isinstance(p, torch.nn.Parameter) and p.is_leaf and p.requires_grad):
            raise RuntimeError(f"2dgs-density: field {name} is no longer a trainable leaf")


@torch.no_grad()
def split_surfels(params, optimizers, mask, n=2):
    """densify_and_split for 2D surfels: samples in the disk plane, scales / (0.8 n)."""
    sel = torch.where(mask)[0]
    rest = torch.where(~mask)[0]
    if len(sel) == 0:
        return
    s2 = params["scales"][sel, :2].exp().repeat(n, 1)
    stds = torch.cat([s2, torch.zeros_like(s2[:, :1])], dim=-1)
    samples = torch.normal(mean=torch.zeros_like(stds), std=stds)
    rots = normalized_quat_to_rotmat(
        torch.nn.functional.normalize(params["quats"][sel], dim=-1)).repeat(n, 1, 1)
    new_means = torch.bmm(rots, samples[..., None])[..., 0] + params["means"][sel].repeat(n, 1)
    new_scales = params["scales"][sel].repeat(n, 1).clone()
    new_scales[:, :2] = torch.log(s2 / (0.8 * n))

    def param_fn(name, p):
        if name == "means":
            new = new_means
        elif name == "scales":
            new = new_scales
        else:
            new = p[sel].repeat(n, *([1] * (p.dim() - 1)))
        return torch.nn.Parameter(torch.cat([p.detach()[rest], new]),
                                  requires_grad=p.requires_grad)

    def optimizer_fn(key, v):
        return torch.cat([v[rest], torch.zeros((n * len(sel), *v.shape[1:]), device=v.device)])

    _update_param_with_optimizer(param_fn, optimizer_fn, params, optimizers)


class Density:
    def __init__(self, spec):
        unknown = set(spec.config) - set(DEFAULTS)
        if unknown:
            raise ValueError(f"Unknown 2DGS density options: {sorted(unknown)}")
        self.c = {**DEFAULTS, **spec.config}
        self.requires = set(REQUIRES)
        self.extent = None
        self.accum = self.denom = None
        self.counts = {"densify": 0, "resets": 0}
        self.state = {"counts": self.counts}

    def initialize(self, params, optimizers, scene_scale=1.0):
        for name in ("means", "scales", "quats", "opacities"):
            if name not in params or name not in optimizers:
                raise ValueError(f"2dgs-density requires the {name} field")
        self.extent = float(scene_scale)  # recipe 3dgs: getNerfppNorm radius
        self.state["extent"] = self.extent
        if self.c["uniform_quat_init"]:
            with torch.no_grad():  # gaussian_model.create_from_pcd: torch.rand((N, 4))
                params["quats"].data.copy_(torch.rand_like(params["quats"]))
        self._reset_stats(len(params["means"]), params["means"].device)

    def _reset_stats(self, n, dev):
        self.accum = torch.zeros(n, device=dev)
        self.denom = torch.zeros(n, device=dev)

    def before_backward(self, params, optimizers, step, outputs):
        means2d = outputs["info"]["means2d"]
        if not means2d.requires_grad:
            raise RuntimeError("2dgs-density: screen-space gradient tensor is not tracked")

    @torch.no_grad()
    def after_backward(self, params, optimizers, step, outputs):
        c = self.c
        if step < c["densify_until_iter"]:
            self.accumulate(outputs["info"])
            if step > c["densify_from_iter"] and step % c["densification_interval"] == 0:
                self.densify_and_prune(params, optimizers, step > c["opacity_reset_interval"])
            if step % c["opacity_reset_interval"] == 0:
                reset_opa(params, optimizers, {}, c["opacity_reset_value"])
                self.counts["resets"] += 1
                check_trainable(params)

    def accumulate(self, info):
        g = info["means2d"].grad
        if g is None:
            raise RuntimeError("2dgs-density: screen-space gradients missing")
        vis = info["radii"] > 0
        self.accum[vis] += torch.linalg.vector_norm(g[vis, :2], dim=-1)
        self.denom[vis] += 1

    def densify_and_prune(self, params, optimizers, world_prune):
        c = self.c
        grads = self.accum / self.denom
        grads[grads.isnan()] = 0.0
        dense = c["percent_dense"] * self.extent
        n0 = len(grads)
        smax = params["scales"][:, :2].exp().max(dim=1).values
        clone = (grads >= c["densify_grad_threshold"]) & (smax <= dense)
        n_clone = int(clone.sum())
        if n_clone:
            duplicate(params, optimizers, {}, clone)
        grads = torch.cat([grads, torch.zeros(n_clone, device=grads.device)])
        smax = params["scales"][:, :2].exp().max(dim=1).values
        sel = (grads >= c["densify_grad_threshold"]) & (smax > dense)
        n_split = int(sel.sum())
        split_surfels(params, optimizers, sel, int(c["split_n"]))
        prune = params["opacities"].sigmoid() < c["opacity_cull"]
        if world_prune:
            prune = prune | (params["scales"][:, :2].exp().max(dim=1).values
                             > c["big_world_factor"] * self.extent)
        if int(prune.sum()):
            remove(params, optimizers, {}, prune)
        check_trainable(params)
        n = len(params["means"])
        self._reset_stats(n, params["means"].device)
        self.counts["densify"] += 1
        self.state["last"] = {"before": n0, "clone": n_clone, "split": n_split,
                              "prune": int(prune.sum()), "after": n}


def create_density(spec):
    return Density(spec)
