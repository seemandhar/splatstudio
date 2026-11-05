"""3DGS adaptive density control as in the official Analytic-Splatting train.py
(github.com/lzhnb/Analytic-Splatting a905939, `dense=False`, `filter3d=False`).

Per training step s (the runtime calls `after_backward` after the optimizer step; the official
loop densifies before it):
  s < densify_until: for visible Gaussians (radii > 0)
      grad_accum += ||dL/d means2D[:, :2]||   (Inria NDC-space screen gradient), denom += 1
    s > densify_from and s % 100 == 0: densify_and_clone, densify_and_split, prune
      (opacity < 0.005; after step 3000 also max scale > 0.1 * extent), reset statistics.
      The 20 px screen-size prune is a no-op in the official code (densification_postfix
      zeroes max_radii2D first) and is not applied.
    s % 3000 == 0: reset_opacity with the Mip-Splatting filter coefficient:
      o <- min(o, 0.01 / coef) (filter recomputed from the training cameras on the
      post-densification population, as compute_3D_filter after densify_and_prune).
"""

import torch
from gsplat.strategy.ops import _update_param_with_optimizer, remove
from torch import nn

from .filters import compute_filter_3d, reset_logits

DEFAULTS = {
    "densify_from": 500,  # OptimizationParams.densify_from_iter
    "densify_until": 15000,  # densify_until_iter
    "densify_every": 100,  # densification_interval
    "grad_threshold": 0.0002,  # densify_grad_threshold
    "percent_dense": 0.01,
    "min_opacity": 0.005,  # train.py:214
    "big_world_after": 3000,  # size_threshold set when iteration > opacity_reset_interval
    "big_world": 0.1,  # big_points_ws: max scale > 0.1 * extent
    "reset_every": 3000,  # opacity_reset_interval
    "reset_value": 0.01,
    "filter_coef": 0.2,  # compute_3D_filter: distance / focal * 0.2 ** 0.5
    "near": 0.2,  # valid_depth = z > 0.2
    "frustum_margin": 0.15,  # in_screen with 15 % margin
}
REQUIRES = {"gradient.screen.signed"}
BASE_FIELDS = ("means", "scales", "quats", "opacities", "sh0", "shN")


def quat_to_rotmat(quats):
    """utils/general_utils.build_rotation: wxyz (any norm) -> rotation matrices [N,3,3]."""
    q = quats / quats.norm(dim=-1, keepdim=True)
    r, x, y, z = q.unbind(-1)
    return torch.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - r * z), 2 * (x * z + r * y),
        2 * (x * y + r * z), 1 - 2 * (x * x + z * z), 2 * (y * z - r * x),
        2 * (x * z - r * y), 2 * (y * z + r * x), 1 - 2 * (x * x + y * y)], -1).reshape(-1, 3, 3)


def append_gaussians(params, optimizers, new):
    """cat_tensors_to_optimizer: append rows, new Adam moments are zero."""
    n_new = len(new["means"])

    def param_fn(name, p):
        return nn.Parameter(torch.cat([p.detach(), new[name].to(p.dtype)]),
                            requires_grad=p.requires_grad)

    def optimizer_fn(key, v):
        return torch.cat([v, torch.zeros((n_new, *v.shape[1:]), dtype=v.dtype,
                                         device=v.device)])

    _update_param_with_optimizer(param_fn, optimizer_fn, params, optimizers)


class AnalyticDensity:
    def __init__(self, spec):
        unknown = set(spec.config) - set(DEFAULTS)
        if unknown:
            raise ValueError(f"Unknown Analytic-Splatting density options: {sorted(unknown)}")
        self.cfg = {**DEFAULTS, **spec.config}
        self.requires = set(REQUIRES)
        self.state = None
        self.cameras = None
        self.generator = None

    # ------------------------------------------------------------------ runtime hooks
    def set_training_cameras(self, cameras):
        cameras = list(cameras)
        if not cameras:
            raise ValueError("Analytic-Splatting opacity reset needs the training cameras")
        self.cameras = cameras

    def initialize(self, params, optimizers, scene_scale=1.0):
        extra = set(params) - set(BASE_FIELDS)
        if extra:
            raise ValueError(f"Analytic-Splatting density does not know fields {sorted(extra)}")
        self.extent = float(scene_scale)
        self._reset_stats(len(params["means"]), params["means"].device)
        self.state = {"events": [], "resets": []}

    def before_backward(self, params, optimizers, step, outputs):
        if step < self.cfg["densify_until"]:
            outputs["info"]["means2d"].retain_grad()

    @torch.no_grad()
    def after_backward(self, params, optimizers, step, outputs):
        cfg = self.cfg
        if step >= cfg["densify_until"]:
            return
        self._accumulate(outputs["info"])
        if step > cfg["densify_from"] and step % cfg["densify_every"] == 0:
            self._densify(params, optimizers, step)
        if cfg["reset_every"] > 0 and step % cfg["reset_every"] == 0:
            self.reset_opacity(params, optimizers, step)

    # ------------------------------------------------------------------ statistics
    def _reset_stats(self, n, device):
        self.grad_accum = torch.zeros(n, device=device)
        self.denom = torch.zeros(n, device=device)

    def _accumulate(self, info):
        g2 = info["means2d"].grad
        if g2 is None:
            raise RuntimeError("screen-space gradient missing: render with the "
                               "Analytic-Splatting renderer")
        vis = info["radii"] > 0
        self.grad_accum[vis] += torch.norm(g2[vis, :2], dim=-1)  # add_densification_stats
        self.denom[vis] += 1

    # ------------------------------------------------------------------ densification
    def _densify(self, params, optimizers, step):
        cfg = self.cfg
        n0 = len(params["means"])
        grads = self.grad_accum / self.denom
        grads[grads.isnan()] = 0.0
        # densify_and_clone
        small = params["scales"].exp().max(dim=1).values <= cfg["percent_dense"] * self.extent
        clone = (grads >= cfg["grad_threshold"]) & small
        append_gaussians(params, optimizers, {k: params[k][clone] for k in BASE_FIELDS})
        n_clone = int(clone.sum())
        # densify_and_split (clones padded with zero gradient)
        grads = torch.cat([grads, torch.zeros(n_clone, device=grads.device)])
        split = (grads >= cfg["grad_threshold"]) & (
            params["scales"].exp().max(dim=1).values > cfg["percent_dense"] * self.extent)
        K = int(split.sum())
        stds = params["scales"][split].exp().repeat(2, 1)
        samples = torch.normal(torch.zeros_like(stds), stds, generator=self.generator)
        rots = quat_to_rotmat(params["quats"][split]).repeat(2, 1, 1)
        new = {"means": torch.bmm(rots, samples[..., None])[..., 0]
               + params["means"][split].repeat(2, 1),
               "scales": torch.log(params["scales"][split].exp().repeat(2, 1) / (0.8 * 2)),
               "quats": params["quats"][split].repeat(2, 1),
               "opacities": params["opacities"][split].repeat(2),
               "sh0": params["sh0"][split].repeat(2, 1, 1),
               "shN": params["shN"][split].repeat(2, 1, 1)}
        append_gaussians(params, optimizers, new)
        remove(params, optimizers, {}, torch.cat([split, torch.zeros(
            2 * K, dtype=torch.bool, device=split.device)]))
        # prune (densify_and_prune)
        prune = params["opacities"].sigmoid() < cfg["min_opacity"]
        if step > cfg["big_world_after"]:
            prune |= params["scales"].exp().max(dim=1).values > cfg["big_world"] * self.extent
        n_prune = int(prune.sum())
        remove(params, optimizers, {}, prune)
        self._reset_stats(len(params["means"]), params["means"].device)
        row = {"step": int(step), "n_before": n0, "clone": n_clone, "split": K,
               "pruned": n_prune, "n_after": len(params["means"])}
        self.state["events"].append(row)
        if step % 1000 == 0:
            print({"analytic-splatting": row}, flush=True)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def filter_3d(self, params):
        if not self.cameras:
            raise RuntimeError("Analytic-Splatting density needs set_training_cameras(...) "
                               "before training (filter-aware opacity reset)")
        dev = params["means"].device
        if self.cameras[0]["viewmat"].device != dev:
            self.cameras = [{k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in c.items()}
                            for c in self.cameras]
        return compute_filter_3d(params["means"], self.cameras, self.cfg["filter_coef"],
                                 self.cfg["near"], self.cfg["frustum_margin"])

    def reset_opacity(self, params, optimizers, step=0):
        f = self.filter_3d(params)
        logits = reset_logits(params["scales"].detach(), params["opacities"].detach(), f,
                              self.cfg["reset_value"])

        def param_fn(name, p):
            return nn.Parameter(logits.to(p.dtype), requires_grad=p.requires_grad)

        _update_param_with_optimizer(param_fn, lambda key, v: torch.zeros_like(v), params,
                                     optimizers, ["opacities"])
        self.state["resets"].append({"step": int(step),
                                     "mean_opacity": float(logits.sigmoid().mean())})


def create_density(spec):
    return AnalyticDensity(spec)
