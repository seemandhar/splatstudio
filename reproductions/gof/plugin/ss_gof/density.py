"""GOF adaptive density control, Mip-Splatting 3D filter schedule and official xyz LR.

Port of GOF @5245b20 train.py (densification block) and scene/gaussian_model.py
(add_densification_stats, densify_and_clone, densify_and_split, densify_and_prune,
reset_opacity, compute_3D_filter):

  stats    step < densify_until, radii > 0: accum += ||g[:, :2]||, accum_abs += |g[:, 2]|,
           denom += 1 (g = means2D.grad from the GOF kernel; g[:, 2] = sum_pixels |gx|+|gy|)
  densify  step > densify_from and step % interval == 0:
           grads = accum/denom, grads_abs = accum_abs/denom (nan -> 0)
           ratio = mean(grads >= thr); Q = quantile(grads_abs, 1 - ratio)      (paper metric M)
           clone: (grads >= thr | grads_abs >= Q) & max(s) <= percent_dense * extent,
                  new position sampled from N(mu, Sigma) (supp. A), rest copied
           split: same criterion with max(s) > percent_dense * extent on the post-clone
                  population (clones padded with zero gradients), 2 samples, s / 1.6
           prune: sigmoid(o) < min_opacity (0.05); after opacity_reset_interval also
                  max(s) > 0.1 * extent. The official screen-size prune (max_radii2D > 20) never
                  fires because densification_postfix zeroes max_radii2D first; not applied.
           all statistics restart at zero; 3D filter recomputed
  reset    step % 3000 == 0 (step < densify_until): filtered opacity clamped to 0.01
  filter   also every 100 steps for densify_until < step < total - 100
  xyz lr   get_expon_lr_func(1.6e-4 * extent -> 1.6e-6 * extent, 30000 steps), overriding the
           runtime's 0.01 ** (step / steps) decay.
Difference to the official loop: the runtime applies the optimizer step before this
controller runs (official densifies first; the new tensors then skip that step).
"""

import math

import torch
from gsplat.strategy.ops import _update_param_with_optimizer, remove, split
from gsplat.utils import normalized_quat_to_rotmat

from .filters import compute_filter_3d, reset_logits

DEFAULTS = {
    "densify_from_iter": 500,
    "densify_until_iter": 15000,
    "densification_interval": 100,
    "densify_grad_threshold": 0.0002,
    "abs_gradient": True,          # GOF metric M (False = plain 3DGS criterion, ablation)
    "clone_resample": True,        # supp. A: cloned Gaussians sample a new position
    "percent_dense": 0.01,
    "min_opacity": 0.05,
    "opacity_reset_interval": 3000,
    "opacity_reset_value": 0.01,
    "big_world_factor": 0.1,
    "filter_coef": 0.2,
    "filter_near": 0.2,
    "filter_margin": 0.15,
    "filter_interval": 100,
    "total_steps": 30000,
    "position_lr_init": 1.6e-4,
    "position_lr_final": 1.6e-6,
    "position_lr_delay_mult": 0.01,
    "position_lr_delay_steps": 0,
    "position_lr_max_steps": 30000,
}
REQUIRES = {"gradient.screen.gof"}


def expon_lr(step, lr_init, lr_final, delay_steps=0, delay_mult=1.0, max_steps=30000):
    """utils/general_utils.get_expon_lr_func."""
    if step < 0 or (lr_init == 0.0 and lr_final == 0.0):
        return 0.0
    if delay_steps > 0:
        delay_rate = delay_mult + (1 - delay_mult) * math.sin(
            0.5 * math.pi * min(max(step / delay_steps, 0), 1))
    else:
        delay_rate = 1.0
    t = min(max(step / max_steps, 0), 1)
    return delay_rate * math.exp(math.log(lr_init) * (1 - t) + math.log(lr_final) * t)


@torch.no_grad()
def clone_resampled(params, optimizers, mask, resample=True):
    """densify_and_clone: append copies; means drawn from N(mu, R diag(s)^2 R^T)."""
    sel = torch.where(mask)[0]
    new_means = params["means"][sel]
    if resample:
        stds = params["scales"][sel].exp()
        rots = normalized_quat_to_rotmat(torch.nn.functional.normalize(params["quats"][sel],
                                                                       dim=-1))
        samples = torch.normal(mean=torch.zeros_like(stds), std=stds)
        new_means = torch.bmm(rots, samples[..., None])[..., 0] + new_means

    def param_fn(name, p):
        new = new_means if name == "means" else p[sel]
        return torch.nn.Parameter(torch.cat([p, new]), requires_grad=p.requires_grad)

    def optimizer_fn(key, v):
        return torch.cat([v, torch.zeros((len(sel), *v.shape[1:]), device=v.device)])

    _update_param_with_optimizer(param_fn, optimizer_fn, params, optimizers)


def is_filter_step(step, interval, densify_from, densify_until, total):
    if step % interval:
        return False
    if densify_from < step < densify_until:
        return True  # after densify_and_prune
    return densify_until < step < total - interval


class Density:
    requires = REQUIRES

    def __init__(self, spec):
        unknown = set(spec.config) - set(DEFAULTS)
        if unknown:
            raise ValueError(f"Unknown GOF density options: {sorted(unknown)}")
        self.cfg = {**DEFAULTS, **spec.config}
        self.requires = set(REQUIRES)
        self.state = None
        self.stats = None
        self.extent = None
        self.params = None
        self.cameras = None
        self.counts = {"filter_updates": 0, "resets": 0, "densify": 0}

    # -- runtime hooks -------------------------------------------------------------------
    def initialize(self, params, optimizers, scene_scale=1.0):
        self.extent = float(scene_scale)  # 3dgs recipe: getNerfppNorm radius
        self.params = params
        self._reset_stats(len(params["means"]), params["means"].device)
        self.state = {"extent": self.extent, "counts": self.counts}
        if self.cameras:
            self.update_filter(params)

    def set_training_cameras(self, cameras):
        cameras = list(cameras)
        if not cameras:
            raise ValueError("GOF needs the training cameras for the 3D filter")
        self.cameras = cameras
        if self.params is not None:
            self.update_filter(self.params)  # official: compute_3D_filter before iteration 1

    def before_backward(self, params, optimizers, step, outputs):
        if not self.cameras:
            raise RuntimeError("GOF 3D filter needs density.set_training_cameras(...) first")
        if step == 1:
            optimizers["means"].param_groups[0]["lr"] = self.xyz_lr(1)

    @torch.no_grad()
    def after_backward(self, params, optimizers, step, outputs):
        c = self.cfg
        if step < c["densify_until_iter"]:
            self.accumulate(outputs["info"])
            if step > c["densify_from_iter"] and step % c["densification_interval"] == 0:
                self.densify_and_prune(params, optimizers, step > c["opacity_reset_interval"])
            if step % c["opacity_reset_interval"] == 0:
                self.reset_opacity(params, optimizers)
        if is_filter_step(step, c["filter_interval"], c["densify_from_iter"],
                          c["densify_until_iter"], c["total_steps"]):
            self.update_filter(params)
        optimizers["means"].param_groups[0]["lr"] = self.xyz_lr(step + 1)

    # -- pieces ----------------------------------------------------------------------------
    def xyz_lr(self, step):
        c = self.cfg
        return expon_lr(step, c["position_lr_init"] * self.extent,
                        c["position_lr_final"] * self.extent, c["position_lr_delay_steps"],
                        c["position_lr_delay_mult"], c["position_lr_max_steps"])

    def _reset_stats(self, n, dev):
        self.stats = {"grad_accum": torch.zeros(n, device=dev),
                      "grad_accum_abs": torch.zeros(n, device=dev),
                      "denom": torch.zeros(n, device=dev)}

    def accumulate(self, info):
        grad = info["means2d"].grad
        if grad is None:
            raise RuntimeError("GOF density: means2d has no gradient (renderer misconfigured)")
        vis = info["radii"] > 0
        st = self.stats
        st["grad_accum"][vis] += torch.linalg.vector_norm(grad[vis, :2], dim=-1)
        st["grad_accum_abs"][vis] += grad[vis, 2].abs()
        st["denom"][vis] += 1

    def update_filter(self, params):
        dev = params["means"].device
        if self.cameras[0]["viewmat"].device != dev:
            self.cameras = [{k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in cam.items()}
                            for cam in self.cameras]
        c = self.cfg
        f = compute_filter_3d(params["means"], self.cameras, c["filter_coef"], c["filter_near"],
                              c["filter_margin"])
        params["filter_3d"].data.copy_(f.to(params["filter_3d"].dtype))
        self.counts["filter_updates"] += 1

    def reset_opacity(self, params, optimizers):
        logits = reset_logits(params["scales"], params["opacities"], params["filter_3d"],
                              self.cfg["opacity_reset_value"])

        def param_fn(name, p):
            return torch.nn.Parameter(logits.to(p.dtype), requires_grad=p.requires_grad)

        _update_param_with_optimizer(param_fn, lambda key, v: torch.zeros_like(v), params,
                                     optimizers, ["opacities"])
        self.counts["resets"] += 1

    @torch.no_grad()
    def densify_and_prune(self, params, optimizers, world_prune):
        c, st = self.cfg, self.stats
        grads = st["grad_accum"] / st["denom"]
        grads[grads.isnan()] = 0.0
        grads_abs = st["grad_accum_abs"] / st["denom"]
        grads_abs[grads_abs.isnan()] = 0.0
        thr, dense = c["densify_grad_threshold"], c["percent_dense"] * self.extent
        n0 = len(grads)
        if c["abs_gradient"]:
            ratio = (grads >= thr).float().mean()
            q = torch.quantile(grads_abs, 1 - ratio)
        smax = params["scales"].exp().max(dim=1).values
        cand = grads >= thr
        if c["abs_gradient"]:
            cand = cand | (grads_abs >= q)
        clone = cand & (smax <= dense)
        clone_resampled(params, optimizers, clone, c["clone_resample"])
        n_clone = int(clone.sum())
        # densify_and_split pads grads with zeros for the clones and re-tests the threshold
        pad = torch.zeros(n_clone, device=grads.device)
        g_pad, ga_pad = torch.cat([grads, pad]), torch.cat([grads_abs, pad])
        cand = g_pad >= thr
        if c["abs_gradient"]:
            cand = cand | (ga_pad >= q)
        smax = params["scales"].exp().max(dim=1).values
        split_mask = cand & (smax > dense)
        split(params, optimizers, {}, split_mask)
        prune = params["opacities"].sigmoid() < c["min_opacity"]
        if world_prune:
            prune = prune | (params["scales"].exp().max(dim=1).values
                             > c["big_world_factor"] * self.extent)
        remove(params, optimizers, {}, prune)
        n = len(params["means"])
        self._reset_stats(n, params["means"].device)
        self.counts["densify"] += 1
        self.state["last"] = {"before": n0, "clone": n_clone, "split": int(split_mask.sum()),
                              "prune": int(prune.sum()), "after": n}


def create_density(spec):
    return Density(spec)
