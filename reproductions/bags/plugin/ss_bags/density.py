"""BAGS density control, Mip 3D filter, xyz LR schedule and the BPN bank.

Port of github.com/snldmt/BAGS @ed04144 train.py (densification block) and
scene/gaussian_model.py (Mip-Splatting based). With it = phase-local iteration (schedule.py):

  stats    it < densify_until: accum += ||dL/d means2D_ndc||, denom += 1 for radii > 0
           (gsplat pixel gradients x (W/2, H/2) of the current render = official NDC gradient)
  densify  it > 500, it % 100 == 0: clone (grad >= thr, max s <= 0.01 extent), split (post-clone
           population, 2 samples, s / 1.6), prune sigmoid(o) < min_opacity. thr / min_opacity =
           init_dgt / min_opacity while step <= ms_steps, densify_grad_threshold afterwards.
           Screen-size prune: official max_radii2D is zeroed by densification_postfix before the
           prune, so it never fires (not ported); world-size prune (max s > 0.1 extent) only when
           min_opacity == 0.005 and it > 3000 (official condition), off for the deblur scripts.
  reset    it % 3000 == 0, it < densify_until: filtered opacity clamped to 0.01 (Mip-Splatting).
  filter   3D filter from the training cameras *at the current resolution*: at init (1/4 res),
           when the resolution changes (computed for the next step's cameras), after every
           densification, and every 100 it for densify_until < it < total - ms_steps - 100.
  xyz lr   get_expon_lr_func(1.6e-4 extent -> 1.6e-6 extent, max_steps 60000) evaluated at it,
           i.e. restarted after ms_steps ("lr reset").
  final    the official loop saves before the last iteration's densification block: nothing
           changes the Gaussians at step == total_steps.
  bank     BPN Adam (lr bpn_lr, eps 1e-15) stepped every step.
Difference to the official loop: the runtime Adam step precedes this controller; officially
densify / reset steps replace the Gaussian tensors before optimizer.step (no Adam update for
the Gaussians on those ~1 % of steps).
"""

import torch
from gsplat.strategy.ops import _update_param_with_optimizer, remove, split

from . import bank as bank_mod
from .filters import compute_filter_3d, reset_logits
from .renderer import scaled_camera
from .schedule import (
    densify_thresholds,
    expon_lr,
    phase_iteration,
    resolution_divisor,
    step_actions,
)

DEFAULTS = {
    # official per-scene script keys (variants.py)
    "init_dgt": 0.0006,
    "total_steps": 60000,
    "ms_steps": 6000,
    "min_opacity": 0.1,
    "init_opacity": -1.0,
    "use_another_mlp": False,
    "bpn_lr": 5e-4,
    # official arguments/__init__.py defaults
    "densify_grad_threshold": 0.0002,
    "densify_from_iter": 500,
    "densify_until_iter": 35000,
    "densification_interval": 100,
    "opacity_reset_interval": 3000,
    "opacity_reset_value": 0.01,
    "percent_dense": 0.01,
    "position_lr_init": 0.00016,
    "position_lr_final": 0.0000016,
    "position_lr_delay_mult": 0.01,
    "position_lr_delay_steps": 0,
    "position_lr_max_steps": 60000,
    "filter_coef": 0.2,
    "filter_near": 0.2,
    "filter_margin": 0.15,
    "filter_interval": 100,
    "upsample_steps": [3000, 6000],
    "kernel_sizes": [5, 9, 17],
    "kernel_size_ss": 17,
    "bpn_start": 250,
    "not_use_rgbd": False,
    "not_use_pe": False,
}
REQUIRES = {"gradient.screen.signed"}


@torch.no_grad()
def append(params, optimizers, new):
    """densification_postfix: concatenate new Gaussians (zero Adam moments)."""
    n_new = len(new["means"])

    def param_fn(name, p):
        return torch.nn.Parameter(torch.cat([p, new[name].to(p.dtype)]),
                                  requires_grad=p.requires_grad)

    def optimizer_fn(key, v):
        return torch.cat([v, torch.zeros((n_new, *v.shape[1:]), device=v.device)])

    _update_param_with_optimizer(param_fn, optimizer_fn, params, optimizers)


@torch.no_grad()
def densify_and_prune(params, optimizers, grads, thr, min_opacity, extent, percent_dense,
                      world_prune=False):
    """gaussian_model.densify_and_prune (clone, split on the post-clone population, prune)."""
    dense = percent_dense * extent
    smax = params["scales"].exp().max(dim=1).values
    sel = torch.where((grads >= thr) & (smax <= dense))[0]
    append(params, optimizers, {k: v[sel] for k, v in params.items()})
    n_clone = len(sel)
    g_pad = torch.cat([grads, torch.zeros(n_clone, device=grads.device)])
    smax = params["scales"].exp().max(dim=1).values
    split_mask = (g_pad >= thr) & (smax > dense)
    n_split = int(split_mask.sum())
    split(params, optimizers, {}, split_mask)
    prune = params["opacities"].sigmoid() < min_opacity
    if world_prune:
        prune |= params["scales"].exp().max(dim=1).values > 0.1 * extent
    n_prune = int(prune.sum())
    remove(params, optimizers, {}, prune)
    return {"clone": n_clone, "split": n_split, "prune": n_prune}


class Density:
    requires = REQUIRES

    def __init__(self, spec):
        unknown = set(spec.config) - set(DEFAULTS)
        if unknown:
            raise ValueError(f"Unknown BAGS density options: {sorted(unknown)}")
        self.cfg = {**DEFAULTS, **spec.config}
        c = self.cfg
        c["upsample_steps"] = [int(s) for s in c["upsample_steps"]]
        if c["ms_steps"] < c["upsample_steps"][1]:  # official: assert opt.ms_steps >= 6000
            raise ValueError("ms_steps must be >= the last upsample step")
        if int(c["kernel_size_ss"]) != int(c["kernel_sizes"][2]) and not c["use_another_mlp"]:
            raise ValueError("kernel_size_ss != K1 implies use_another_mlp (official train.py)")
        self.requires = set(REQUIRES)
        self.state = {}
        self.stats = None
        self.extent = None
        self.params = None
        self.cameras = None
        self.bank = None
        self.counts = {"filter_updates": 0, "resets": 0, "densify": 0, "clone": 0, "split": 0,
                       "prune": 0}

    # -- runtime hooks -------------------------------------------------------------------
    def initialize(self, params, optimizers, scene_scale=1.0):
        self.extent = float(scene_scale)  # 3dgs recipe: getNerfppNorm radius
        self.params = params
        optimizers["means"].param_groups[0]["lr"] = self.xyz_lr(1)
        self._reset_stats(len(params["means"]), params["means"].device)
        self.state = {"extent": self.extent, "counts": self.counts}
        if self.cameras:
            self._setup()

    def set_training_cameras(self, cameras):
        cameras = list(cameras)
        if not cameras:
            raise ValueError("BAGS needs the training cameras")
        self.cameras = cameras
        if self.params is not None:
            self._setup()

    def _setup(self):
        dev = self.params["means"].device
        self.cameras = [{k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in cam.items()}
                        for cam in self.cameras]
        self.update_filter(self.params, 1)  # official compute_3D_filter(scale-4 cameras)
        self.bank = bank_mod.BlurBank(self.cameras, self.cfg, dev)
        bank_mod.ACTIVE = self.bank
        self.state["bank"] = self.bank.export()

    def before_backward(self, params, optimizers, step, outputs):
        if self.bank is None:
            raise RuntimeError("BAGS needs density.set_training_cameras(...) first")
        outputs["info"]["means2d"].retain_grad()

    @torch.no_grad()
    def after_backward(self, params, optimizers, step, outputs):
        c = self.cfg
        self.bank.after_step(step)
        act = step_actions(step, c)
        if act["stats"]:
            self.accumulate(outputs["info"])
        if act["densify"]:
            thr, opa = densify_thresholds(step, c)
            world = opa == 0.005 and act["iteration"] > c["opacity_reset_interval"]
            grads = self.stats["grad_accum"] / self.stats["denom"]
            grads[grads.isnan()] = 0.0
            n = densify_and_prune(params, optimizers, grads, thr, opa, self.extent,
                                  c["percent_dense"], world)
            for k, v in n.items():
                self.counts[k] += v
            self.counts["densify"] += 1
            self._reset_stats(len(params["means"]), params["means"].device)
        if act["filter"]:
            self.update_filter(params, step)
        if act["reset"]:
            self.reset_opacity(params, optimizers)
        up = c["upsample_steps"]
        if (step < c["total_steps"]
                and resolution_divisor(step + 1, up) != resolution_divisor(step, up)):
            self.update_filter(params, step + 1)  # official: at the resolution change
        optimizers["means"].param_groups[0]["lr"] = self.xyz_lr(step + 1)
        if step % 1000 == 0 or step == c["total_steps"]:
            self.state["bank"] = self.bank.export()

    # -- pieces ----------------------------------------------------------------------------
    def xyz_lr(self, step):
        c = self.cfg
        it = phase_iteration(step, c["ms_steps"])
        return expon_lr(it, c["position_lr_init"] * self.extent,
                        c["position_lr_final"] * self.extent, c["position_lr_delay_steps"],
                        c["position_lr_delay_mult"], c["position_lr_max_steps"])

    def _reset_stats(self, n, dev):
        self.stats = {"grad_accum": torch.zeros(n, device=dev), "denom": torch.zeros(n, device=dev)}

    def accumulate(self, info):
        grad = info["means2d"].grad
        if grad is None:
            raise RuntimeError("BAGS density: means2d has no gradient")
        grad = grad[0].clone()
        grad[..., 0] *= info["width"] / 2.0
        grad[..., 1] *= info["height"] / 2.0
        radii = info["radii"][0]
        vis = (radii > 0).all(dim=-1) if radii.dim() == 2 else radii > 0
        if len(vis) != len(self.stats["denom"]):
            raise RuntimeError("BAGS density: statistics and population are misaligned")
        self.stats["grad_accum"][vis] += torch.linalg.vector_norm(grad[vis], dim=-1)
        self.stats["denom"][vis] += 1

    def cameras_at(self, step):
        d = resolution_divisor(step, self.cfg["upsample_steps"])
        return [scaled_camera(cam, d) for cam in self.cameras]

    def update_filter(self, params, step):
        c = self.cfg
        f = compute_filter_3d(params["means"], self.cameras_at(step), c["filter_coef"],
                              c["filter_near"], c["filter_margin"])
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


def create_density(spec):
    return Density(spec)
