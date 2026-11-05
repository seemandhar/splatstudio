"""Official FatesGS / 2DGS adaptive density control and xyz learning-rate schedule.

Port of scene/gaussian_model.py (densify_and_clone / densify_and_split / densify_and_prune /
reset_opacity / add_densification_stats) and the train.py schedule:
  stats      while step < densify_until: accum += |d L / d means2D| (all 3 components) for
             radii > 0, denom += 1, max_radii = max(max_radii, radii)
  densify    step > densify_from and step % interval == 0 (step < densify_until):
             clone  grad >= thr and max(s1,s2) <= percent_dense * extent
             split  grad >= thr and max(s1,s2) >  percent_dense * extent: 2 samples
                    N(0, diag(s1, s2, 0)) in the surfel plane, scales / (0.8 * 2), originals removed
             prune  opacity < min_opacity; after opacity_reset_interval also radii > 20 px or
                    max(s1,s2) > 0.1 * extent; statistics reset
  reset      step % opacity_reset_interval == 0 (step < densify_until): o = min(o, 0.1)
  xyz lr     get_expon_lr_func(1.6e-4 * extent, 1.6e-6 * extent, max_steps 30000) — the official
             schedule ends at 1.6e-5 * extent after 15000 steps; it overrides the runtime's
             0.01**(step/steps) decay (set here for the next step's optimizer update).
The scales field keeps three columns (runtime state); the third is unused by the 2D rasterizer
and is never read here.
"""

import math

import torch
import torch.nn.functional as F
from gsplat.strategy.ops import _update_param_with_optimizer, duplicate, remove, reset_opa
from gsplat.utils import normalized_quat_to_rotmat

DEFAULTS = {
    "densify_from_iter": 500,
    "densify_until_iter": 15000,
    "densification_interval": 100,
    "densify_grad_threshold": 0.0002,
    "percent_dense": 0.01,
    "min_opacity": 0.1,          # opt.opacity_cull
    "opacity_reset_interval": 1000,
    "reset_value": 0.1,          # reset_opacity: min(o, 0.1)
    "max_screen_size": 20,
    "big_world_factor": 0.1,
    "position_lr_init": 1.6e-4,
    "position_lr_final": 1.6e-6,
    "position_lr_delay_mult": 0.01,
    "position_lr_delay_steps": 0,
    "position_lr_max_steps": 30000,
}


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
def split_in_plane(params, optimizers, state, mask, n=2):
    """densify_and_split: in-plane samples, scale / (0.8 n), originals removed."""
    sel = torch.where(mask)[0]
    rest = torch.where(~mask)[0]
    s2 = params["scales"][sel, :2].exp().repeat(n, 1)
    stds = torch.cat([s2, torch.zeros_like(s2[:, :1])], dim=-1)
    samples = torch.normal(mean=torch.zeros_like(stds), std=stds)
    rots = normalized_quat_to_rotmat(F.normalize(params["quats"][sel], dim=-1)).repeat(n, 1, 1)
    new_xyz = torch.bmm(rots, samples[..., None])[..., 0] + params["means"][sel].repeat(n, 1)
    new_scales = params["scales"][sel].repeat(n, 1).clone()
    new_scales[:, :2] = torch.log(s2 / (0.8 * n))

    def param_fn(name, p):
        if name == "means":
            new = new_xyz
        elif name == "scales":
            new = new_scales
        else:
            new = p[sel].repeat([n] + [1] * (p.dim() - 1))
        return torch.nn.Parameter(torch.cat([p[rest], new]), requires_grad=p.requires_grad)

    def optimizer_fn(key, v):
        return torch.cat([v[rest], torch.zeros((n * len(sel), *v.shape[1:]), device=v.device)])

    _update_param_with_optimizer(param_fn, optimizer_fn, params, optimizers)
    for k, v in state.items():
        if isinstance(v, torch.Tensor) and v.shape[:1] == mask.shape:
            state[k] = torch.cat([v[rest], v[sel].repeat([n] + [1] * (v.dim() - 1))])


class Density:
    requires = {"gradient.surfel"}

    def __init__(self, spec):
        unknown = set(spec.config) - set(DEFAULTS)
        if unknown:
            raise ValueError(f"Unknown FatesGS density options: {sorted(unknown)}")
        self.cfg = {**DEFAULTS, **spec.config}
        self.state = None
        self.extent = None

    def initialize(self, params, optimizers, scene_scale=1.0):
        n = len(params["means"])
        dev = params["means"].device
        self.extent = float(scene_scale)  # 3dgs recipe: getNerfppNorm radius (spatial_lr_scale)
        self.state = {"grad_accum": torch.zeros(n, device=dev), "denom": torch.zeros(n, device=dev),
                      "max_radii": torch.zeros(n, device=dev), "extent": self.extent}

    def xyz_lr(self, step):
        c = self.cfg
        return expon_lr(step, c["position_lr_init"] * self.extent,
                        c["position_lr_final"] * self.extent, c["position_lr_delay_steps"],
                        c["position_lr_delay_mult"], c["position_lr_max_steps"])

    def before_backward(self, params, optimizers, step, outputs):
        if step == 1:
            optimizers["means"].param_groups[0]["lr"] = self.xyz_lr(1)

    @torch.no_grad()
    def after_backward(self, params, optimizers, step, outputs):
        c = self.cfg
        st = self.state
        if step < c["densify_until_iter"]:
            info = outputs["info"]
            radii = info["radii"].float()
            vis = radii > 0
            grad = info["means2d"].grad
            st["max_radii"][vis] = torch.maximum(st["max_radii"][vis], radii[vis])
            st["grad_accum"][vis] += torch.linalg.vector_norm(grad[vis], dim=-1)
            st["denom"][vis] += 1
            if step > c["densify_from_iter"] and step % c["densification_interval"] == 0:
                size = c["max_screen_size"] if step > c["opacity_reset_interval"] else None
                self.densify_and_prune(params, optimizers, size)
            if step % c["opacity_reset_interval"] == 0:
                reset_opa(params, optimizers, st, value=c["reset_value"])
        optimizers["means"].param_groups[0]["lr"] = self.xyz_lr(step + 1)

    @torch.no_grad()
    def densify_and_prune(self, params, optimizers, max_screen_size):
        c, st = self.cfg, self.state
        grads = st["grad_accum"] / st["denom"]
        grads[grads.isnan()] = 0.0
        smax = params["scales"][:, :2].exp().max(dim=1).values
        thr, dense = c["densify_grad_threshold"], c["percent_dense"] * self.extent
        clone = (grads >= thr) & (smax <= dense)
        split = (grads >= thr) & (smax > dense)
        n0 = len(grads)
        duplicate(params, optimizers, st, clone)
        split = torch.cat([split, torch.zeros(int(clone.sum()), dtype=torch.bool,
                                              device=split.device)])
        split_in_plane(params, optimizers, st, split)
        prune = params["opacities"].sigmoid() < c["min_opacity"]
        if max_screen_size:
            # Official quirk: densification_postfix (clone/split) zeroes max_radii2D before
            # densify_and_prune tests it (gaussian_model.py:344 vs :396), so the screen-size
            # prune never fires; only the world-size prune is active (iteration 1, found via
            # SparseGS where a real screen prune collapsed test PSNR).
            big_ws = params["scales"][:, :2].exp().max(dim=1).values > (
                c["big_world_factor"] * self.extent)
            prune = prune | big_ws
        remove(params, optimizers, st, prune)
        n = len(params["means"])
        dev = params["means"].device
        # densification_postfix / prune: statistics restart from zero for every Gaussian
        st["grad_accum"] = torch.zeros(n, device=dev)
        st["denom"] = torch.zeros(n, device=dev)
        st["max_radii"] = torch.zeros(n, device=dev)
        st["last"] = {"before": n0, "clone": int(clone.sum()), "split": int(split.sum()),
                      "prune": int(prune.sum()), "after": n}
        torch.cuda.empty_cache()


def create_density(spec):
    return Density(spec)
