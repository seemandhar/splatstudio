"""Official Gaussian Surfels density control and xyz LR (scene/gaussian_model.py, train.py).

  init       sidecar init.npz (prepare_dtu.py): scales log sqrt(knn3_d2 / 4), third -1e10;
             quats normal2rotation(random normals) (un-normalised, official) — copied into the
             state when the sidecar points equal the initial means
  extent     getNerfppNorm radius of the training cameras (set_training_cameras)
  stats      it > densify_from (200), visibility radii > 1:
               xyz   += ||dL/dmeans2D[:2]||, scale += sum(dL/d log s[:2]),
               opac  += raw opacity logit, denom += 1
  every max(100, N_train) its (200 < it < 15000; official densify_until 30000 never binds, and
  its it-15000 densify happens after the model is saved, so it is excluded here):
    adaptive_prune   opacity < 0.1 | max(s_xy) > 0.5 extent | s_min s_max < 1e-8 extent^2 |
                     invisible (see below)
    adaptive_densify clone:  g >= 1e-4, max(s) <= 0.01 extent, mean scale-grad <= 1e-7,
                             mean opacity logit <= 2
                     split:  g >= 1e-4, max(s) > 0.01 extent; 2 in-plane samples
                             N(0, diag(s_x, s_y, 0)), log(s / 1.6), third scale -1e10,
                             originals removed
                     statistics restart from zero (densification_postfix)
  reset      (it - 1) % 1000 == 0, it > 200: o <- o * 0.12, opacity Adam moments zeroed
  xyz lr     get_expon_lr_func(1.6e-4 extent, 1.6e-6 extent, max_steps 15000), no delay
No screen-size prune (official adaptive_prune has none; max_radii2D is tracked but unused).

Invisible-point prune ("points that do not receive gradients every N iterations", §3.5): the
official code prunes denom == 0 over the last 100 iterations, during which its per-epoch camera
stack shows essentially every training view. The native runtime samples views with
replacement, so a fixed 100-step window would miss views (p ~ 0.08 per view for N = 42) and
prune Gaussians seen only from them. Here the visibility window instead lasts until every
training view has been rendered at least once; the prune at a densification step only fires
when the window is complete, then the window restarts.
"""

import math

import torch
import torch.nn.functional as F
from gsplat.strategy.ops import _update_param_with_optimizer, duplicate, remove
from gsplat.utils import normalized_quat_to_rotmat

from . import sidecar
from .renderer import view_key

DEFAULTS = {
    "densify_from_iter": 200,
    "densify_until_iter": 15000,  # official 30000; its it-15000 densify runs after the save
    "densification_interval": 100,
    "densify_grad_threshold": 0.0001,
    "percent_dense": 0.01,
    "min_opacity": 0.1,
    "big_world_factor": 0.5,
    "tiny_area_factor": 1e-8,
    "clone_scale_grad_max": 1e-7,
    "clone_opacity_logit_max": 2.0,
    "opacity_reset_interval": 1000,
    "reset_ratio": 0.12,
    "prune_invisible": True,
    "position_lr_init": 1.6e-4,
    "position_lr_final": 1.6e-6,
    "position_lr_delay_mult": 0.01,
    "position_lr_delay_steps": 0,
    "position_lr_max_steps": 15000,
    "use_sidecar_init": True,
}
STAT_KEYS = ("xyz_accum", "scale_accum", "opac_accum", "denom")


def expon_lr(step, lr_init, lr_final, delay_steps=0, delay_mult=1.0, max_steps=15000):
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


def nerfpp_radius(viewmats):
    centers = torch.stack([torch.linalg.inv(v.float())[:3, 3] for v in viewmats])
    return 1.1 * float(torch.linalg.vector_norm(centers - centers.mean(0), dim=1).max())


@torch.no_grad()
def split_in_plane(params, optimizers, state, mask, n=2):
    """densify_and_split: samples ~ N(0, diag(s)) rotated, log(s / (0.8 n)), z scale -1e10."""
    sel = torch.where(mask)[0]
    rest = torch.where(~mask)[0]
    s = params["scales"][sel].exp().repeat(n, 1)
    samples = torch.normal(mean=torch.zeros_like(s), std=s)
    rots = normalized_quat_to_rotmat(F.normalize(params["quats"][sel], dim=-1)).repeat(n, 1, 1)
    new_xyz = torch.bmm(rots, samples[..., None])[..., 0] + params["means"][sel].repeat(n, 1)
    new_scales = torch.log(s / (0.8 * n))
    new_scales[:, -1] = -1e10

    def param_fn(name, p):
        if name == "means":
            new = new_xyz
        elif name == "scales":
            new = new_scales
        else:
            new = p[sel].repeat([n] + [1] * (p.dim() - 1))
        return torch.nn.Parameter(torch.cat([p[rest], new]), requires_grad=p.requires_grad)

    def optimizer_fn(key, v):
        return torch.cat([v[rest], torch.zeros((n * len(sel), *v.shape[1:]), device=v.device,
                                               dtype=v.dtype)])

    _update_param_with_optimizer(param_fn, optimizer_fn, params, optimizers)
    for k, v in state.items():
        if isinstance(v, torch.Tensor) and v.shape[:1] == mask.shape:
            state[k] = torch.cat([v[rest], v[sel].repeat([n] + [1] * (v.dim() - 1))])


@torch.no_grad()
def scale_opacity(params, optimizers, ratio):
    """reset_opacity: inverse_sigmoid(sigmoid(o) * ratio), Adam moments zeroed."""
    def param_fn(name, p):
        return torch.nn.Parameter(torch.logit(p.sigmoid() * ratio), requires_grad=p.requires_grad)

    def optimizer_fn(key, v):
        return torch.zeros_like(v)

    _update_param_with_optimizer(param_fn, optimizer_fn, params, optimizers, names=["opacities"])


class Density:
    requires = {"gradient.gsurfels"}

    def __init__(self, spec):
        unknown = set(spec.config) - set(DEFAULTS)
        if unknown:
            raise ValueError(f"Unknown Gaussian Surfels density options: {sorted(unknown)}")
        self.cfg = {**DEFAULTS, **spec.config}
        self.state = None
        self.extent = None
        self.train_keys = None
        self.interval = int(self.cfg["densification_interval"])
        self._hook = None
        self._scale_grad = None

    def _zero_stats(self, n, dev):
        for k in STAT_KEYS:
            self.state[k] = torch.zeros(n, device=dev)

    def initialize(self, params, optimizers, scene_scale=1.0):
        n = len(params["means"])
        dev = params["means"].device
        self.extent = float(scene_scale)
        self.state = {"seen": torch.zeros(n, dtype=torch.bool, device=dev),
                      "extent": self.extent, "init_from_sidecar": False}
        self.views_seen = set()  # kept off the checkpointed state (weights_only torch.load)
        self._zero_stats(n, dev)
        tab = sidecar.table() if self.cfg["use_sidecar_init"] else None
        init = tab.init() if tab is not None else None
        if init is not None and len(init["points"]) == n and torch.allclose(
                init["points"].to(dev), params["means"].detach(), atol=1e-6):
            with torch.no_grad():
                params["scales"].copy_(init["log_scales"].to(dev))
                params["quats"].copy_(init["quats"].to(dev))
            self.state["init_from_sidecar"] = True

    def set_training_cameras(self, cameras):
        """Runtime hook: getNerfppNorm extent and the view keys of the visibility window."""
        self.extent = nerfpp_radius([c["viewmat"] for c in cameras])
        self.state["extent"] = self.extent
        self.train_keys = {view_key(c["viewmat"]) for c in cameras}
        self.interval = max(int(self.cfg["densification_interval"]), len(cameras))

    def xyz_lr(self, step):
        c = self.cfg
        return expon_lr(step, c["position_lr_init"] * self.extent,
                        c["position_lr_final"] * self.extent, c["position_lr_delay_steps"],
                        c["position_lr_delay_mult"], c["position_lr_max_steps"])

    def before_backward(self, params, optimizers, step, outputs):
        if step == 1:
            optimizers["means"].param_groups[0]["lr"] = self.xyz_lr(1)
        if self._hook is not None:
            self._hook.remove()
        self._scale_grad = None

        def hook(g):
            self._scale_grad = g[:, :2].sum(1).detach()

        self._hook = params["scales"].register_hook(hook)

    def window_complete(self):
        if self.train_keys is None:
            return True
        return self.train_keys <= self.views_seen

    @torch.no_grad()
    def after_backward(self, params, optimizers, step, outputs):
        c, st = self.cfg, self.state
        if self._hook is not None:
            self._hook.remove()
            self._hook = None
        if step > c["densify_from_iter"]:
            info = outputs["info"]
            vis = info["radii"] > 1
            g = info["means2d"].grad
            st["xyz_accum"][vis] += torch.linalg.vector_norm(g[vis, :2], dim=-1)
            if self._scale_grad is not None:
                st["scale_accum"][vis] += self._scale_grad[vis]
            st["opac_accum"][vis] += params["opacities"][vis]
            st["denom"][vis] += 1
            st["seen"][vis] = True
            self.views_seen.add(view_key(outputs["camera"]["viewmat"]))
            if step % self.interval == 0 and step < c["densify_until_iter"]:
                self.adaptive_prune(params, optimizers)
                self.adaptive_densify(params, optimizers)
                torch.cuda.empty_cache()
            if (step - 1) % c["opacity_reset_interval"] == 0:
                scale_opacity(params, optimizers, c["reset_ratio"])
        optimizers["means"].param_groups[0]["lr"] = self.xyz_lr(step + 1)

    @torch.no_grad()
    def adaptive_prune(self, params, optimizers):
        c, st = self.cfg, self.state
        s = params["scales"][:, :2].exp()
        smin, smax = s.min(1).values, s.max(1).values
        prune_opac = params["opacities"].sigmoid() < c["min_opacity"]
        prune_scale = (smax > c["big_world_factor"] * self.extent) | (
            smin * smax < c["tiny_area_factor"] * self.extent**2)
        invisible = torch.zeros_like(prune_opac)
        complete = self.window_complete()
        if c["prune_invisible"] and complete:
            invisible = ~st["seen"]
        prune = prune_opac | prune_scale | invisible
        remove(params, optimizers, st, prune)
        if complete:
            st["seen"] = torch.zeros(len(params["means"]), dtype=torch.bool,
                                     device=params["means"].device)
            self.views_seen = set()
        st["last"] = {"opacity": int(prune_opac.sum()), "scale": int(prune_scale.sum()),
                      "invisible": int(invisible.sum()), "window_complete": bool(complete)}

    @torch.no_grad()
    def adaptive_densify(self, params, optimizers):
        c, st = self.cfg, self.state
        den = st["denom"]
        grads = st["xyz_accum"] / den
        grad_scale = st["scale_accum"] / den
        grad_opac = st["opac_accum"] / den
        for t in (grads, grad_scale, grad_opac):
            t[t.isnan()] = 0.0
        pre_mask = (grad_scale <= c["clone_scale_grad_max"]) & (
            grad_opac <= c["clone_opacity_logit_max"])
        dense = c["percent_dense"] * self.extent
        smax = params["scales"].exp().max(dim=1).values
        thr = c["densify_grad_threshold"]
        clone = (grads.abs() >= thr) & (smax <= dense) & pre_mask
        n0 = len(grads)
        duplicate(params, optimizers, st, clone)
        split = torch.zeros(len(params["means"]), dtype=torch.bool, device=grads.device)
        smax = params["scales"].exp().max(dim=1).values
        split[:n0] = grads >= thr
        split &= smax > dense
        split_in_plane(params, optimizers, st, split)
        self._zero_stats(len(params["means"]), params["means"].device)
        st["last"].update({"before": n0, "clone": int(clone.sum()), "split": int(split.sum()),
                           "after": len(params["means"])})


def create_density(spec):
    return Density(spec)
