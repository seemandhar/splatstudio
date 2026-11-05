"""DNGaussian training loop pieces that live outside the photometric pass (official
train_llff.py, scene/gaussian_model.py), owned by the density component.

Per official iteration t the code runs: hard-depth pass + optimizer step, soft-depth pass +
step (t > 1000), photometric pass, invisible-Gaussian cleaning (t in {1, 6000}), densification
(t < 6000), near-camera pruning ((t-1) % 25 == 0, t > 2000), optimizer step. The native runtime
renders / backpropagates / steps the photometric pass of step s itself; ``after_backward(s)``
then performs, in order:
  1. the Neural Color Renderer's Adam step (its gradients come from the photometric pass);
  2. the post-photometric bookkeeping of iteration s (clean_views / densify / near prune);
  3. the hard and soft depth passes of iteration s+1 (each its own backward + Adam step of
     ``means`` resp. ``opacities``), on the camera of step s (cameras are i.i.d. random, so the
     pairing is equivalent in distribution; the hard pass of iteration 1 is the one pass lost);
  4. the xyz learning rate of iteration s+1: expon(0.016 -> 0.00016, 5500 steps) x extent,
     evaluated at max(t - 500, 0) (official ``position_lr_start``);
  5. a bake of the inference state every ``bake_every`` steps and at ``total_steps``.

Densification (gaussian_model.py:454-524): grad statistics |dL/d NDC xy| of the photometric
pass for visible Gaussians, accumulated from iteration 1 (except clean iterations); at
t > 500, t % 100 == 0, t < 6000: clone (grad >= 0.0013, max scale <= 0.01 extent,
opacity > 0.1), split (padded grad >= 0.0013, max scale > 0.01 extent, opacity > 0.1; two
samples, scale / 1.6), prune opacity < 0.01; no screen-size pruning, no opacity reset.
"""

import math
import random

import torch
from gsplat.strategy.ops import duplicate, remove, split

from . import neural
from .depth import DEPTH_DEFAULTS, DepthTable, data_root, depth_regularization
from .renderer import hard_depth, inria_radii, soft_depth

DEFAULTS = {
    "total_steps": 6000,
    # densification (run_llff.sh)
    "grad_threshold": 0.0013, "densify_from_iter": 500, "densify_until_iter": 6000,
    "densification_interval": 100, "percent_dense": 0.01, "split_opacity_thresh": 0.1,
    "prune_threshold": 0.01,
    # learning rates
    "position_lr_init": 0.016, "position_lr_final": 0.00016, "position_lr_max_steps": 5500,
    "position_lr_start": 500, "scaling_lr": 0.003, "opacity_lr": 0.05, "rotation_lr": 0.001,
    # depth regularization
    "hard_depth": True, "soft_depth": True, "hard_depth_start": 0, "soft_depth_start": 1000,
    "tau": 0.95, **DEPTH_DEFAULTS,
    # pruning
    "near": 10.0, "near_every": 25, "near_start": 2000, "clean_first": True, "clean_last": True,
    # rasterizer / bake
    "near_plane": 0.2, "bake_every": 1000, "seed": 0,
}


def expon_lr(step, lr_init, lr_final, max_steps):
    """Official get_expon_lr_func with lr_delay_steps = 0."""
    if step < 0:
        return 0.0
    t = min(max(step / max_steps, 0.0), 1.0)
    return math.exp(math.log(lr_init) * (1 - t) + math.log(lr_final) * t)


class DNGaussianDensity:
    requires = frozenset({"gradient.screen.signed"})

    def __init__(self, spec):
        unknown = set(spec.config) - set(DEFAULTS)
        if unknown:
            raise ValueError(f"Unknown DNGaussian density options: {sorted(unknown)}")
        self.cfg = {**DEFAULTS, **spec.config}
        self.state = {"grad2d": None, "count": None, "log": []}
        self.extent = 1.0
        self.net = None
        self.net_opt = None
        self.cameras = None
        self.table = None  # DepthTable, loaded lazily from the data root (probes may inject one)
        self.rng = random.Random(int(self.cfg["seed"]))

    # ------------------------------------------------------------------ setup
    def initialize(self, params, optimizers, scene_scale=1.0):
        cfg = self.cfg
        self.extent = float(scene_scale)
        self.net = neural.GridRenderer.from_points(params["means"])
        self.net_opt = self.net.optimizer()
        neural.ACTIVE.update(net=self.net, baked=False)
        optimizers["scales"].param_groups[0]["lr"] = float(cfg["scaling_lr"])
        optimizers["opacities"].param_groups[0]["lr"] = float(cfg["opacity_lr"])
        optimizers["quats"].param_groups[0]["lr"] = float(cfg["rotation_lr"])
        optimizers["means"].param_groups[0]["lr"] = self.means_lr(1)
        self._reset_stats(len(params["means"]), params["means"].device)

    def set_training_cameras(self, cameras):
        self.cameras = cameras

    def means_lr(self, iteration):
        cfg = self.cfg
        return expon_lr(max(iteration - int(cfg["position_lr_start"]), 0),
                        float(cfg["position_lr_init"]), float(cfg["position_lr_final"]),
                        int(cfg["position_lr_max_steps"])) * self.extent

    def _reset_stats(self, n, device):
        self.state["grad2d"] = torch.zeros(n, device=device)
        self.state["count"] = torch.zeros(n, device=device)

    def depth_table(self):
        if self.table is None:
            self.table = DepthTable.load(data_root())
        return self.table

    # ------------------------------------------------------------------ hooks
    def before_backward(self, params, optimizers, step, outputs):
        outputs["info"]["means2d"].retain_grad()

    def after_backward(self, params, optimizers, step, outputs):
        cfg = self.cfg
        total = int(cfg["total_steps"])
        row = {"step": step}
        # 1. neural colour renderer step (photometric gradients)
        self.net_opt.step()
        self.net_opt.zero_grad(set_to_none=True)
        # 2. post-photometric bookkeeping of iteration `step`
        clean = (step == 1 and cfg["clean_first"]) or (step == total and cfg["clean_last"])
        if clean:
            row["cleaned"] = self.clean_views(params, optimizers)
        if step < int(cfg["densify_until_iter"]) and not clean:
            self.accumulate(outputs["info"])
            if (step > int(cfg["densify_from_iter"])
                    and step % int(cfg["densification_interval"]) == 0):
                row.update(self.densify_and_prune(params, optimizers))
        if (step - 1) % int(cfg["near_every"]) == 0 and step > int(cfg["near_start"]):
            row["near_pruned"] = self.near_prune(params, optimizers)
        # 3. depth passes of iteration step + 1
        nxt = step + 1
        if nxt <= total:
            optimizers["means"].param_groups[0]["lr"] = self.means_lr(nxt)
            camera = outputs["camera"]
            mono = None
            if cfg["hard_depth"] and nxt > int(cfg["hard_depth_start"]):
                mono = self.depth_table().lookup(camera["viewmat"], params["means"].device)
                row["hard"] = self.depth_pass(params, optimizers, camera, mono, nxt, "hard")
            if cfg["soft_depth"] and nxt > int(cfg["soft_depth_start"]):
                if mono is None:
                    mono = self.depth_table().lookup(camera["viewmat"], params["means"].device)
                row["soft"] = self.depth_pass(params, optimizers, camera, mono, nxt, "soft")
        # 4. xyz lr for the photometric pass of iteration step + 1
        optimizers["means"].param_groups[0]["lr"] = self.means_lr(min(nxt, total))
        # 5. bake the inference state
        if step % int(cfg["bake_every"]) == 0 or step == total:
            neural.bake(params, self.net)
            neural.ACTIVE["baked"] = True
            row["baked"] = len(params["means"])
        self.last_row = row  # observability for probes; not used by the method
        if len(row) > 1 and (step % 200 == 0 or step == 1 or "baked" in row or step == total):
            row["gaussians"] = len(params["means"])
            self.state["log"].append(row)
            print({"dngaussian": row}, flush=True)

    # ------------------------------------------------------------------ depth passes
    def depth_pass(self, params, optimizers, camera, mono, iteration, kind):
        near = float(self.cfg["near_plane"])
        if kind == "hard":
            depth, _, _ = hard_depth(params, camera, float(self.cfg["tau"]), near)
            field = "means"
        else:
            depth, _, _ = soft_depth(params, camera, near)
            field = "opacities"
        loss, terms = depth_regularization(depth, mono, iteration, self.rng, self.cfg)
        if not torch.isfinite(loss):
            raise RuntimeError(f"DNGaussian {kind} depth loss is not finite: {terms}")
        loss.backward()
        stray = [k for k, p in params.items() if k != field and p.grad is not None]
        if stray:
            raise RuntimeError(f"{kind} depth pass leaked gradients into {stray}")
        optimizers[field].step()
        optimizers[field].zero_grad(set_to_none=True)
        return {"loss": float(loss), **terms}

    # ------------------------------------------------------------------ population control
    def accumulate(self, info):
        means2d = info["means2d"]
        if means2d.grad is None:
            return
        n = means2d.shape[-2]
        if self.state["grad2d"] is None or len(self.state["grad2d"]) != n:
            self._reset_stats(n, means2d.device)
        grads = means2d.grad.clone()
        grads[..., 0] *= info["width"] / 2.0 * info["n_cameras"]
        grads[..., 1] *= info["height"] / 2.0 * info["n_cameras"]
        sel = (info["visibility_radii"] > 0).all(dim=-1)
        ids = torch.where(sel)[1]
        self.state["grad2d"].index_add_(0, ids, grads[sel].norm(dim=-1))
        self.state["count"].index_add_(0, ids, torch.ones_like(ids, dtype=torch.float32))

    @torch.no_grad()
    def densify_and_prune(self, params, optimizers):
        cfg, st = self.cfg, self.state
        tau, pd = float(cfg["grad_threshold"]), float(cfg["percent_dense"])
        opa_t = float(cfg["split_opacity_thresh"])
        grads = st["grad2d"] / st["count"]
        grads[grads.isnan()] = 0.0
        big = params["scales"].exp().max(dim=1).values > pd * self.extent
        opaque = params["opacities"].sigmoid() > opa_t
        is_dup = (grads >= tau) & ~big & opaque
        n_dup = int(is_dup.sum())
        if n_dup:
            duplicate(params, optimizers, st, is_dup)
        padded = torch.cat([grads, torch.zeros(n_dup, device=grads.device)])
        big = params["scales"].exp().max(dim=1).values > pd * self.extent
        opaque = params["opacities"].sigmoid() > opa_t
        is_split = (padded >= tau) & big & opaque
        n_split = int(is_split.sum())
        if n_split:
            split(params, optimizers, st, is_split)
        n_prune = self._remove(params, optimizers,
                               params["opacities"].sigmoid() < float(cfg["prune_threshold"]))
        self._reset_stats(len(params["means"]), params["means"].device)
        return {"cloned": n_dup, "split": n_split, "pruned": n_prune}

    def _remove(self, params, optimizers, mask):
        n = int(mask.sum())
        if n:
            if n == len(mask):
                raise RuntimeError("DNGaussian pruning would remove every Gaussian")
            remove(params, optimizers, self.state, mask)
        return n

    @torch.no_grad()
    def visible_any(self, params):
        """Union over training views of radii > 0 (official clean_views)."""
        if not self.cameras:
            raise RuntimeError("DNGaussian clean_views needs the training cameras")
        vis = torch.zeros(len(params["means"]), dtype=torch.bool, device=params["means"].device)
        for cam in self.cameras:
            vis |= (inria_radii(params, cam, float(self.cfg["near_plane"]))[0] > 0).all(dim=-1)
        return vis

    @torch.no_grad()
    def clean_views(self, params, optimizers):
        return self._remove(params, optimizers, ~self.visible_any(params))

    @torch.no_grad()
    def near_prune(self, params, optimizers):
        """Prune Gaussians within `near` of any of the 180 official spiral camera centres."""
        centers = self.depth_table().spiral.to(params["means"].device)
        d = torch.cdist(params["means"].detach(), centers).min(dim=1).values
        return self._remove(params, optimizers, d < float(self.cfg["near"]))


def create_density(spec):
    return DNGaussianDensity(spec)
