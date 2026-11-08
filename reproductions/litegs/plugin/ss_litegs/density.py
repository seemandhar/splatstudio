"""LiteGS density control (arXiv 2503.01199 Sec. 3.3; official litegs/training/densify.py
DensityControllerTamingGS, trainer.py, optimizer.py, utils/statistic_helper.py).

Per densification epoch (epoch % 5 == 0, 3 <= epoch < until) the controller accumulates, over every
training view of that epoch, the per-Gaussian moments of the per-pixel opacity gradient
(S = sum g^2, M = sum g, C = #fragments) and the blend weight sum W = sum T*alpha (CUDA kernel).
At the end of that epoch:
  1. budgeted growth: budget = min(max(int(target(epoch) - N), 1) + #prunable, N) primitives are drawn
     without replacement from the opacity-gradient-variance score  var * C * sigmoid(o)^2;
     small ones are cloned, large ones get one split child (parent kept);
  2. weight prune: pre-existing primitives with W == 0;
  3. every 10 epochs, opacity decay sigmoid(o) <- max(sigmoid(o)/2, 1/128) and all Adam states are
     cleared (official `optimizer.state.clear()`).
The runtime's Adam optimizers are replaced by LiteAdam (visible-only, no bias correction) with
LiteGS/TamingGS learning rates. Population changes go through gsplat.strategy.ops.
"""

import json
import math

import torch
from gsplat.strategy.ops import _update_param_with_optimizer, remove
from gsplat.utils import normalized_quat_to_rotmat

from . import schedule
from .kernels import opacity_grad_moments
from .optim import LiteAdam, Visibility
from .renderer import META_CAPABILITY

# Initial SfM point counts of our native scenes -> LiteGS-quality targets (official full_eval.py:
# last entry of target_primitives_list = the 3DGS/TamingGS-big count of each scene).
SCENE_SFM_COUNTS = {
    "bicycle": 54275, "flowers": 38347, "garden": 138766, "stump": 32049, "treehill": 52363,
    "room": 112627, "counter": 155767, "kitchen": 241367, "bonsai": 206613,
    "truck": 136029, "train": 182686, "drjohnson": 80861, "playroom": 37005,
}
QUALITY_TARGETS = {
    "bicycle": 5987095, "flowers": 3618411, "garden": 5728191, "stump": 4867429,
    "treehill": 3770257, "room": 1548960, "counter": 1190919, "kitchen": 1803735,
    "bonsai": 1252367, "truck": 2584171, "train": 1085480, "playroom": 2326100,
    "drjohnson": 3273600,
}
TARGET_BY_COUNT = {str(SCENE_SFM_COUNTS[s]): v for s, v in QUALITY_TARGETS.items()}

DEFAULTS = {
    "total_steps": 30000,           # OptimizationParams.iterations
    "densify_from": 3,              # DensifyParams.densify_from (epochs)
    "densification_interval": 5,    # DensifyParams.densification_interval (epochs)
    "opacity_decay_interval": 10,   # DensifyParams.opacity_reset_interval (epochs)
    "until_fraction": 0.8,          # trainer.py densify_until rule
    "opacity_mode": "decay",        # 'decay' (LiteGS) or 'reset' (ablation, official 'reset')
    "decay_rate": 0.5,              # densify.py reset_opacity
    "decay_floor": 1.0 / 128,
    "clear_adam_on_decay": True,    # densify.py: optimizer.state.clear()
    "percent_dense": 0.01,          # DensifyParams.percent_dense
    "split_scale_div": 1.6,         # (0.8*2)
    "score": "variance",            # 'variance' (LiteGS) or 'magnitude' (ablation: M^2/C)
    "score_opacity_power": 2,       # get_score: * sigmoid(o)^2
    "target_primitives": None,      # None -> lookup by SfM count (LiteGS-quality)
    "target_table": TARGET_BY_COUNT,
    "lr": {"means": 1.6e-4, "sh0": 0.0025, "shN": 0.00025, "opacities": 0.025,
           "scales": 0.005, "quats": 0.001},
    "adam_sparse": True,
    "adam_bias_correction": False,
    "adam_eps": 1e-15,
    "seed": 0,
    "log": True,
}


def variance_score(sq_sum, grad_sum, count, opacity, power=2, kind="variance"):
    """statistic_helper.get_var + DensityControllerTamingGS.get_score."""
    mean = grad_sum / (count + 1)
    if kind == "variance":
        var = (sq_sum / (count + 1) - mean.square()).clamp_min(0)
    elif kind == "magnitude":
        var = mean.square()
    else:
        raise ValueError("score must be 'variance' or 'magnitude'")
    score = var * count * opacity.pow(power)
    return score.nan_to_num(0).clamp_min(0)


def growth_target(epoch, init_count, target, start, until):
    """split_and_clone: linear growth from the initial count to the target over [start, until)."""
    return (target - init_count) / (until - start) * (epoch - start) + init_count


def growth_budget(epoch, n, prune_num, init_count, target, start, until):
    cur = growth_target(epoch, init_count, target, start, until)
    return min(max(int(cur - n), 1) + int(prune_num), n)


class LiteDensity:
    def __init__(self, spec):
        unknown = set(spec.config) - set(DEFAULTS)
        if unknown:
            raise ValueError(f"Unknown LiteGS density options: {sorted(unknown)}")
        c = {**DEFAULTS, **spec.config}
        c["lr"] = {**DEFAULTS["lr"], **c["lr"]}
        if c["opacity_mode"] not in ("decay", "reset"):
            raise ValueError("opacity_mode must be 'decay' or 'reset'")
        self.cfg = c
        self.requires = {META_CAPABILITY}
        self.state = {}
        self.visibility = Visibility()
        self.generator = None
        self.n_train = None
        self.stats = None
        self.pending = None
        self.history = []

    # ------------------------------------------------------------------ lifecycle
    def initialize(self, params, optimizers, scene_scale=1.0):
        c = self.cfg
        self.extent = float(scene_scale)
        self.init_count = len(params["means"])
        if c["target_primitives"] is not None:
            self.target = int(c["target_primitives"])
        else:
            value = c["target_table"].get(str(self.init_count))
            # Unknown scenes fail at the first densification (not here, so that contract probes on
            # toy scenes can exercise the component); there is no silent default budget.
            self.target = None if value is None else int(value)
        for name in list(optimizers):
            old = optimizers[name]
            (p,) = old.param_groups[0]["params"]
            lr = c["lr"].get(name)
            if lr is None:
                lr = old.param_groups[0]["lr"]
            elif name == "means":
                lr = lr * self.extent  # runtime convention: means lr scaled by the scene extent
            optimizers[name] = LiteAdam([p], lr=lr, visibility=self.visibility,
                                        eps=c["adam_eps"], sparse=c["adam_sparse"],
                                        bias_correction=c["adam_bias_correction"])
        self.generator = torch.Generator(device=params["means"].device)
        self.generator.manual_seed(int(c["seed"]))
        self.state = {"target": self.target, "init_count": self.init_count}

    def set_training_cameras(self, cameras):
        self.n_train = len(cameras)
        schedule.bind(self.n_train)
        c = self.cfg
        self.total_epoch = schedule.total_epochs(c["total_steps"], self.n_train)
        self.until = schedule.densify_until(self.total_epoch, c["opacity_decay_interval"],
                                            c["until_fraction"])
        self.state.update(n_train=self.n_train, total_epoch=self.total_epoch, until=self.until)

    def _stats_epoch(self, epoch):
        c = self.cfg
        return schedule.is_densify_epoch(epoch, c["densify_from"], self.until,
                                         c["densification_interval"])

    def before_backward(self, params, optimizers, step, outputs):
        # The epoch schedule needs n_train from set_training_cameras(), which the native trainer
        # always calls before step 1; without it (toy contract probes) only the optimizer
        # visibility is maintained.
        outputs["info"]["means2d"].retain_grad()
        radii = outputs["info"]["radii"]
        vis = (radii > 0).all(dim=-1) if radii.dim() == 3 else radii > 0
        self.visibility.mask = vis[0]
        self.pending = None
        if self.n_train is not None and self._stats_epoch(schedule.epoch_of(step, self.n_train)):
            outputs["rgb_raw"].retain_grad()
            # Colours of this render, from the pre-update parameters (gsplat's SH path).
            self.pending = (outputs["rgb_raw"], outputs["info"],
                            self._colors(params, outputs["camera"], outputs["sh_degree"],
                                         vis[0]))

    @torch.no_grad()
    def _colors(self, params, camera, degree, mask):
        from gsplat import spherical_harmonics

        campos = torch.linalg.inv(camera["viewmat"])[:3, 3]
        dirs = params["means"] - campos[None]
        coeffs = torch.cat((params["sh0"], params["shN"]), dim=1)
        colors = spherical_harmonics(degree, dirs, coeffs, masks=mask)
        return torch.clamp_min(colors + 0.5, 0.0)

    @torch.no_grad()
    def after_backward(self, params, optimizers, step, outputs):
        if self.n_train is None:
            return
        schedule.SHARED["step"] = step
        epoch = schedule.epoch_of(step, self.n_train)
        if self.pending is not None:
            raw, info, colors = self.pending
            if raw.grad is None:
                raise RuntimeError("LiteGS statistics: rendered image has no gradient")
            m = opacity_grad_moments(info, colors, raw, raw.grad)
            if self.stats is None or len(self.stats["count"]) != len(colors):
                self.stats = {k: torch.zeros_like(v) for k, v in m.items()}
            for k, v in m.items():
                self.stats[k] += v
            self.pending = None
        if step % self.n_train != 0:
            return
        c = self.cfg
        changed = False
        if self._stats_epoch(epoch):
            self.densify(params, optimizers, epoch)
            changed = True
        if schedule.is_decay_epoch(epoch, c["densify_from"], self.until,
                                   c["opacity_decay_interval"]):
            self.decay(params, optimizers)
            changed = True
        if changed:
            self.stats = None
            check_trainable(params)

    # ------------------------------------------------------------------ operations
    @torch.no_grad()
    def densify(self, params, optimizers, epoch):
        c = self.cfg
        if self.target is None:
            raise ValueError(
                f"No LiteGS target for a scene with {self.init_count} initial points; set the "
                "density config 'target_primitives'"
            )
        n = len(params["means"])
        st = self.stats
        if st is None or len(st["count"]) != n:
            raise RuntimeError("LiteGS densify without statistics for the current population")
        prune_mask = st["weight"] == 0
        prune_num = int(prune_mask.sum())
        if prune_num > 0.8 * n:
            raise RuntimeError("LiteGS weight prune would remove >80% of the primitives")
        budget = growth_budget(epoch, n, prune_num, self.init_count, self.target,
                               c["densify_from"], self.until)
        score = variance_score(st["sq_sum"], st["sum"], st["count"],
                               params["opacities"].sigmoid(), c["score_opacity_power"],
                               c["score"])
        positive = int((score > 0).sum())
        budget = min(budget, positive)
        n_clone = n_split = 0
        if budget > 0:
            idx = torch.multinomial(score, budget, replacement=False, generator=self.generator)
            big = params["scales"][idx].exp().max(dim=-1).values > c["percent_dense"] * self.extent
            split_idx, clone_idx = idx[big], idx[~big]
            n_clone, n_split = len(clone_idx), len(split_idx)
            append_new(params, optimizers, self.new_rows(params, split_idx, clone_idx))
        keep_new = torch.zeros(len(params["means"]) - n, dtype=torch.bool,
                               device=prune_mask.device)
        if prune_num:
            remove(params, optimizers, {}, torch.cat([prune_mask, keep_new]))
        row = {"litegs_densify": {"epoch": epoch, "n_before": n, "budget": budget,
                                  "clone": n_clone, "split": n_split, "prune": prune_num,
                                  "n_after": len(params["means"]),
                                  "target_epoch": growth_target(epoch, self.init_count,
                                                                self.target, c["densify_from"],
                                                                self.until)}}
        self.history.append(row["litegs_densify"])
        if c["log"]:
            print(json.dumps(row), flush=True)

    @torch.no_grad()
    def new_rows(self, params, split_idx, clone_idx):
        """Split child: xyz + R N(0, s), scale s/1.6, rest copied; clone: exact copy."""
        c = self.cfg
        out = {}
        stds = params["scales"][split_idx].exp()
        samples = torch.randn(stds.shape, device=stds.device, generator=self.generator) * stds
        rot = normalized_quat_to_rotmat(torch.nn.functional.normalize(params["quats"][split_idx],
                                                                       dim=-1))
        shift = torch.einsum("nij,nj->ni", rot, samples)
        for name, p in params.items():
            if name == "means":
                child = p[split_idx] + shift
            elif name == "scales":
                child = p[split_idx] - math.log(c["split_scale_div"])
            else:
                child = p[split_idx]
            out[name] = torch.cat([child, p[clone_idx]])
        return out

    @torch.no_grad()
    def decay(self, params, optimizers):
        c = self.cfg
        o = params["opacities"]
        if c["opacity_mode"] == "decay":
            o.data.copy_(torch.logit((o.sigmoid() * c["decay_rate"]).clamp_min(c["decay_floor"])))
            if c["clear_adam_on_decay"]:
                for opt in optimizers.values():
                    opt.state.clear()
        else:  # official 'reset': clamp_max(0.005) and zero the opacity moments
            o.data.copy_(torch.logit(o.sigmoid().clamp_max(0.005)))
            optimizers["opacities"].state.clear()


@torch.no_grad()
def append_new(params, optimizers, rows):
    """Append rows to every field; new Adam moments are zero (official _cat_tensors_to_optimizer)."""

    def param_fn(name, p):
        return torch.nn.Parameter(torch.cat([p.detach(), rows[name].detach()]),
                                  requires_grad=p.requires_grad)

    def optimizer_fn(key, v):
        k = len(next(iter(rows.values())))
        return torch.cat([v, torch.zeros((k, *v.shape[1:]), device=v.device, dtype=v.dtype)])

    _update_param_with_optimizer(param_fn, optimizer_fn, params, optimizers)


def check_trainable(params):
    for name, p in params.items():
        if not (isinstance(p, torch.nn.Parameter) and p.requires_grad and p.is_leaf):
            raise RuntimeError(f"LiteGS: field {name} is no longer a trainable leaf")


def create_density(spec):
    return LiteDensity(spec)
