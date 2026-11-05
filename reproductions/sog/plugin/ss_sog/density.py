"""SOG density component: modified 3DGS densification + square grid + PLAS sorting + codec.

Official train.py event order within an iteration (after backward):
densify/prune (500 < it < 15000, it % 1000 == 0) -> prune to square + sort (it > 500,
it % 1000 == 0) -> optimizer step; compression at the final iteration. The native runtime
steps the optimizer before `after_backward` (shared runtime difference).

* C5 densification (supp. Table "params"): interval 1000, grad threshold 7e-5, prune opacity
  < 0.1, percent_dense 0.1, opacity reset disabled -> no screen/world-size pruning.
* C3 log positions: `params["means"]` stores sign(x) log1p|x|; `initialize` converts the SfM
  init and split samples the children in world space (official densify_and_split, line 415).
* C1 square grid + PLAS sort at init and after every densification interval.
* C4 at `total_steps`: quantize + JPEG XL encode, decode, and replace the parameters so the
  checkpoint, `validation.json` and `ss-eval` measure the decoded model (pattern copied from the
  Reduced-3DGS / EAGLES plugins of this workspace). Encoded files are kept in the density state.
"""

import json
import math
import time
from dataclasses import replace

import torch
import torch.nn.functional as F
from gsplat import DefaultStrategy
from gsplat.strategy import ops
from gsplat.strategy.ops import duplicate
from gsplat.utils import normalized_quat_to_rotmat
from splatstudio.backends.gsplat import Density

from . import codec, sorting
from .renderer import inverse_log_transform, log_transform

DEFAULTS = {
    "base": {
        "absgrad": False,
        "grow_grad2d": 7e-5,
        "prune_opa": 0.1,
        "grow_scale3d": 0.1,
        "refine_every": 1000,
        "refine_start_iter": 500,
        "refine_stop_iter": 15000,
        # opacity_reset_interval = 1e7 in the official yaml: never resets and, because
        # size_threshold is only set after the first reset, never prunes by size either.
        "reset_every": 10**9,
        "opacity_reset": False,
    },
    "total_steps": 30000,
    "sort_every": 1000,
    "sort_after": 500,
    "sort_weights": {"means": 1.0, "sh0": 1.0, "shN": 0.0, "opacities": 0.0, "scales": 1.0,
                     "quats": 0.0},
    "sort_normalize": True,
    "sort_activated": True,
    "sort_shuffle": True,
    "improvement_break": 1e-4,
    "encode": True,
    "codec_env": None,
}


@torch.no_grad()
def split_world(params, optimizers, state, mask, revised_opacity=False):
    """gsplat 1.5.3 strategy/ops.py:split (Apache-2.0) with log-space means: children are
    sampled around the world-space centre and mapped back (official densify_and_split)."""
    device = mask.device
    sel = torch.where(mask)[0]
    rest = torch.where(~mask)[0]
    scales = torch.exp(params["scales"][sel])
    quats = F.normalize(params["quats"][sel], dim=-1)
    rotmats = normalized_quat_to_rotmat(quats)
    samples = torch.einsum("nij,nj,bnj->bni", rotmats, scales,
                           torch.randn(2, len(scales), 3, device=device))

    def param_fn(name, p):
        repeats = [2] + [1] * (p.dim() - 1)
        if name == "means":
            p_split = log_transform((inverse_log_transform(p[sel]) + samples).reshape(-1, 3))
        elif name == "scales":
            p_split = torch.log(scales / 1.6).repeat(2, 1)
        elif name == "opacities" and revised_opacity:
            p_split = torch.logit(1.0 - torch.sqrt(1.0 - torch.sigmoid(p[sel]))).repeat(repeats)
        else:
            p_split = p[sel].repeat(repeats)
        return torch.nn.Parameter(torch.cat([p[rest], p_split]), requires_grad=p.requires_grad)

    def optimizer_fn(key, v):
        return torch.cat([v[rest], torch.zeros((2 * len(sel), *v.shape[1:]), device=device)])

    ops._update_param_with_optimizer(param_fn, optimizer_fn, params, optimizers)
    for k, v in state.items():
        if isinstance(v, torch.Tensor):
            repeats = [2] + [1] * (v.dim() - 1)
            state[k] = torch.cat((v[rest], v[sel].repeat(repeats)))


class LogSpaceStrategy(DefaultStrategy):
    """DefaultStrategy whose split works on log-space means (`_grow_gs` copied from gsplat
    1.5.3 strategy/default.py, Apache-2.0, with `split` -> `split_world`)."""

    @torch.no_grad()
    def _grow_gs(self, params, optimizers, state, step):
        count = state["count"]
        grads = state["grad2d"] / count.clamp_min(1)
        device = grads.device
        is_grad_high = grads > self.grow_grad2d
        is_small = (torch.exp(params["scales"]).max(dim=-1).values
                    <= self.grow_scale3d * state["scene_scale"])
        is_dupli = is_grad_high & is_small
        n_dupli = int(is_dupli.sum().item())
        is_split = is_grad_high & ~is_small
        if step < self.refine_scale2d_stop_iter:
            is_split |= state["radii"] > self.grow_scale2d
        n_split = int(is_split.sum().item())
        if n_dupli > 0:
            duplicate(params=params, optimizers=optimizers, state=state, mask=is_dupli)
        is_split = torch.cat([is_split, torch.zeros(n_dupli, dtype=torch.bool, device=device)])
        if n_split > 0:
            split_world(params, optimizers, state, is_split, self.revised_opacity)
        return n_dupli, n_split


def schedule_events(step, cfg):
    """SOG events after optimizer step `step` (1-based)."""
    if cfg["encode"] and step == cfg["total_steps"]:
        return ["encode"]
    if step > cfg["sort_after"] and step % cfg["sort_every"] == 0:
        return ["sort"]
    return []


class SOGDensity:
    def __init__(self, spec):
        unknown = set(spec.config) - set(DEFAULTS)
        if unknown:
            raise ValueError(f"Unknown sog-density options: {sorted(unknown)}")
        self.cfg = {**DEFAULTS, **spec.config}
        base_cfg = {**DEFAULTS["base"], **self.cfg["base"]}
        self.base = Density(replace(spec, config=dict(base_cfg)))
        strategy_cfg = {k: v for k, v in base_cfg.items() if k != "opacity_reset"}
        self.base.strategy = LogSpaceStrategy(**strategy_cfg)
        self.requires = self.base.requires
        self.history = []
        self.encoded = None
        self.files = None
        self.lr0 = None

    @property
    def state(self):
        files = None
        if self.files is not None:
            files = {k: torch.frombuffer(bytearray(v), dtype=torch.uint8)
                     for k, v in self.files.items()}
        return {"strategy": self.base.state,
                "sog": {"events": self.history, "encoded": self.encoded, "files": files}}

    def initialize(self, params, optimizers, scene_scale=1.0):
        with torch.no_grad():  # C3: the SfM init is stored as the official _xyz (log space)
            params["means"].copy_(log_transform(params["means"]))
        self.base.initialize(params, optimizers, scene_scale)
        self.lr0 = optimizers["means"].param_groups[0]["lr"]
        self._regrid(params, optimizers, 0)  # official train.py:65-67 (before iteration 1)

    def before_backward(self, params, optimizers, step, outputs):
        self.base.before_backward(params, optimizers, step, outputs)

    def check_run_length(self, step, lr):
        """Raise if the run length differs from total_steps while encoding is on (copied from
        the Reduced-3DGS plugin): the runtime sets lr = lr0 * 0.01 ** (step / steps)."""
        if not self.cfg["encode"] or step != 1 or self.lr0 is None:
            return
        if not 0 < lr < self.lr0:
            print("sog-density: WARNING cannot infer the run length from the means LR; the "
                  f"encode step runs only at step {self.cfg['total_steps']}", flush=True)
            return
        steps = math.log(0.01) / math.log(lr / self.lr0)
        if abs(steps - self.cfg["total_steps"]) > 0.5:
            raise ValueError(
                f"run has ~{round(steps)} steps but sog-density total_steps="
                f"{self.cfg['total_steps']}: the final encode -> decode would never run. "
                "Train for total_steps or set the density config accordingly.")

    def after_backward(self, params, optimizers, step, outputs):
        self.check_run_length(step, optimizers["means"].param_groups[0]["lr"])
        self.base.after_backward(params, optimizers, step, outputs)
        for event in schedule_events(step, self.cfg):
            if event == "sort":
                self._regrid(params, optimizers, step)
            elif event == "encode":
                self._encode(params, optimizers, step)

    def _log(self, record):
        self.history.append(record)
        print(json.dumps(record), flush=True)

    def _regrid(self, params, optimizers, step):
        start = time.perf_counter()
        before = len(params["means"])
        side, removed = sorting.prune_to_square(params, optimizers, self.base.state)
        sorting.sort_into_grid(params, optimizers, self.base.state, self.cfg)
        self._log({"sog": "sort", "step": step, "before": before, "pruned_to_square": removed,
                   "grid_side": side, "seconds": round(time.perf_counter() - start, 2)})

    @torch.no_grad()
    def _encode(self, params, optimizers, step):
        start = time.perf_counter()
        before = len(params["means"])
        side, removed = sorting.prune_to_square(params, optimizers, self.base.state)
        decoded, report, files = codec.encode_decode(params, codec_env=self.cfg["codec_env"])

        # Training is over: the decoded tensors replace the parameters (same grid order) and
        # the Adam moments are reset, as in the Reduced-3DGS plugin's final encode.
        def param_fn(name, p):
            if name not in decoded:
                return p
            return torch.nn.Parameter(decoded[name].to(p.dtype).reshape(p.shape),
                                      requires_grad=p.requires_grad)

        def optimizer_fn(key, v):
            return torch.zeros_like(v)

        ops._update_param_with_optimizer(param_fn, optimizer_fn, params, optimizers,
                                         names=list(decoded))
        self.encoded = {**report, "step": step, "pruned_to_square": removed, "before": before}
        self.files = files
        self._log({"sog": "encode", "step": step, "size_bytes": report["size_bytes"],
                   "size_mb": round(report["size_mb"], 3), "grid_side": side,
                   "file_bytes": report["file_bytes"],
                   "seconds": round(time.perf_counter() - start, 2)})


def create_density(spec):
    return SOGDensity(spec)
