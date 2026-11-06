"""Faster-GS density control = the unchanged 3DGS adaptive density control as implemented in
the official `Model.py` / `Trainer.py` (main @ 090e8ad), plus Morton re-ordering (paper §3.3) and
the official CUDA Adam (optim.py) installed in place of the runtime's torch Adam.

Step mapping: an official callback at 0-based iteration i runs before training step i, i.e.
after i optimizer steps — the runtime's `after_backward(step=i)` (also after i steps). So:
  initialize            : FusedAdam swap (opacity lr 0.025), Morton sort (official iteration 0)
  after_backward(s):
    ADC          s in [600, 14900], s % 100 == 0  (prune-large only for s > 3000)
    info re-zero after ADC if s < 14900; info disabled once s >= 14900
    Morton       s % 5000 == 0 and s <= 15000      (after ADC, before reset: priorities 100/99/90)
    reset        s % 3000 == 0, 3000 <= s <= 14900 (clamp to 0.01, opacity moments zeroed)
    cleanup      s == total_steps: prune opacity < 1/255 and degenerate quaternions
Every population change goes through gsplat's `_update_param_with_optimizer` / `remove` /
`reset_opa`, so Adam moments stay aligned with the parameters (state key `step` is preserved).
"""

import math

import torch
from gsplat.strategy.ops import _update_param_with_optimizer, remove, reset_opa
from torch import nn

from .morton import morton_order
from .optim import replace_optimizers

DEFAULTS = {
    "densify_from": 600,  # DENSIFICATION_START_ITERATION
    "densify_until": 14900,  # DENSIFICATION_END_ITERATION (inclusive)
    "densify_every": 100,  # DENSIFICATION_INTERVAL
    "grad_threshold": 0.0002,  # DENSIFICATION_GRAD_THRESHOLD
    "percent_dense": 0.01,  # DENSIFICATION_PERCENT_DENSE
    "min_opacity": 0.005,  # adaptive_density_control(..., 0.005, ...)
    "big_world": 0.1,  # prune max scale > 0.1 * extent
    "big_world_after": 3000,  # ... when iteration > OPACITY_RESET_INTERVAL
    "reset_every": 3000,  # OPACITY_RESET_INTERVAL
    "reset_value": 0.01,  # clamp_max(logit(0.01))
    "morton_every": 5000,  # MORTON_ORDERING_INTERVAL
    "morton_until": 15000,  # MORTON_ORDERING_END_ITERATION (inclusive)
    "lr_opacities": 0.025,  # OPTIMIZER.LEARNING_RATE_OPACITIES
    "fused_adam": True,  # official FusedAdam CUDA kernel (False: torch.optim.Adam, same math)
    "final_prune_opacity": 1.0 / 255.0,  # MIN_OPACITY_AFTER_TRAINING
    "total_steps": 30000,  # NUM_ITERATIONS (cleanup step)
}
REQUIRES = {"densification.fastergs"}
BASE_FIELDS = ("means", "scales", "quats", "opacities", "sh0", "shN")


def quat_to_rotmat(quats):
    """Rotation of q/|q| (wxyz), Cameras.utils.quaternion_to_rotation_matrix."""
    r, x, y, z = torch.nn.functional.normalize(quats, dim=-1).unbind(-1)
    return torch.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - r * z), 2 * (x * z + r * y),
        2 * (x * y + r * z), 1 - 2 * (x * x + z * z), 2 * (y * z - r * x),
        2 * (x * z - r * y), 2 * (y * z + r * x), 1 - 2 * (x * x + y * y)], -1).reshape(-1, 3, 3)


def schedule(cfg, steps=30000):
    """(ADC steps, Morton steps incl. 0, reset steps) implied by the official callbacks."""
    adc = [s for s in range(1, steps + 1) if cfg["densify_from"] <= s <= cfg["densify_until"]
           and s % cfg["densify_every"] == 0]
    morton = [0] + [s for s in range(1, steps + 1)
                    if s <= cfg["morton_until"] and s % cfg["morton_every"] == 0]
    reset = [s for s in range(1, steps + 1) if cfg["reset_every"] <= s <= cfg["densify_until"]
             and s % cfg["reset_every"] == 0]
    return adc, morton, reset


def append_gaussians(params, optimizers, new):
    """Optim.adam_utils.extend_param_groups: concatenate rows; new Adam moments are zero."""
    n_new = len(new["means"])

    def param_fn(name, p):
        return nn.Parameter(torch.cat([p.detach(), new[name].to(p.dtype)]),
                            requires_grad=p.requires_grad)

    def optimizer_fn(key, v):
        return torch.cat([v, torch.zeros((n_new, *v.shape[1:]), dtype=v.dtype, device=v.device)])

    _update_param_with_optimizer(param_fn, optimizer_fn, params, optimizers)


def permute_gaussians(params, optimizers, order):
    """Optim.adam_utils.sort_param_groups: re-order parameters and moments identically."""

    def param_fn(name, p):
        return nn.Parameter(p.detach()[order], requires_grad=p.requires_grad)

    def optimizer_fn(key, v):
        return v[order]

    _update_param_with_optimizer(param_fn, optimizer_fn, params, optimizers)


@torch.no_grad()
def adc_masks(params, info, cfg, extent):
    """Official `adaptive_density_control` selection (before the population changes)."""
    denominator = info[0].clamp_min(1.0)
    densify = info[1] >= cfg["grad_threshold"] * denominator
    is_small = params["scales"].max(dim=1).values <= math.log(cfg["percent_dense"] * extent)
    return densify & is_small, densify & ~is_small


@torch.no_grad()
def adaptive_density_control(params, optimizers, info, cfg, extent, prune_large, generator=None):
    """Duplicate small, split large (2 children, scale / 1.6, sampled offsets), then prune split
    parents, low opacity (< min_opacity), degenerate quaternions and (optionally) large ones."""
    duplicate_mask, split_mask = adc_masks(params, info, cfg, extent)
    dup = {k: params[k][duplicate_mask] for k in BASE_FIELDS}
    scales = params["scales"][split_mask].exp().expand(2, -1, -1).flatten(end_dim=1)
    quats = params["quats"][split_mask].expand(2, -1, -1).flatten(end_dim=1)
    noise = torch.randn(scales.shape, device=scales.device, generator=generator)
    offsets = (quat_to_rotmat(quats) @ (scales * noise)[..., None])[..., 0]
    split = {
        "means": params["means"][split_mask].expand(2, -1, -1).flatten(end_dim=1) + offsets,
        "scales": scales.mul(0.625).log(),  # 1 / 1.6
        "quats": quats,
        "opacities": params["opacities"][split_mask].expand(2, -1).flatten(end_dim=1),
        "sh0": params["sh0"][split_mask].expand(2, -1, -1, -1).flatten(end_dim=1),
        "shN": params["shN"][split_mask].expand(2, -1, -1, -1).flatten(end_dim=1),
    }
    new = {k: torch.cat([dup[k], split[k]]) for k in BASE_FIELDS}
    n_new = len(new["means"])
    append_gaussians(params, optimizers, new)
    prune = torch.cat([split_mask, torch.zeros(n_new, dtype=torch.bool, device=split_mask.device)])
    m = cfg["min_opacity"]
    prune |= params["opacities"] < math.log(m / (1 - m))
    prune |= params["quats"].square().sum(dim=1) < 1e-8
    if prune_large:
        prune |= params["scales"].max(dim=1).values > math.log(cfg["big_world"] * extent)
    remove(params, optimizers, {}, prune)
    return {"duplicated": int(duplicate_mask.sum()), "split": int(split_mask.sum()),
            "pruned": int(prune.sum()), "gaussians": len(params["means"])}


class FasterGSDensity:
    def __init__(self, spec):
        unknown = set(spec.config) - set(DEFAULTS)
        if unknown:
            raise ValueError(f"Unknown Faster-GS density options: {sorted(unknown)}")
        self.cfg = {**DEFAULTS, **spec.config}
        self.requires = set(REQUIRES)
        self.state = None
        self.extent = 1.0
        self.generator = None  # optional torch.Generator for the split offsets (probes)

    def initialize(self, params, optimizers, scene_scale=1.0):
        extra = set(params) - set(BASE_FIELDS)
        if extra:
            raise ValueError(f"Faster-GS density does not know fields {sorted(extra)}")
        self.extent = float(scene_scale)
        lr = {"opacities": self.cfg["lr_opacities"]}
        if self.cfg["fused_adam"]:
            replace_optimizers(params, optimizers, lr)
        else:
            optimizers["opacities"].param_groups[0]["lr"] = float(lr["opacities"])
        self.state = {"events": []}
        self.morton(params, optimizers, 0)

    def before_backward(self, params, optimizers, step, outputs):
        pass

    @torch.no_grad()
    def morton(self, params, optimizers, step, holder=None):
        order = morton_order(params["means"].detach())
        permute_gaussians(params, optimizers, order)
        if holder is not None and holder.buffer is not None and holder.buffer.shape[1] == len(order):
            holder.buffer.copy_(holder.buffer[:, order])
        self.state["events"].append({"step": step, "op": "morton"})

    @torch.no_grad()
    def after_backward(self, params, optimizers, step, outputs):
        cfg = self.cfg
        holder = outputs["info"]["fastergs"]
        if (cfg["densify_from"] <= step <= cfg["densify_until"]
                and step % cfg["densify_every"] == 0):
            info = holder.buffer
            if info is None or info.shape[1] != len(params["means"]):
                raise RuntimeError("Faster-GS densification info is missing or misaligned")
            event = adaptive_density_control(params, optimizers, info, cfg, self.extent,
                                             step > cfg["big_world_after"], self.generator)
            self.state["events"].append({"step": step, "op": "adc", **event})
            if step < cfg["densify_until"]:
                holder.get(len(params["means"]), params["means"].device).zero_()
        if step >= cfg["densify_until"]:
            holder.enabled = False
            holder.buffer = None
        if step <= cfg["morton_until"] and step % cfg["morton_every"] == 0:
            self.morton(params, optimizers, step, holder)
        if (cfg["reset_every"] <= step <= cfg["densify_until"]
                and step % cfg["reset_every"] == 0):
            reset_opa(params, optimizers, {}, cfg["reset_value"])
            self.state["events"].append({"step": step, "op": "reset"})
        if step == cfg["total_steps"]:
            m = cfg["final_prune_opacity"]
            prune = params["opacities"] < math.log(m / (1 - m))
            prune |= params["quats"].square().sum(dim=1) < 1e-8
            if prune.any() and not prune.all():
                remove(params, optimizers, {}, prune)
            self.state["events"].append({"step": step, "op": "cleanup", "pruned": int(prune.sum())})


def create_density(spec):
    return FasterGSDensity(spec)
