"""Reduced-3DGS density component: 3DGS densification + C1 pruning + C2 SH culling + C3 codec.

Wraps the SplatStudio 3DGS density (gsplat DefaultStrategy with the opacity-reset fix) and adds,
after each runtime optimizer step (official train.py event order within an iteration:
densify/prune -> opacity reset -> mercy -> optimizer step -> save -> SH cull):

* dead-point pruning, opacity < 1/255: every 100 steps once step >= densify_until (15000)
  (`elif args.prune_dead_points`), and at the official save iterations (7000, final);
* "mercy" redundancy + opacity pruning (C1) every 1000 steps up to total - 3000, skipped on
  opacity-reset steps before 15000;
* SH band culling (C2) once at step 15000;
* at the final step: K-means codebooks + float16 (C3), encoded to the official
  `point_cloud_quantised_half.ply` byte layout and decoded back into the parameters, so the
  checkpoint, `validation.json` and `ss-eval` all measure the decoded model.
"""

import json
import math
import time
from dataclasses import replace

import torch
from gsplat.strategy import ops
from splatstudio.backends.gsplat import Density

from . import codec, redundancy, shcull
from .renderer import DEGREE_FIELD, ReducedRenderer

DEFAULTS = {
    "base": {"absgrad": False, "grow_grad2d": 0.0002},
    "total_steps": 30000,
    "densify_until": 15000,
    "opacity_reset_every": 3000,
    "dead_opacity": 1.0 / 255.0,
    "dead_prune_every": 100,
    "save_steps": [7000],
    "mercy": True,
    "mercy_interval": 1000,
    "mercy_finetune": 3000,
    "lambda_mercy": 1.0,
    "mercy_minimum": 3,
    "opacity_quantile": 0.03,
    "opacity_cap": 0.05,
    "knn": 30,
    "box_size": 1.0,
    "cull_sh_step": 15000,
    "std_threshold": 0.04,
    "cdist_threshold": 6 * math.sqrt(3) / 255,
    "official_shcull_quirks": True,
    "encode": True,
    "codebook_size": 256,
    "half_float": True,
    "codebook_seed": 0,
}


def schedule_events(step, cfg):
    """Reduced-3DGS events after optimizer step `step` (1-based), in official order.

    dead_prune (periodic, step >= densify_until) -> mercy -> save_prune (official save
    iterations) -> cull_sh -> encode.
    """
    events = []
    total = cfg["total_steps"]
    if step >= cfg["densify_until"] and step % cfg["dead_prune_every"] == 0:
        events.append("dead_prune")
    if (
        cfg["mercy"]
        and step % cfg["mercy_interval"] == 0
        and step <= total - cfg["mercy_finetune"]
        and (step >= cfg["densify_until"] or step % cfg["opacity_reset_every"] != 0)
    ):
        events.append("mercy")
    if step in cfg["save_steps"] or step == total:
        events.append("save_prune")
    if cfg["cull_sh_step"] is not None and step == cfg["cull_sh_step"]:
        events.append("cull_sh")
    if cfg["encode"] and step == total:
        events.append("encode")
    return events


class ReducedDensity:
    def __init__(self, spec):
        unknown = set(spec.config) - set(DEFAULTS)
        if unknown:
            raise ValueError(f"Unknown reduced-density options: {sorted(unknown)}")
        self.cfg = {**DEFAULTS, **spec.config}
        self.base = Density(replace(spec, config=dict(self.cfg["base"])))
        self.requires = self.base.requires
        self.renderer = ReducedRenderer(replace(spec, config={}))
        self.cameras = None
        self.history = []
        self.encoded = None
        self.lr0 = None

    @property
    def state(self):
        return {"strategy": self.base.state, "reduced3dgs": {"events": self.history,
                                                             "encoded": self.encoded}}

    def set_training_cameras(self, cameras):
        self.cameras = cameras

    def initialize(self, params, optimizers, scene_scale=1.0):
        if DEGREE_FIELD not in params:
            raise ValueError(f"reduced-density needs the '{DEGREE_FIELD}' Gaussian field")
        self.base.initialize(params, optimizers, scene_scale)
        self.lr0 = optimizers["means"].param_groups[0]["lr"]

    def before_backward(self, params, optimizers, step, outputs):
        self.base.before_backward(params, optimizers, step, outputs)

    def check_run_length(self, step, lr):
        """Raise if the run length differs from total_steps while encoding is on.

        The runtime does not pass its step budget, but it sets the means LR to
        lr0 * 0.01 ** (step / steps) before this call (runtime.py), so steps is recoverable.
        """
        if not self.cfg["encode"] or step != 1 or self.lr0 is None:
            return
        if not 0 < lr < self.lr0:  # not the native runtime schedule (e.g. a probe driver)
            print("reduced-density: WARNING cannot infer the run length from the means LR; the "
                  f"encode step runs only at step {self.cfg['total_steps']}", flush=True)
            return
        steps = math.log(0.01) / math.log(lr / self.lr0)
        if abs(steps - self.cfg["total_steps"]) > 0.5:
            raise ValueError(
                f"run has ~{round(steps)} steps but reduced-density total_steps="
                f"{self.cfg['total_steps']}: the final encode -> decode would never run. "
                "Train for total_steps or set the density config accordingly.")

    def after_backward(self, params, optimizers, step, outputs):
        self.check_run_length(step, optimizers["means"].param_groups[0]["lr"])
        self.base.after_backward(params, optimizers, step, outputs)
        events = schedule_events(step, self.cfg)
        if not events:
            return
        if ("mercy" in events or "cull_sh" in events) and not self.cameras:
            raise RuntimeError("reduced-density needs set_training_cameras() (training views)")
        for event in events:
            start = time.perf_counter()
            before = len(params["means"])
            record = {"reduced3dgs": event, "step": step}
            if event in ("dead_prune", "save_prune"):
                self._remove(params, optimizers,
                             params["opacities"].detach().sigmoid() < self.cfg["dead_opacity"])
            elif event == "mercy":
                record.update(self._mercy(params, optimizers))
            elif event == "cull_sh":
                record.update(shcull.cull_sh_bands(
                    self.renderer, params, self.cameras, self.cfg["std_threshold"],
                    self.cfg["cdist_threshold"],
                    official_quirks=self.cfg["official_shcull_quirks"]))
            elif event == "encode":
                record.update(self._encode(params, optimizers))
            record.update(before=before, after=len(params["means"]),
                          seconds=round(time.perf_counter() - start, 2))
            if event not in ("dead_prune", "save_prune") or before != len(params["means"]):
                self.history.append(record)
                print(json.dumps(record), flush=True)

    def _remove(self, params, optimizers, mask):
        if bool(mask.any()):
            ops.remove(params, optimizers, self.base.state, mask)

    @torch.no_grad()
    def _mercy(self, params, optimizers):
        c = self.cfg
        score = redundancy.redundancy_scores(
            params["means"].detach(), params["scales"].detach().exp(), params["quats"].detach(),
            self.cameras, c["knn"], c["box_size"])
        mask, info = redundancy.mercy_mask(
            score, params["opacities"].detach().sigmoid(), c["lambda_mercy"],
            c["mercy_minimum"], c["opacity_quantile"], c["opacity_cap"])
        self._remove(params, optimizers, mask)
        info["score_mean"] = float(score.float().mean())
        return info

    @torch.no_grad()
    def _encode(self, params, optimizers):
        decoded, report = codec.encode_decode(
            params, self.cfg["codebook_size"], self.cfg["codebook_seed"], self.cfg["half_float"])
        report["paper_mem_MB_baseline_equivalent"] = (
            codec.baseline_memory_bytes(len(params["means"])) / codec.MIB)

        # The PLY groups Gaussians by band count, so the decoded order differs; training is
        # over, so the Adam moments are simply reset alongside the new parameter tensors.
        def param_fn(name, p):
            return torch.nn.Parameter(decoded[name].to(p.dtype).reshape(p.shape),
                                      requires_grad=p.requires_grad)

        def optimizer_fn(key, v):
            return torch.zeros_like(v)

        ops._update_param_with_optimizer(param_fn, optimizer_fn, params, optimizers)
        for k, v in list(self.base.state.items()):
            if isinstance(v, torch.Tensor) and v.dim() > 0 and len(v) == len(params["means"]):
                self.base.state[k] = torch.zeros_like(v)
        self.encoded = report
        return report


def create_density(spec):
    return ReducedDensity(spec)
