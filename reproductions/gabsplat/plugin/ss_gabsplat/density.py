"""Frequency-adaptive density control (paper Sec. 3.4.2, supp. B).

gsplat's DefaultStrategy (3DGS adaptive density control) with:
- new children (clones and split children) get their Gabor frequencies reset to `child_freq`
  and weights to `child_weight` right after growth (gsplat appends the duplicates, then the
  2*n_split split children, so the children are the last n_dupli + 2*n_split rows);
- every opacity reset (3DGS schedule: every `reset_every` steps before `refine_stop_iter`) also
  resets all Gabor weights to `reset_weight`;
- opacity prune threshold 0.01 and reset value 2 * prune_opa = 0.02 (supp. B);
- opacity learning rate 0.025 (Sec. 4.1, supp. B; runtime default 0.05).
Field values are written in place (`.data`), so Parameters keep requires_grad; the Adam moments
of the reset rows are zeroed, like gsplat's reset_opa does for opacity.
"""

import math
from dataclasses import replace

import torch
from gsplat.strategy.ops import reset_opa
from splatstudio.backends.gsplat import Density, resets_opacity

from .renderer import FREQ, WEIGHT

GABOR_KEYS = ("child_freq", "child_weight", "reset_weight", "child_reset", "weight_reset",
              "opacity_lr")
DEFAULTS = {"child_freq": 0.001, "child_weight": 0.01, "reset_weight": 0.01,
            "child_reset": True, "weight_reset": True, "opacity_lr": 0.025}


def _logit(p):
    return math.log(p / (1.0 - p))


@torch.no_grad()
def reset_rows(params, optimizers, name, rows, value):
    """params[name][rows] = value in place; zero the Adam moments of those rows."""
    p = params[name]
    p.data[rows] = value
    for state in optimizers[name].state.values():
        for key in ("exp_avg", "exp_avg_sq"):
            if key in state:
                state[key][rows] = 0.0


def check_trainable(params, optimizers):
    """Every field must remain a trainable leaf owned by its optimizer (TODO-runtime pitfall)."""
    for name, p in params.items():
        if not (p.requires_grad and p.is_leaf):
            raise RuntimeError(f"{name} is no longer a trainable leaf after density control")
        if optimizers[name].param_groups[0]["params"][0] is not p:
            raise RuntimeError(f"{name}: optimizer does not own the current Parameter")


class GaborDensity(Density):
    def __init__(self, spec):
        config = dict(spec.config)
        gab = {**DEFAULTS, **{k: config.pop(k) for k in GABOR_KEYS if k in config}}
        super().__init__(replace(spec, config=config))
        self.child_freq = float(gab["child_freq"])
        self.child_weight_logit = _logit(float(gab["child_weight"]))
        self.reset_weight_logit = _logit(float(gab["reset_weight"]))
        self.child_reset = bool(gab["child_reset"])
        self.weight_reset = bool(gab["weight_reset"])
        self.opacity_lr = gab["opacity_lr"]
        self.events = []  # (step, kind, count) for diagnostics/probes

    def initialize(self, params, optimizers, scene_scale=1.0):
        super().initialize(params, optimizers, scene_scale)
        if self.opacity_lr is not None:
            optimizers["opacities"].param_groups[0]["lr"] = float(self.opacity_lr)

    def after_backward(self, params, optimizers, step, outputs):
        s = self.strategy
        if step >= s.refine_stop_iter:
            return
        s._update_state(params, self.state, outputs["info"], packed=False)
        if (step > s.refine_start_iter and step % s.refine_every == 0
                and step % s.reset_every >= s.pause_refine_after_reset):
            n_dupli, n_split = s._grow_gs(params, optimizers, self.state, step)
            n_new = n_dupli + 2 * n_split
            if self.child_reset and n_new > 0:
                rows = torch.arange(len(params["means"]) - n_new, len(params["means"]),
                                    device=params["means"].device)
                reset_rows(params, optimizers, FREQ, rows, self.child_freq)
                reset_rows(params, optimizers, WEIGHT, rows, self.child_weight_logit)
                self.events.append((step, "child_reset", n_new))
            s._prune_gs(params, optimizers, self.state, step)
            self.state["grad2d"].zero_()
            self.state["count"].zero_()
            if s.refine_scale2d_stop_iter > 0:
                self.state["radii"].zero_()
            torch.cuda.empty_cache()
            check_trainable(params, optimizers)
        # 3DGS opacity reset schedule (step_post_backward is re-implemented here, so gsplat's
        # broken 1.5.x reset condition is not involved)
        if self.opacity_reset and resets_opacity(s, step):
            reset_opa(params, optimizers, self.state, value=s.prune_opa * 2.0)
            if self.weight_reset:
                rows = torch.arange(len(params["means"]), device=params["means"].device)
                reset_rows(params, optimizers, WEIGHT, rows, self.reset_weight_logit)
                self.events.append((step, "weight_reset", len(rows)))
            check_trainable(params, optimizers)


def create_density(spec):
    return GaborDensity(spec)
