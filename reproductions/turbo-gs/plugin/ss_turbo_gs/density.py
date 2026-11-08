"""Turbo-GS density: position-appearance densification (Sec. 4.3) under the convergence-aware
budget (Sec. 4.2), on the 3DGS clone/split/prune schedule (gsplat ops keep the optimizer and
state aligned).

Per refinement event (3DGS: every 100 steps in (500, 15000)):
  1. candidates = g_pos > tau_pos  OR  (with probability p_color) g_col > tau_color,
     g_pos = mean visible NDC 2D-mean gradient norm (3DGS / gsplat DefaultStrategy),
     g_col = mean visible norm of dL/d(sh0) (the DC colour coefficients, Inria features_dc);
     tau_color = 0.01 * tau_pos.
  2. at most B(t) - n candidates are kept (largest max(g_pos/tau_pos, g_col/tau_color)).
  3. small candidates are cloned, large ones split (3DGS), then the 3DGS prune
     (opacity < 0.005, world size > 0.1 extent after the first reset).
Opacity reset to 0.01 every 3000 steps before 15000 (3DGS; Sec. 5.3 keeps it for Turbo-3DGS).
"""

import json
import random

import torch
from gsplat import DefaultStrategy
from gsplat.strategy.ops import duplicate, reset_opa, split

from .budget import (
    BUDGET_TABLE,
    BudgetScheduler,
    progress_index,
    resolve_budget,
    training_progress,
)
from .renderer import COLORS

LOSS_VALUE = "training.loss_value"
DEFAULTS = {
    "grow_grad2d": 0.0002,
    "color_ratio": 0.01,
    "color_probability": 0.2,
    "use_color": True,
    "use_budget": True,
    "budget": None,
    "budget_table": dict(BUDGET_TABLE),
    "warmup": 100,
    "ema": 0.1,
    "lam": 0.5,
    "window": 500,
    "total_steps": 30000,
    "t_mode": "training",
    "refine_start_iter": 500,
    "refine_stop_iter": 15000,
    "refine_every": 100,
    "reset_every": 3000,
    "grow_scale3d": 0.01,
    "prune_scale3d": 0.1,
    "prune_opa": 0.005,
    "seed": 0,
    "verbose": True,
}


class TurboDensity:
    def __init__(self, spec):
        unknown = set(spec.config) - set(DEFAULTS)
        if unknown:
            raise ValueError(f"Unknown turbo density options: {sorted(unknown)}")
        self.c = {**DEFAULTS, **spec.config}
        c = self.c
        self.strategy = DefaultStrategy(
            prune_opa=c["prune_opa"], grow_grad2d=c["grow_grad2d"],
            grow_scale3d=c["grow_scale3d"], prune_scale3d=c["prune_scale3d"],
            refine_start_iter=c["refine_start_iter"], refine_stop_iter=c["refine_stop_iter"],
            reset_every=c["reset_every"], refine_every=c["refine_every"],
        )
        self.tau_pos = float(c["grow_grad2d"])
        self.tau_col = float(c["color_ratio"]) * self.tau_pos
        self.rng = random.Random(int(c["seed"]))
        # Renderer statistics only (the runtime requires density.requires <= renderer caps).
        # The loss value comes from the `photometric` loss (provides LOSS_VALUE) and is
        # enforced at runtime: after_backward raises if outputs["training_loss"] is missing.
        self.requires = {"gradient.screen.signed", COLORS}
        if c["t_mode"] not in ("training", "densify_window"):
            raise ValueError("t_mode must be 'training' or 'densify_window'")
        self.scheduler = None
        self.peak = 0
        self.state = None
        self.log = []

    # -- lifecycle -------------------------------------------------------------------------
    def initialize(self, params, optimizers, scene_scale=1.0):
        self.strategy.check_sanity(params, optimizers)
        self.state = self.strategy.initialize_state(scene_scale=scene_scale)
        self.state["colgrad"] = None
        self.initial_count = len(params["means"])
        self.state["budget"] = {}

    def _scheduler(self):
        # Resolved lazily (first densification) so probes on synthetic scenes can run the
        # statistics path without a scene budget; training fails at step 600 without one.
        if self.scheduler is None:
            c = self.c
            m = resolve_budget(self.initial_count, c["budget"], c["budget_table"])
            self.scheduler = BudgetScheduler(self.initial_count, m, warmup=c["warmup"],
                                             ema=c["ema"], lam=c["lam"], window=c["window"])
        return self.scheduler

    def before_backward(self, params, optimizers, step, outputs):
        outputs["info"]["means2d"].retain_grad()
        outputs["colors"].retain_grad()

    def after_backward(self, params, optimizers, step, outputs):
        s = self.strategy
        if step >= s.refine_stop_iter:
            return
        if step > self.c["warmup"]:
            if "training_loss" not in outputs:
                raise KeyError("turbo density needs outputs['training_loss'] from the loss")
            self._scheduler().record(step, outputs["training_loss"].item())
        self._update_state(params, outputs)
        if step > s.refine_start_iter and step % s.refine_every == 0:
            self.refine(params, optimizers, step)
        if 0 < step < s.refine_stop_iter and step % s.reset_every == 0:
            reset_opa(params, optimizers, self.state, value=s.prune_opa * 2.0)

    # -- statistics ------------------------------------------------------------------------
    def _update_state(self, params, outputs):
        info = outputs["info"]
        self.strategy._update_state(params, self.state, info, packed=False)
        n = len(params["means"])
        if self.state["colgrad"] is None:
            self.state["colgrad"] = torch.zeros(n, device=params["means"].device)
        grad = outputs["colors"].grad
        if grad is None:
            raise RuntimeError("no gradient on the rendered SH coefficients")
        sel = (info["radii"] > 0.0).all(dim=-1)  # [C, N], same visibility as grad2d/count
        ids = torch.where(sel)[1]
        self.state["colgrad"].index_add_(0, ids, grad[:, 0, :].norm(dim=-1)[ids])

    def averaged(self):
        count = self.state["count"].clamp_min(1)
        return self.state["grad2d"] / count, self.state["colgrad"] / count

    # -- densification ---------------------------------------------------------------------
    def select(self, g_pos, g_col, n_current, allowance, color_active):
        """Candidate mask (Sec. 4.3) capped at `allowance` new Gaussians (Sec. 4.2)."""
        pos = g_pos > self.tau_pos
        col = (g_col > self.tau_col) if color_active else torch.zeros_like(pos)
        cand = pos | col
        n_cand = int(cand.sum())
        if allowance is not None and n_cand > allowance:
            score = g_pos / self.tau_pos
            if color_active:
                score = torch.maximum(score, g_col / self.tau_col)
            score = torch.where(cand, score, torch.full_like(score, -1.0))
            keep = torch.zeros_like(cand)
            if allowance > 0:
                keep[score.topk(int(allowance)).indices] = True
            cand = keep
        return cand, {"pos": int(pos.sum()), "col_only": int((col & ~pos).sum()),
                      "candidates": n_cand, "selected": int(cand.sum())}

    @torch.no_grad()
    def refine(self, params, optimizers, step):
        s, c = self.strategy, self.c
        n = len(params["means"])
        g_pos, g_col = self.averaged()
        color_active = bool(c["use_color"]) and self.rng.random() < float(c["color_probability"])
        allowance, row = None, {"step": step, "n": n, "color_active": color_active}
        if c["use_budget"]:
            sched = self._scheduler()
            sched.update()
            if c["t_mode"] == "training":
                t = training_progress(step, c["total_steps"])
            elif c["t_mode"] == "densify_window":
                t = progress_index(step, s.refine_start_iter, s.refine_stop_iter)
            else:
                raise ValueError("t_mode must be 'training' or 'densify_window'")
            target = int(sched.budget(t))
            allowance = max(target - n, 0)
            row.update(t=t, target=target, **sched.state())
            self.state["budget"] = {k: v for k, v in sched.state().items() if v is not None}
        cand, stats = self.select(g_pos, g_col, n, allowance, color_active)
        row.update(stats)
        is_small = (torch.exp(params["scales"]).max(dim=-1).values
                    <= s.grow_scale3d * self.state["scene_scale"])
        is_dupli = cand & is_small
        is_split = cand & ~is_small
        n_dupli = int(is_dupli.sum())
        if n_dupli > 0:
            duplicate(params=params, optimizers=optimizers, state=self.state, mask=is_dupli)
        is_split = torch.cat([is_split, torch.zeros(n_dupli, dtype=torch.bool,
                                                    device=is_split.device)])
        if int(is_split.sum()) > 0:
            split(params=params, optimizers=optimizers, state=self.state, mask=is_split)
        row["pruned"] = s._prune_gs(params, optimizers, self.state, step)
        row["after"] = len(params["means"])
        self.peak = max(self.peak, row["after"], n + int(cand.sum()))
        row["peak"] = self.peak
        self.log.append(row)
        if self.c.get("verbose", True):
            print("turbo-density " + json.dumps(row), flush=True)
        for key in ("grad2d", "count", "colgrad"):
            self.state[key].zero_()
        torch.cuda.empty_cache()


def create_density(spec):
    return TurboDensity(spec)
