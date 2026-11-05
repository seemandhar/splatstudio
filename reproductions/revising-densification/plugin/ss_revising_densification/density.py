"""Revised adaptive density control (Rota Bulo et al., arXiv 2404.06109, Sec. 3).

Re-derived from the paper (no official code): error-based densification score (Sec. 3.2),
opacity correction on clone (Sec. 3.3), primitive growth control (Sec. 3.4), opacity decrement
instead of opacity reset (Sec. 3.5), ADC active until 27k iterations (Sec. 4.2). Population
changes go through gsplat.strategy.ops so params and Adam moments stay aligned.
"""

import json

import torch
from gsplat.strategy.ops import duplicate, remove, reset_opa, split

from . import core
from .kernels import error_stats
from .losses import ERROR_KEY

DEFAULTS = {
    "densify_from": 500,  # 3DGS densify_from_iter (first run at 600)
    "interval": 100,  # 3DGS densification_interval
    "densify_until": 27000,  # Sec. 4.2: ADC active for 90% of 30k
    # C1 error-based score (Sec. 3.2, 4.2)
    "score": "error",  # "gradient" = ablation (3DGS tau_k, mean 2D gradient norm)
    "error_threshold": 0.1,
    "grad_threshold": 2e-4,  # ablation only (3DGS densify_grad_threshold)
    # C3 growth control (Sec. 3.4, 4.2)
    "growth_control": True,  # False = ablation (every candidate grows, no cap)
    "growth_fraction": 0.05,
    "max_primitives": "baseline",  # or an int
    # C2 opacity correction on clone (Sec. 3.3)
    "opacity_correction": True,
    # C4 opacity regularisation (Sec. 3.5)
    "opacity_mode": "decrement",  # "reset" = ablation (3DGS hard reset)
    "opacity_decrement": 0.001,
    "reset_every": 3000,  # ablation only
    "reset_until": 15000,  # ablation only
    "reset_value": 0.01,  # ablation only
    # unchanged 3DGS rules
    "min_opacity": 0.005,
    "percent_dense": 0.01,
    "big_prune_scale": 0.1,  # world-size prune, 3DGS 0.1 * extent
    "big_prune_after": 3000,  # 3DGS: size pruning once iteration > opacity_reset_interval
    "log": True,  # one JSON line per ADC run on stdout (train log)
}


class RevisedADC:
    def __init__(self, spec):
        unknown = set(spec.config) - set(DEFAULTS)
        if unknown:
            raise ValueError(f"Unknown Revising-Densification density options: {sorted(unknown)}")
        c = {**DEFAULTS, **spec.config}
        if c["score"] not in ("error", "gradient"):
            raise ValueError("score must be 'error' or 'gradient'")
        if c["opacity_mode"] not in ("decrement", "reset"):
            raise ValueError("opacity_mode must be 'decrement' or 'reset'")
        if not 0 < c["growth_fraction"] <= 1:
            raise ValueError("growth_fraction must be in (0, 1]")
        self.cfg = c
        self.requires = {"render.rgb"}
        self.state = None
        self.history = []

    # ------------------------------------------------------------------ lifecycle
    def initialize(self, params, optimizers, scene_scale=1.0):
        n = len(params["means"])
        dev = params["means"].device
        c = self.cfg
        if c["growth_control"]:
            cap, cap_source = core.primitive_cap(c["max_primitives"], n)
        else:
            cap, cap_source = None, "none"
        self.state = {
            "score": torch.zeros(n, device=dev),  # E_k = max over views since the last run
            "grad_accum": torch.zeros(n, device=dev),  # gradient ablation only
            "count": torch.zeros(n, device=dev),
            "extent": float(scene_scale),
            "cap": cap,
            "cap_source": cap_source,
            "runs": 0,
        }
        if c["log"]:
            print(json.dumps({"revdens": "init", "n_init": n, "cap": cap,
                              "cap_source": cap_source, "extent": float(scene_scale)}),
                  flush=True)

    def before_backward(self, params, optimizers, step, outputs):
        # screen-space gradient: the 3DGS score (ablation) and the renderer-gradient probe
        if step < self.cfg["densify_until"]:
            outputs["info"]["means2d"].retain_grad()

    def after_backward(self, params, optimizers, step, outputs):
        c = self.cfg
        if step < c["densify_until"]:
            self.accumulate(outputs)
            if step > c["densify_from"] and step % c["interval"] == 0:
                self.densify_and_prune(params, optimizers, step)
        if (c["opacity_mode"] == "reset" and step < c["reset_until"]
                and step % c["reset_every"] == 0):
            reset_opa(params, optimizers, {}, value=c["reset_value"])

    # ------------------------------------------------------------------ statistics
    @torch.no_grad()
    def accumulate(self, outputs):
        info = outputs["info"]
        if self.cfg["score"] == "error":
            if ERROR_KEY not in outputs:
                raise RuntimeError("revdens-aux loss did not provide the per-pixel error map")
            e_view, _ = error_stats(info, outputs[ERROR_KEY])  # E^pi_k, Sec. 3.2
            torch.maximum(self.state["score"], e_view, out=self.state["score"])  # E_k = max_pi
            return
        g = info["means2d"].grad
        if g is None:
            raise RuntimeError("means2d gradient missing; before_backward must retain it")
        g = g[0].clone()
        g[:, 0] *= info["width"] / 2.0  # Inria NDC gradient units (as gsplat DefaultStrategy)
        g[:, 1] *= info["height"] / 2.0
        vis = (info["radii"][0] > 0).all(dim=-1)
        self.state["grad_accum"][vis] += g.norm(dim=-1)[vis]
        self.state["count"][vis] += 1.0

    def scores(self):
        if self.cfg["score"] == "error":
            return self.state["score"], self.cfg["error_threshold"]
        tau = torch.nan_to_num(self.state["grad_accum"] / self.state["count"], nan=0.0)
        # 3DGS uses tau >= threshold; the error rule is strict (E_k > 0.1)
        return tau, self.cfg["grad_threshold"] * (1 - 1e-7)

    # ------------------------------------------------------------------ ADC run
    @torch.no_grad()
    def densify_and_prune(self, params, optimizers, step):
        c = self.cfg
        st = self.state
        st["runs"] += 1
        extent = st["extent"]
        score, thr = self.scores()
        score = score.clone()
        n0 = len(params["means"])

        # 1. prune (3DGS rules): low opacity; world-size once step > big_prune_after
        prune = params["opacities"].sigmoid() < c["min_opacity"]
        if step > c["big_prune_after"]:
            prune |= params["scales"].exp().max(dim=-1).values > c["big_prune_scale"] * extent
        n_pruned = int(prune.sum())
        if n_pruned:
            remove(params, optimizers, {}, prune)
            score = score[~prune]
        n1 = len(params["means"])

        # 2. grow: candidates above threshold, top-score within the budget (Sec. 3.4)
        if c["growth_control"]:
            budget = core.growth_budget(n1, st["cap"], c["growth_fraction"])
        else:
            budget = n1
        sel, n_cand = core.select_growth(score, thr, budget)
        max_scale = params["scales"].exp().max(dim=-1).values
        large = max_scale > c["percent_dense"] * extent
        split_mask = sel & large
        clone_mask = sel & ~large

        # clone: source and copy get alpha_hat = 1 - sqrt(1 - alpha) (Sec. 3.3)
        if c["opacity_correction"] and clone_mask.any():
            o = params["opacities"][clone_mask].sigmoid()
            params["opacities"].data[clone_mask] = torch.logit(core.corrected_opacity(o), eps=1e-6)
        duplicate(params, optimizers, {}, clone_mask)
        n_clone = int(clone_mask.sum())

        # split: 3DGS rule (2 samples from G_k, scale / 1.6), opacity preserved (Sec. 3.3)
        pad = torch.zeros(len(params["means"]), dtype=torch.bool, device=sel.device)
        pad[:n1] = split_mask
        split(params, optimizers, {}, pad, revised_opacity=False)
        n_split = int(split_mask.sum())

        # 3. opacity decrement instead of hard reset (Sec. 3.5)
        if c["opacity_mode"] == "decrement":
            o = params["opacities"].sigmoid()
            params["opacities"].data.copy_(
                torch.logit(core.decrement_opacity(o, c["opacity_decrement"]), eps=1e-6))

        n = len(params["means"])
        dev = params["means"].device
        for k in ("score", "grad_accum", "count"):
            st[k] = torch.zeros(n, device=dev)
        row = {"revdens": "adc", "step": step, "run": st["runs"], "before": n0,
               "pruned": n_pruned, "candidates": n_cand, "budget": budget, "clone": n_clone,
               "split": n_split, "after": n, "score_max": float(score.max()) if len(score) else 0.0,
               "score_median": float(score.median()) if len(score) else 0.0}
        self.history.append(row)
        if c["log"]:
            print(json.dumps(row), flush=True)
        return row


def create_density(spec):
    return RevisedADC(spec)
