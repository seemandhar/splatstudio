"""Taming 3DGS budgeted, score-guided densification (arXiv 2406.15643 Sec. 3.2-3.4, Alg. 1).

Port of the official code https://github.com/humansensinglab/taming-3dgs (fd0f7d9):
`train.py` (densification loop, opacity reset, ho_iteration, sh_lower),
`utils/taming_utils.py` (compute_gaussian_score, get_loss_map, get_count_array) and
`scene/gaussian_model.py` (densify_with_score, *_taming, modify_functions, training_setup).
Population changes go through gsplat.strategy.ops so params and Adam moments stay aligned.
"""

import json
import random

import torch
from gsplat import rasterization
from gsplat.strategy.ops import duplicate, remove, reset_opa, split

from . import core
from .kernels import pixel_stats
from .optim import IntervalAdam
from .renderer import TARGET_CAPABILITY

DEFAULTS = {
    # C1 growth curve / budget (full_eval.py "big" mode = "Ours (#3DGS)")
    "mode": "final_count",
    "budget": None,
    "budget_table": core.table_by_count(core.BIG_BUDGETS),
    "densify_from": 500,
    "densify_until": 15000,
    "densification_interval": 100,
    # C2 score-based densification
    "cams": 10,
    "score_weights": dict(core.SCORE_WEIGHTS),
    "grad_threshold": 0.0002,
    "percent_dense": 0.01,
    "min_opacity": 0.005,
    "big_scale": 0.1,
    "big_prune_after": 3000,
    "prune_fraction": 0.5,
    "guided_prune_calls": 27,
    # 3DGS opacity reset (kept by Taming)
    "reset_every": 3000,
    "reset_value": 0.01,
    # C3 high-opacity switch: "freeze" = official modify_functions(); "off" = ablation
    "ho_iteration": 15000,
    "high_opacity": "freeze",
    # C4 batched SH updates and learning rates (arguments/__init__.py, train.py --sh_lower)
    "sh_update_every": 16,
    "lr": {"opacities": 0.025, "shN": 0.005 / 20.0},
    "seed": 0,
    "log": True,  # one JSON line per densification call on stdout (train log)
}


class TamingDensity:
    def __init__(self, spec):
        unknown = set(spec.config) - set(DEFAULTS)
        if unknown:
            raise ValueError(f"Unknown Taming density options: {sorted(unknown)}")
        c = {**DEFAULTS, **spec.config}
        self.cfg = c
        self.weights = {**core.SCORE_WEIGHTS, **c["score_weights"]}
        if set(self.weights) != set(core.SCORE_WEIGHTS):
            raise ValueError("Unknown score weight names")
        if c["high_opacity"] not in ("freeze", "off"):
            raise ValueError("high_opacity must be 'freeze' or 'off'")
        if c["mode"] not in ("final_count", "multiplier"):
            raise ValueError("mode must be 'final_count' or 'multiplier'")
        self.requires = {"gradient.screen.signed", TARGET_CAPABILITY}
        self.state = None
        self.cache = {}
        self.n_train = None
        self.rng = random.Random(int(c["seed"]))
        self.calls = 0
        self.counts = None
        self.history = []  # (step, target, count after) for logs/probes

    # ---------------------------------------------------------------- lifecycle
    def initialize(self, params, optimizers, scene_scale=1.0):
        c = self.cfg
        self.extent = float(scene_scale)
        self.start_count = len(params["means"])
        self.counts = None  # growth curve, resolved at the first densification (see below)
        n = self.start_count
        dev = params["means"].device
        self.state = {
            "grad_accum": torch.zeros(n, device=dev),
            "count": torch.zeros(n, device=dev),
        }
        for name, lr in c["lr"].items():
            optimizers[name].param_groups[0]["lr"] = float(lr)
        if int(c["sh_update_every"]) > 1:
            old = optimizers["shN"]
            optimizers["shN"] = IntervalAdam(
                [params["shN"]], lr=old.param_groups[0]["lr"], every=int(c["sh_update_every"]),
                eps=old.param_groups[0]["eps"],
            )

    def set_training_cameras(self, cameras):
        self.n_train = len(cameras)

    def before_backward(self, params, optimizers, step, outputs):
        outputs["info"]["means2d"].retain_grad()
        if step < self.cfg["densify_until"]:
            self._cache_view(outputs)

    def after_backward(self, params, optimizers, step, outputs):
        c = self.cfg
        if step < c["densify_until"]:
            self._accumulate(outputs["info"])
            if step > c["densify_from"] and step % c["densification_interval"] == 0:
                self.densify(params, optimizers, step)
            if step % c["reset_every"] == 0:
                reset_opa(params, optimizers, self.state, value=c["reset_value"])
        if step == c["ho_iteration"] and c["high_opacity"] == "freeze":
            # official modify_functions(): _opacity becomes a plain tensor detached from the
            # optimizer -> opacities stay at their 15k values (abs(sigmoid(x)) == sigmoid(x)).
            optimizers["opacities"].param_groups[0]["lr"] = 0.0

    # ---------------------------------------------------------------- statistics
    def _cache_view(self, outputs):
        image = outputs.get("target_image")
        if image is None or (self.n_train is not None and len(self.cache) >= self.n_train):
            return
        cam = outputs["camera"]
        key = tuple(round(v, 4) for v in cam["viewmat"].flatten().tolist())
        if key in self.cache:
            return
        img8 = (image.detach() * 255).round().clamp(0, 255).to(torch.uint8).cpu()
        self.cache[key] = {
            "K": cam["K"].detach().clone(),
            "viewmat": cam["viewmat"].detach().clone(),
            "width": int(cam["width"]),
            "height": int(cam["height"]),
            "image": img8,
            "edges": core.edge_map(img8),
        }

    @torch.no_grad()
    def _accumulate(self, info):
        grad = info["means2d"].grad
        if grad is None:
            return
        g = grad[0].clone()
        g[:, 0] *= info["width"] / 2.0
        g[:, 1] *= info["height"] / 2.0
        vis = (info["radii"][0] > 0).all(dim=-1)
        self.state["grad_accum"][vis] += g[vis].norm(dim=-1)
        self.state["count"][vis] += 1

    def _grads(self):
        return torch.nan_to_num(self.state["grad_accum"] / self.state["count"], nan=0.0)

    @torch.no_grad()
    def _render(self, params, view, sh_degree):
        dev = params["means"].device
        colors = torch.cat((params["sh0"], params["shN"]), dim=1)
        render, alpha, meta = rasterization(
            means=params["means"], quats=params["quats"], scales=params["scales"].exp(),
            opacities=params["opacities"].sigmoid(), colors=colors,
            viewmats=view["viewmat"].to(dev)[None], Ks=view["K"].to(dev)[None],
            width=view["width"], height=view["height"], sh_degree=sh_degree, packed=False,
        )
        return render[0] + (1 - alpha[0]), meta  # white background (dataset convention)

    @torch.no_grad()
    def score(self, params, step, views=None):
        """S_G of Eq. 4 / compute_gaussian_score summed over `cams` cached training views."""
        if views is None:
            keys = list(self.cache)
            views = [self.cache[k] for k in self.rng.sample(keys, min(self.cfg["cams"], len(keys)))]
        if not views:
            raise RuntimeError("Taming score needs at least one cached training view")
        w = self.weights
        n = len(params["means"])
        dev = params["means"].device
        total = torch.zeros(n, device=dev)
        opac = params["opacities"].sigmoid()
        scale_prod = params["scales"].exp().prod(dim=-1)
        grads = self._grads() if self.state is not None else torch.zeros(n, device=dev)
        sh_degree = min((step - 1) // 1000, 3)
        self.last_views = []
        for view in views:
            rgb, meta = self._render(params, view, sh_degree)
            gt = view["image"].to(dev).float() / 255.0
            p_i = core.photometric(rgb, gt)
            sal = core.saliency(
                rgb, gt, core.minmax(view["edges"].to(dev).float() / 255.0),
                w["mse_importance"], w["edge_importance"],
            )
            stats = pixel_stats(meta, sal)
            vis = (meta["radii"][0] > 0).all(dim=-1)
            depth = torch.where(vis, meta["depths"][0], torch.zeros_like(meta["depths"][0]))
            radius = torch.where(vis, meta["radii"][0].max(dim=-1).values.float(), 0.0)
            g = core.gaussian_terms(w, grads, opac, scale_prod, depth, radius)
            p = core.pixel_terms(w, stats)
            agg = w["view_importance"] * p_i * (p + g)
            total[vis] += agg[vis]
            self.last_views.append({"photometric": float(p_i), "visible": int(vis.sum())})
        return total

    # ---------------------------------------------------------------- densification
    @torch.no_grad()
    def densify(self, params, optimizers, step, scores=None):
        c = self.cfg
        if scores is None:
            scores = self.score(params, step)
        if self.counts is None:
            # Budget lookup is deferred to the first densification so that generic runtime
            # probes on toy scenes can initialize the density; an unknown scene without an
            # explicit `budget` still stops training here with an error.
            self.final_budget = core.resolve_budget(
                self.start_count, c["mode"], c["budget"], c["budget_table"]
            )
            self.counts = core.count_array(
                self.start_count, self.final_budget, c["densify_from"], c["densify_until"],
                c["densification_interval"],
            )
        n_old = len(params["means"])
        grads = self._grads()
        qual = grads >= c["grad_threshold"]
        max_scale = params["scales"].exp().max(dim=-1).values
        small = max_scale <= c["percent_dense"] * self.extent
        clones, splits = qual & small, qual & ~small
        n_clone, n_split = int(clones.sum()), int(splits.sum())
        idx = min(self.calls + 1, len(self.counts) - 1)
        target = self.counts[idx]
        clone_budget, split_budget = core.split_budgets(target, n_old, n_clone, n_split)
        clone_idx = core.sample_without_replacement(scores * clones, clone_budget)
        split_idx = core.sample_without_replacement(scores * splits, split_budget)
        clone_mask = torch.zeros(n_old, dtype=torch.bool, device=scores.device)
        clone_mask[clone_idx] = True
        split_mask = torch.zeros(n_old, dtype=torch.bool, device=scores.device)
        split_mask[split_idx] = True
        if len(clone_idx):
            duplicate(params, optimizers, self.state, clone_mask)
        if len(split_idx):
            pad = torch.zeros(len(clone_idx), dtype=torch.bool, device=scores.device)
            split(params, optimizers, self.state, torch.cat([split_mask, pad]))
        # densification_postfix: statistics restart after every densification
        n = len(params["means"])
        self.state["grad_accum"] = torch.zeros(n, device=scores.device)
        self.state["count"] = torch.zeros(n, device=scores.device)

        prune = params["opacities"].sigmoid() < c["min_opacity"]
        if step > c["big_prune_after"]:
            # official also tests max_radii2D > 20, but densification_postfix has just zeroed
            # max_radii2D, so that term never fires; only the world-space test is effective.
            prune |= params["scales"].exp().max(dim=-1).values > c["big_scale"] * self.extent
        remove_budget = int(c["prune_fraction"] * int(prune.sum()))
        removed = 0
        if remove_budget and self.calls < c["guided_prune_calls"]:
            # official: weights 1/(1e-6+score) indexed by the post-densification order for the
            # first len(score) Gaussians and 0 for the appended ones (reproduced as is).
            weights = torch.zeros(n, device=scores.device)
            m = min(n_old, n)
            weights[:m] = 1.0 / (1e-6 + scores[:m])
            sel = core.sample_without_replacement(weights, remove_budget)
            chosen = torch.zeros(n, dtype=torch.bool, device=scores.device)
            chosen[sel] = True
            final = prune & chosen
            removed = int(final.sum())
            if removed:
                remove(params, optimizers, self.state, final)
        row = {
            "step": step, "target": target, "clone_candidates": n_clone,
            "split_candidates": n_split, "clones": len(clone_idx), "splits": len(split_idx),
            "pruned": removed, "count": len(params["means"]),
        }
        self.history.append(row)
        if c["log"]:
            print(json.dumps({"taming_densify": row}), flush=True)
        self.calls += 1
        if scores.is_cuda:
            torch.cuda.empty_cache()


def create_density(spec):
    return TamingDensity(spec)
