"""StableGS density controller: Taming-3DGS budgeted score densification as modified by StableGS,
plus the pair schedule (official train_pair.py, scene/gaussian_model.py:densify_with_score).

Reuses the matched Taming-3DGS port (`ss_taming_3dgs.density.TamingDensity`: growth curve, score,
sampled clone/split, guided prune, opacity reset, α freeze at `ho_iteration`, 16-step SH Adam,
learning-rate overrides). StableGS-specific changes:
  * statistics from both pair views each iteration (add_densification_stats twice, Inria NDC
    screen gradients of the RaDe-GS kernel, so no pixel rescaling);
  * GOF qualifier (gaussian_model.py:546-559, disabled by `--nogof` for Deep Blending):
    qualifies if ‖g‖ ≥ τ OR g_abs ≥ quantile(g_abs, 1 − mean(‖g‖ ≥ τ));
  * all training views (with images) are available to the score from step 1 (official
    `scene.getTrainCameras()`), instead of the Taming port's cache of seen views;
  * the pair stack (pairs.py) and the dual/geometric switch (iteration > 15000).
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from gsplat.strategy.ops import duplicate, remove, split
from ss_taming_3dgs import core
from ss_taming_3dgs.density import DEFAULTS as TAMING_DEFAULTS
from ss_taming_3dgs.density import TamingDensity

from . import pairs as P
from .link import LINK
from .renderer import PAIR_CAPABILITY

# run.sh --budget values (= Taming full_eval.py big budgets = final 3DGS counts).
RUN_SH_BUDGETS = dict(core.BIG_BUDGETS)

DEFAULTS = {
    **TAMING_DEFAULTS,
    "budget_table": core.table_by_count(RUN_SH_BUDGETS),
    "gof": True,  # run.sh: --nogof only for drjohnson / playroom
    "dual_from": 15000,  # dual_alpha = iteration > 15000; geometric path losses likewise
    "pair_min_covisible": 30,
    "pair_min_rotation": 16.0,
    "pair_max_rotation": 60.0,
    "pair_max_h_ratio": 0.8,
    "pair_h_threshold": 4.0,
    "pairs": None,  # explicit list of [i, j] training-index pairs (probes / tests)
    "data": None,
    "downscale": None,
}


def _argv_value(flag):
    argv = sys.argv
    for i, arg in enumerate(argv):
        if arg == flag and i + 1 < len(argv):
            return argv[i + 1]
        if arg.startswith(flag + "="):
            return arg.split("=", 1)[1]
    return None


def gof_qualifiers(grads, grads_abs, threshold, use_gof=True):
    """gaussian_model.py:densify_with_score qualifier (grads = ‖accum‖/denom, grads_abs alike)."""
    qual = grads >= threshold
    if not use_gof:
        return qual
    ratio = qual.float().mean()
    q = torch.quantile(grads_abs.reshape(-1).float(), 1 - ratio)
    return qual | (grads_abs >= q)


class StableGSDensity(TamingDensity):
    def __init__(self, spec):
        unknown = set(spec.config) - set(DEFAULTS)
        if unknown:
            raise ValueError(f"Unknown stablegs-density options: {sorted(unknown)}")
        taming_cfg = {k: v for k, v in spec.config.items() if k in TAMING_DEFAULTS}
        taming_cfg.setdefault("budget_table", DEFAULTS["budget_table"])

        class _Spec:
            config = taming_cfg

        super().__init__(_Spec)
        self.sc = {**DEFAULTS, **spec.config}
        self.requires = {PAIR_CAPABILITY, "gradient.screen.gof"}
        self.np_rng = np.random.default_rng(int(self.sc["seed"]))
        self.cameras = None
        self.images = None
        self.pairs = None
        self.pair_set = set()
        self.stack = []
        self.batch = None
        self.load_seconds = 0.0
        LINK.attach(self)

    # ---------------------------------------------------------------- lifecycle
    def initialize(self, params, optimizers, scene_scale=1.0):
        super().initialize(params, optimizers, scene_scale)
        n = len(params["means"])
        self.state["abs_accum"] = torch.zeros(n, device=params["means"].device)
        if self.pairs is not None:
            self.state["pairs"] = [list(p) for p in self.pairs]

    def set_training_cameras(self, cameras):
        """Runtime hook: reload the training images (not exposed by the runtime), build P."""
        super().set_training_cameras(cameras)
        root = self.sc["data"] or _argv_value("--data")
        if root is None:
            raise RuntimeError("stablegs-density needs the training images: set density config "
                               "'data' or run through ss-train --data")
        downscale = int(self.sc["downscale"] or _argv_value("--downscale") or 1)
        from splatstudio.data import load_dataset

        t0 = time.perf_counter()
        dataset = load_dataset(Path(root), downscale)
        names = dataset["splits"]["train"]
        if len(names) != len(cameras):
            raise RuntimeError("Reloaded training split does not match the runtime's cameras")
        images = []
        for name, cam in zip(names, cameras):
            frame = dataset["frames"][name]
            dev = cam["K"].device
            same = (frame["width"] == cam["width"] and frame["height"] == cam["height"]
                    and torch.allclose(frame["viewmat"].to(dev), cam["viewmat"]))
            if not same:
                raise RuntimeError(f"Reloaded training view {name} differs from the runtime camera")
            images.append(frame["image"])
        del dataset
        if self.sc["pairs"] is not None:
            pairs = [tuple(int(v) for v in p) for p in self.sc["pairs"]]
        else:
            colmap = P.find_colmap(root)
            obs = P.read_images_bin(colmap / "images.bin")
            base = [Path(n).name for n in names]
            missing = [b for b in base if b not in obs]
            if missing:
                raise RuntimeError(f"Training images missing from COLMAP model: {missing[:3]}")
            rots = np.stack([c["viewmat"][:3, :3].double().cpu().numpy() for c in cameras])
            pairs = P.select_pairs(base, rots, obs, self.sc["pair_min_covisible"],
                                   self.sc["pair_min_rotation"], self.sc["pair_max_rotation"],
                                   self.sc["pair_max_h_ratio"], self.sc["pair_h_threshold"])
        self.set_training_frames(cameras, images, pairs)
        self.load_seconds = time.perf_counter() - t0
        print(json.dumps({"stablegs_pairs": {"count": len(pairs), "views": len(cameras),
                                             "seconds": round(self.load_seconds, 1)}}), flush=True)

    def set_training_frames(self, cameras, images, pairs):
        if len(cameras) != len(images) or not cameras:
            raise ValueError("cameras and images must be non-empty and aligned")
        if not pairs:
            raise RuntimeError("StableGS pair selection produced no pairs")
        dev = cameras[0]["K"].device
        self.cameras = [dict(c) for c in cameras]
        self.images = [(im.to(dev) * 255).round().clamp(0, 255).to(torch.uint8) for im in images]
        self.pairs = [tuple(p) for p in pairs]
        self.pair_set = {tuple(p) for p in self.pairs}
        if self.state is not None:
            self.state["pairs"] = [list(p) for p in self.pairs]  # saved in the checkpoint
        # Score views: every training view (official compute_gaussian_score over train cameras).
        self.cache = {}
        for k, (cam, im) in enumerate(zip(self.cameras, self.images)):
            img8 = im.cpu()
            self.cache[k] = {"K": cam["K"].detach().clone(), "viewmat": cam["viewmat"].clone(),
                             "width": int(cam["width"]), "height": int(cam["height"]),
                             "image": img8, "edges": core.edge_map(img8)}
        self.stack = []
        self.batch = self._make_batch(1)

    # ---------------------------------------------------------------- pairs
    def _pop_pair(self):
        if not self.stack:
            self.stack = P.refill_stack(self.pairs, len(self.cameras), self.np_rng)
        return self.stack.pop(int(self.rng.randint(0, len(self.stack) - 1)))

    def _make_batch(self, step):
        i, j = self._pop_pair()
        on = step > int(self.sc["dual_from"])
        views = [{"index": k, "camera": self.cameras[k], "target": self.images[k].float() / 255}
                 for k in (i, j)]
        return {"step": step, "views": views, "indices": (i, j),
                "in_P": (i, j) in self.pair_set, "dual": on, "geo": on}

    def current_batch(self):
        return self.batch if self.images is not None else None

    # ---------------------------------------------------------------- statistics
    def before_backward(self, params, optimizers, step, outputs):
        for m in outputs["info"]["means2d"]:
            if m.requires_grad:
                m.retain_grad()

    def after_backward(self, params, optimizers, step, outputs):
        super().after_backward(params, optimizers, step, outputs)
        if self.images is not None:
            self.batch = self._make_batch(step + 1)

    @torch.no_grad()
    def _accumulate(self, info):
        """add_densification_stats for each rendered view (Inria NDC gradient, [N,3])."""
        for m, radii in zip(info["means2d"], info["radii"]):
            if m.grad is None:
                continue
            vis = radii > 0
            self.state["grad_accum"][vis] += m.grad[vis, :2].norm(dim=-1)
            self.state["abs_accum"][vis] += m.grad[vis, 2:].norm(dim=-1)
            self.state["count"][vis] += 1

    def _grads_abs(self):
        return torch.nan_to_num(self.state["abs_accum"] / self.state["count"], nan=0.0)

    # ---------------------------------------------------------------- densification
    @torch.no_grad()
    def densify(self, params, optimizers, step, scores=None):
        """TamingDensity.densify with the StableGS GOF qualifier (densify_with_score)."""
        c = self.cfg
        if scores is None:
            scores = self.score(params, step)
        if self.counts is None:
            self.final_budget = core.resolve_budget(
                self.start_count, c["mode"], c["budget"], c["budget_table"])
            self.counts = core.count_array(
                self.start_count, self.final_budget, c["densify_from"], c["densify_until"],
                c["densification_interval"])
        n_old = len(params["means"])
        qual = gof_qualifiers(self._grads(), self._grads_abs(), c["grad_threshold"],
                              bool(self.sc["gof"]))
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
        n = len(params["means"])
        for key in ("grad_accum", "abs_accum", "count"):  # densification_postfix
            self.state[key] = torch.zeros(n, device=scores.device)

        prune = params["opacities"].sigmoid() < c["min_opacity"]
        if step > c["big_prune_after"]:
            # max_radii2D > 20 never fires (zeroed by densification_postfix), see Taming port.
            prune |= params["scales"].exp().max(dim=-1).values > c["big_scale"] * self.extent
        remove_budget = int(c["prune_fraction"] * int(prune.sum()))
        removed = 0
        if remove_budget and self.calls < c["guided_prune_calls"]:
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
        row = {"step": step, "target": target, "clone_candidates": n_clone,
               "split_candidates": n_split, "clones": len(clone_idx), "splits": len(split_idx),
               "pruned": removed, "count": len(params["means"])}
        self.history.append(row)
        if c["log"]:
            print(json.dumps({"stablegs_densify": row}), flush=True)
        frozen = [k for k, v in params.items() if not v.requires_grad]
        if frozen:  # requires_grad pitfall (TODO-runtime.md): every field must stay trainable
            raise RuntimeError(f"density op froze fields {frozen}")
        self.calls += 1
        if scores.is_cuda:
            torch.cuda.empty_cache()


def create_density(spec):
    return StableGSDensity(spec)
