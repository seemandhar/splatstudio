"""FastGS multi-view consistent densification (VCD) and pruning (VCP), arXiv 2511.04283
Sec. 3.2-3.3 and supp. Sec. 8.1.

Port of the official code https://github.com/fastgs/FastGS @ 44e02a5:
`train.py` (training loop), `utils/fast_utils.py` (sampling_cameras, get_loss,
compute_photometric_loss, compute_gaussian_score_fastgs) and `scene/gaussian_model.py`
(densify_and_prune_fastgs, densify_and_clone_fastgs, densify_and_split_fastgs,
final_prune_fastgs, reset_opacity, training_setup, optimizer_step).
Population changes go through gsplat.strategy.ops so params and Adam moments stay aligned
(new rows get zero moments, like cat_tensors_to_optimizer).

Per training step s (the runtime calls `after_backward` after the optimizer step; the official
loop densifies before it — the same Gaussians and statistics, one step later):
  s < 15000: accumulate ||g[:, :2]|| (3DGS) and ||g[:, 2:]|| (AbsGS) for radii > 0;
    s > 500 and s % interval == 0: scores (K views) -> clone -> split -> prune -> clamp o<=0.8;
    s % 3000 == 0: opacity reset to min(o, 0.01).
  s % 3000 == 0 and 15000 < s < 30000: late VCP prune (o < 0.1 or s_p > 0.9).
"""

import json
import random

import torch
from gsplat.strategy.ops import duplicate, remove, reset_opa, split

from . import scenes
from .losses import photometric
from .optim import ScheduledAdam, wrap
from .raster import render
from .renderer import FASTGS_GRADIENT, TARGET_CAPABILITY

DEFAULTS = {
    "table": "base",  # per-scene argument table: "base" (train_base.sh) or "big"
    "scene_args": {},  # explicit overrides of the per-scene arguments (keys of scenes.ARG_DEFAULTS)
    "densify_from": 500,  # densify_from_iter (strict >)
    "densify_until": 15000,  # densify_until_iter (strict <)
    "densification_interval": 500,  # train_base.sh 500, train_big.sh 100
    "cams": 10,  # K, fast_utils.sampling_cameras num_cams
    "importance_threshold": 5,  # tau_d, gaussian_model.py `importance_score > 5`
    "grad_thresh": 0.0002,  # clone gradient threshold (3DGS)
    "min_opacity": 0.005,  # densify_and_prune_fastgs(min_opacity=0.005)
    "big_scale": 0.1,  # big_points_ws: max scale > 0.1 * extent
    "big_prune_after": 3000,  # size_threshold set when iteration > opacity_reset_interval
    "prune_fraction": 0.5,  # remove_budget = int(0.5 * to_remove)
    "opacity_cap": 0.8,  # post-densification clamp
    "reset_every": 3000,  # opacity_reset_interval
    "reset_value": 0.01,
    "late_prune_every": 3000,  # train.py: iteration % 3000 == 0 and 15000 < it < 30000
    "late_prune_after": 15000,
    "late_prune_until": 30000,
    "late_min_opacity": 0.1,
    "late_score_threshold": 0.9,  # tau_p
    "lambda_dssim": 0.2,
    "optimizer_schedule": True,  # 3DGS-accel optimizer_step schedule (optim.py)
    "lr": {"opacities": 0.025, "scales": 0.005, "quats": 0.001},
    "mult": 0.5,  # CB beta of the score renders; must equal the renderer's
    "background": [0.0, 0.0, 0.0],  # official black background of the score renders
    "seed": 0,
    "log": True,
}
REQUIRES = {"gradient.screen.signed", FASTGS_GRADIENT, TARGET_CAPABILITY}
BASE_FIELDS = ("means", "scales", "quats", "opacities", "sh0", "shN")


# ------------------------------------------------------------------------- pure functions
def error_mask(image_hwc, gt_hwc, loss_thresh):
    """Eq. 1-3 / fast_utils.get_loss: channel-mean |r - g|, min-max normalised, > tau."""
    e = (image_hwc - gt_hwc).abs().mean(dim=-1)
    e = (e - e.min()) / (e.max() - e.min())
    return (e > loss_thresh).int()


def combine_scores(counts, photos, num_views):
    """Eq. 4 and Eq. 6 from per-view counts [V, N] (int) and E_photo [V]:
    s_d = floor(sum_j c_j / K); s_p = minmax(sum_j c_j * E_j)."""
    counts = torch.as_tensor(counts)
    photos = torch.as_tensor(photos, dtype=torch.float32, device=counts.device)
    total = counts.sum(dim=0)
    importance = torch.div(total, num_views, rounding_mode="floor")
    raw = (counts.float() * photos[:, None]).sum(dim=0)
    pruning = (raw - raw.min()) / (raw.max() - raw.min())
    return importance, pruning


def prune_weights(pruning_score, n_current):
    """densify_and_prune_fastgs: padded_importance[:len(score)] = 1 / (1e-6 + (1 - s_p)),
    positional over the post-densification arrays, 0 for the remaining rows."""
    w = torch.zeros(n_current, dtype=torch.float32, device=pruning_score.device)
    m = min(len(pruning_score), n_current)
    w[:m] = 1.0 / (1e-6 + (1.0 - pruning_score[:m].float()))
    return torch.nan_to_num(w, nan=0.0, posinf=0.0)  # guard: official crashes on NaN


def densify_steps(cfg, steps=30000):
    return [s for s in range(1, steps + 1) if s < cfg["densify_until"]
            and s > cfg["densify_from"] and s % cfg["densification_interval"] == 0]


def reset_steps(cfg, steps=30000):
    return [s for s in range(1, steps + 1) if s < cfg["densify_until"]
            and s % cfg["reset_every"] == 0]


def late_prune_steps(cfg, steps=30000):
    return [s for s in range(1, steps + 1) if s % cfg["late_prune_every"] == 0
            and cfg["late_prune_after"] < s < cfg["late_prune_until"]]


class FastGSDensity:
    def __init__(self, spec):
        unknown = set(spec.config) - set(DEFAULTS)
        if unknown:
            raise ValueError(f"Unknown FastGS density options: {sorted(unknown)}")
        c = {**DEFAULTS, **spec.config}
        if c["table"] not in scenes.TABLES:
            raise ValueError(f"table must be one of {sorted(scenes.TABLES)}")
        bad = set(c["scene_args"]) - set(scenes.ARG_DEFAULTS)
        if bad:
            raise ValueError(f"Unknown scene_args {sorted(bad)}")
        self.cfg = c
        self.requires = set(REQUIRES)
        self.state = None
        self.cache = {}
        self.n_train = None
        self.rng = random.Random(int(c["seed"]))
        self.generator = None
        self.history = []
        self.scene = None
        self.args = None
        self.render_fn = None  # (params, view, sh_degree, metric_map|None) -> (CHW, counts)

    # ---------------------------------------------------------------- lifecycle
    def initialize(self, params, optimizers, scene_scale=1.0):
        extra = set(params) - set(BASE_FIELDS)
        if extra:
            raise ValueError(f"FastGS density does not know fields {sorted(extra)}")
        c = self.cfg
        self.extent = float(scene_scale)
        n = len(params["means"])
        self.scene, args = scenes.resolve(n, c["table"])
        args.update(c["scene_args"])
        if abs(float(args["mult"]) - float(c["mult"])) > 1e-9:
            raise ValueError(
                f"Scene {self.scene!r} uses CB mult {args['mult']} in the official scripts but "
                f"this method is configured with {c['mult']}; use the -tdb variant (0.7) or the "
                f"m360 variant (0.5)")
        self.args = args
        self._reset_stats(n, params["means"].device)
        lrs = {**c["lr"], "sh0": args["lowfeature_lr"], "shN": args["highfeature_lr"] / 20.0}
        for name, lr in lrs.items():
            optimizers[name].param_groups[0]["lr"] = float(lr)
        if c["optimizer_schedule"]:
            for name in BASE_FIELDS:
                wrap(optimizers, name, "sh" if name == "shN" else "main")
        self.generator = torch.Generator(device=params["means"].device)
        self.generator.manual_seed(int(c["seed"]))
        if c["log"]:
            print(json.dumps({"fastgs_scene": {"scene": self.scene, "initial": n, **args,
                                               "extent": self.extent}}), flush=True)

    def set_training_cameras(self, cameras):
        self.n_train = len(cameras)

    def before_backward(self, params, optimizers, step, outputs):
        for opt in optimizers.values():
            if isinstance(opt, ScheduledAdam):
                opt.current_step = step
        if step < self.cfg["densify_until"]:
            outputs["info"]["means2d"].retain_grad()
            self._cache_view(outputs)

    @torch.no_grad()
    def after_backward(self, params, optimizers, step, outputs):
        c = self.cfg
        if step < c["densify_until"]:
            self._accumulate(outputs["info"])
            if step > c["densify_from"] and step % c["densification_interval"] == 0:
                self.densify(params, optimizers, step)
            if step % c["reset_every"] == 0:
                reset_opa(params, optimizers, {}, value=c["reset_value"])
                self._check_trainable(params)
        if (step % c["late_prune_every"] == 0
                and c["late_prune_after"] < step < c["late_prune_until"]):
            self.late_prune(params, optimizers, step)

    # ---------------------------------------------------------------- statistics
    def _reset_stats(self, n, device):
        self.state = {
            "grad_accum": torch.zeros(n, device=device),
            "grad_accum_abs": torch.zeros(n, device=device),
            "denom": torch.zeros(n, device=device),
        }

    def _cache_view(self, outputs):
        image = outputs.get("target_image")
        if image is None or (self.n_train is not None and len(self.cache) >= self.n_train):
            return
        cam = outputs["camera"]
        key = tuple(round(v, 4) for v in cam["viewmat"].flatten().tolist())
        if key in self.cache:
            return
        self.cache[key] = {
            "K": cam["K"].detach().clone(),
            "viewmat": cam["viewmat"].detach().clone(),
            "width": int(cam["width"]),
            "height": int(cam["height"]),
            # the dataset stores 8-bit images: uint8 caching is exact
            "image": (image.detach() * 255).round().clamp(0, 255).to(torch.uint8).cpu(),
        }

    @torch.no_grad()
    def _accumulate(self, info):
        grad = info["means2d"].grad
        if grad is None:
            return
        vis = info["radii"] > 0
        self.state["grad_accum"][vis] += grad[vis, :2].norm(dim=-1)
        self.state["grad_accum_abs"][vis] += grad[vis, 2:].norm(dim=-1)
        self.state["denom"][vis] += 1

    # ---------------------------------------------------------------- multi-view scores
    def _default_render(self, params, view, sh_degree, metric_map=None):
        dev = params["means"].device
        bg = torch.tensor(self.cfg["background"], dtype=torch.float32, device=dev)
        cam = {"K": view["K"].to(dev), "viewmat": view["viewmat"].to(dev),
               "width": view["width"], "height": view["height"]}
        color, _, _, stats = render(params, cam, sh_degree, bg, self.cfg["mult"], metric_map)
        return color, stats["counts"]

    def sample_views(self):
        keys = list(self.cache)
        if not keys:
            raise RuntimeError("FastGS scores need at least one cached training view")
        k = min(int(self.cfg["cams"]), len(keys))
        return [self.cache[key] for key in self.rng.sample(keys, k)]

    @torch.no_grad()
    def scores(self, params, step, views=None):
        """compute_gaussian_score_fastgs: (importance s_d [N] int, pruning s_p [N] float)."""
        if views is None:
            views = self.sample_views()
        render_fn = self.render_fn or self._default_render
        sh_degree = min(step // 1000, 3)  # official active_sh_degree at this iteration
        dev = params["means"].device
        counts, photos = [], []
        self.last_views = []
        for view in views:
            image, _ = render_fn(params, view, sh_degree, None)
            img = image.permute(1, 2, 0)
            gt = view["image"].to(dev).float() / 255.0
            photo = photometric(img, gt, self.cfg["lambda_dssim"])  # Eq. 5
            mask = error_mask(img, gt, self.args["loss_thresh"])  # Eq. 1-3
            _, cnt = render_fn(params, view, sh_degree, mask)
            counts.append(cnt.to(dev))
            photos.append(float(photo))
            self.last_views.append({"photometric": float(photo),
                                    "high_error_pixels": int(mask.sum())})
        return combine_scores(torch.stack(counts), photos, len(views))

    # ---------------------------------------------------------------- densification
    def _grads(self):
        d = self.state["denom"]
        g = torch.nan_to_num(self.state["grad_accum"] / d, nan=0.0)
        a = torch.nan_to_num(self.state["grad_accum_abs"] / d, nan=0.0)
        return g, a

    @torch.no_grad()
    def densify(self, params, optimizers, step, importance=None, pruning=None):
        c, a = self.cfg, self.args
        if importance is None or pruning is None:
            importance, pruning = self.scores(params, step)
        dev = params["means"].device
        grads, grads_abs = self._grads()
        max_scale = params["scales"].exp().max(dim=-1).values
        small = max_scale <= a["dense"] * self.extent
        metric = importance > c["importance_threshold"]
        clones = (grads >= c["grad_thresh"]) & small & metric
        splits = (grads_abs >= a["grad_abs_thresh"]) & ~small & metric
        n_clone, n_split = int(clones.sum()), int(splits.sum())
        if n_clone:
            duplicate(params, optimizers, self.state, clones)
        if n_split:
            pad = torch.zeros(n_clone, dtype=torch.bool, device=dev)
            split(params, optimizers, self.state, torch.cat([splits, pad]))
        n = len(params["means"])
        self._reset_stats(n, dev)  # densification_postfix

        prune = params["opacities"].sigmoid() < c["min_opacity"]
        if step > c["big_prune_after"]:
            # the official max_radii2D > 20 test never fires (densification_postfix zeroed
            # max_radii2D just before); only the world-space test acts.
            prune |= params["scales"].exp().max(dim=-1).values > c["big_scale"] * self.extent
        n_cand = int(prune.sum())
        budget = int(c["prune_fraction"] * n_cand)
        removed = 0
        if budget:
            w = prune_weights(pruning, n)
            budget = min(budget, int((w > 0).sum()))
            sel = torch.multinomial(w, budget, replacement=False, generator=self.generator)
            chosen = torch.zeros(n, dtype=torch.bool, device=dev)
            chosen[sel] = True
            final = prune & chosen
            removed = int(final.sum())
            if removed:
                remove(params, optimizers, self.state, final)
        reset_opa(params, optimizers, {}, value=c["opacity_cap"])  # min(o, 0.8)
        self._check_trainable(params)
        row = {"step": step, "metric_pass": int(metric.sum()), "clones": n_clone,
               "splits": n_split, "prune_candidates": n_cand, "pruned": removed,
               "count": len(params["means"]),
               "mean_high_error_pixels": sum(v["high_error_pixels"] for v in
                                             getattr(self, "last_views", [])) /
               max(1, len(getattr(self, "last_views", [])))}
        self._log("fastgs_densify", row)
        if dev.type == "cuda":
            torch.cuda.empty_cache()
        return row

    @torch.no_grad()
    def late_prune(self, params, optimizers, step, pruning=None):
        c = self.cfg
        if pruning is None:
            _, pruning = self.scores(params, step)
        mask = (params["opacities"].sigmoid() < c["late_min_opacity"]) | (
            pruning > c["late_score_threshold"])
        removed = int(mask.sum())
        if removed:
            remove(params, optimizers, self.state, mask)
        self._check_trainable(params)
        row = {"step": step, "pruned": removed, "count": len(params["means"])}
        self._log("fastgs_late_prune", row)
        return row

    # ---------------------------------------------------------------- helpers
    def _log(self, key, row):
        self.history.append({key: row})
        if self.cfg["log"]:
            print(json.dumps({key: row}), flush=True)

    @staticmethod
    def _check_trainable(params):
        # requires_grad pitfall (TODO-runtime.md): every field must stay a trainable leaf.
        for name, p in params.items():
            if not (isinstance(p, torch.nn.Parameter) and p.requires_grad and p.is_leaf):
                raise RuntimeError(f"FastGS density left field {name} non-trainable")


def create_density(spec):
    return FastGSDensity(spec)
