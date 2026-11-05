"""MVGS density controller: owns the multi-view batch schedule and the MVGS densification.

Official code (https://github.com/xiaobiaodu/MVGS @ 36d073a) mapped here:
- train.py:86-112      M views per iteration popped from a shuffled stack, summed loss
- train.py:124-126     densification statistics from the *last* rendered view only
- train.py:128-176     MVAD threshold (camera_t, diffs > 1) and CRD boxes from per-view L1 maps
- train.py:178         opacity reset every 3000 it and at densify_from_iter (white background)
- gaussian_model.py:376-520  clone (gradient OR inside a cross-ray cuboid), split, prune
- run_360.py           cross-intrinsic stages; each stage restarts the iteration counter
"""

import random
import sys
import time
from pathlib import Path

import torch
from gsplat.strategy import ops

from . import geometry as G
from .link import LINK

INDOOR_STAGES = ((48, 8, 3000), (24, 4, 3000), (12, 2, 3000), (8, -1, 30000))
OUTDOOR_STAGES = ((3, -1, 30000),)

DEFAULTS = {
    # Cross-intrinsic guidance: [views M, official -r flag, iterations] (run_360.py MV_normal).
    "stages": [list(s) for s in INDOOR_STAGES],
    # W_full / W_train of the Mip-NeRF 360 indoor scenes (3115 / 1600); used for -r 8/4/2 sizes.
    "full_width_ratio": 1.946875,
    # 3DGS densification (arguments/__init__.py), restarted per stage.
    "grow_grad": 0.0002,
    "densify_from": 500,
    "densify_until": 15000,
    "densify_every": 100,
    "reset_every": 3000,
    "white_reset_at": 500,  # train.py:178 `white_background and iteration == densify_from_iter`
    "reset_value": 0.01,
    "min_opacity": 0.005,
    "percent_dense": 0.01,
    "big_world": 0.1,
    "big_after": 3000,  # size_threshold only when iteration > opacity_reset_interval
    # Position LR: 3DGS exponential decay to 1 %, max_steps 30000, restarted per stage.
    "lr_final_factor": 0.01,
    "lr_max_steps": 30000,
    # MVGS densification.
    "mvad": True,
    "tau": 1.0,
    "crd": True,
    "window": [20, 40],  # (h_box, w_box) train.py:155-156
    "box_index": "official",
    # "last": official (one view per step feeds the statistics); "all": every view.
    "stats_views": "last",
    "seed": 0,
    # Training-image source; default: ss-train's --data / --downscale.
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


class MVGSDensity:
    requires = {"gradient.screen.signed"}

    def __init__(self, spec):
        unknown = set(spec.config) - set(DEFAULTS)
        if unknown:
            raise ValueError(f"Unknown mvgs-density options: {sorted(unknown)}")
        self.c = {**DEFAULTS, **spec.config}
        self.stages = G.check_stages(self.c["stages"])
        if self.c["stats_views"] not in ("last", "all"):
            raise ValueError("stats_views must be 'last' or 'all'")
        if self.c["box_index"] not in ("official", "paper"):
            raise ValueError("box_index must be 'official' or 'paper'")
        self.rng = random.Random(int(self.c["seed"]))
        self.cameras = None  # native training cameras
        self.native = None  # list of uint8 HWC GPU images
        self.cache_stage = None
        self.cache = None  # (cameras, uint8 images) at the cached stage resolution
        self.stack = []
        self.batch = None
        self.history = []
        self.state = {"history": self.history}
        self.load_seconds = 0.0
        LINK.attach(self)

    # ---- set-up -------------------------------------------------------------------------
    def initialize(self, params, optimizers, scene_scale=1.0):
        for name in ("means", "scales", "quats", "opacities"):
            if name not in params or name not in optimizers:
                raise ValueError(f"mvgs-density requires the {name} field")
        self.extent = float(scene_scale)
        self.lr0 = optimizers["means"].param_groups[0]["lr"]  # already x scene scale
        self._reset_stats(params)

    def set_training_cameras(self, cameras):
        """Runtime hook: reload the training images (the runtime does not expose them)."""
        steps = _argv_value("--steps")
        if sys.argv and Path(sys.argv[0]).name == "ss-train":
            steps = int(steps) if steps is not None else 30000  # ss-train default
            if steps != G.total_steps(self.stages):
                raise ValueError(
                    f"MVGS schedule has {G.total_steps(self.stages)} steps but ss-train runs "
                    f"{steps}; pass --steps {G.total_steps(self.stages)}"
                )
        root = self.c["data"] or _argv_value("--data")
        if root is None:
            raise RuntimeError(
                "mvgs-density needs the training images for its multi-view batches: set density "
                "config 'data' or run through ss-train --data"
            )
        downscale = int(self.c["downscale"] or _argv_value("--downscale") or 1)
        from splatstudio.data import load_dataset

        t0 = time.perf_counter()
        dataset = load_dataset(Path(root), downscale)
        names = dataset["splits"]["train"]
        if len(names) != len(cameras):
            raise RuntimeError("Reloaded training split does not match the runtime's cameras")
        device = cameras[0]["K"].device
        images = []
        for name, cam in zip(names, cameras):
            frame = dataset["frames"][name]
            same = (
                frame["width"] == cam["width"]
                and frame["height"] == cam["height"]
                and torch.allclose(frame["K"].to(device), cam["K"])
                and torch.allclose(frame["viewmat"].to(device), cam["viewmat"])
            )
            if not same:
                raise RuntimeError(f"Reloaded training view {name} differs from the runtime camera")
            images.append(frame["image"])
        del dataset
        self.set_training_frames(cameras, images)
        self.load_seconds = time.perf_counter() - t0

    def set_training_frames(self, cameras, images):
        """Activate multi-view batches. images: HWC float in [0, 1] (any device)."""
        if len(cameras) != len(images) or not cameras:
            raise ValueError("cameras and images must be non-empty and aligned")
        device = cameras[0]["K"].device
        self.cameras = [dict(c) for c in cameras]
        self.native = [(im.to(device) * 255).round().clamp(0, 255).to(torch.uint8) for im in images]
        self.cache_stage = None
        self.stack = []
        self.batch = self._make_batch(1)

    @property
    def active(self):
        return self.native is not None

    # ---- batches ------------------------------------------------------------------------
    def stage_cache(self, k):
        if self.cache_stage != k:
            _, flag, _ = self.stages[k]
            cams, ims = [], []
            for cam, im in zip(self.cameras, self.native):
                w, h = G.stage_size(cam["width"], cam["height"], flag, self.c["full_width_ratio"])
                cams.append(G.scaled_camera(cam, w, h))
                if (w, h) == (cam["width"], cam["height"]):
                    ims.append(im)
                else:
                    small = G.resize_image(im.float() / 255, w, h)
                    ims.append((small * 255).round().to(torch.uint8))
            self.cache, self.cache_stage = (cams, ims), k
        return self.cache

    def sample(self, m):
        """Official: `if not stack: refill; stack.pop(randint(0, len - 1))`, m times."""
        out = []
        for _ in range(m):
            if not self.stack:
                self.stack = list(range(len(self.cameras)))
            out.append(self.stack.pop(self.rng.randint(0, len(self.stack) - 1)))
        return out

    def _make_batch(self, step):
        k, it = G.stage_at(self.stages, step)
        m = self.stages[k][0]
        cams, ims = self.stage_cache(k)
        views = [
            {"index": i, "camera": cams[i], "target": ims[i].float() / 255} for i in self.sample(m)
        ]
        # Views whose rasterizer outputs feed the statistics (the renderer checkpoints the others).
        for j, v in enumerate(views):
            v["stats"] = self.c["stats_views"] == "all" or j == len(views) - 1
        return {"step": step, "stage": k, "iteration": it, "views": views}

    def current_batch(self):
        return self.batch if self.active else None

    # ---- statistics ---------------------------------------------------------------------
    def _reset_stats(self, params):
        n = len(params["means"])
        device = params["means"].device
        self.grad_accum = torch.zeros(n, device=device)
        self.count = torch.zeros(n, device=device)

    def _stats_views(self, outputs):
        views = outputs.get("mv_views")
        if not views:
            return [outputs["info"]]
        if self.c["stats_views"] == "last":
            return [views[-1]["info"]]
        return [v["info"] for v in views]

    def before_backward(self, params, optimizers, step, outputs):
        if self.active and outputs.get("mv_step") != step:
            raise RuntimeError("Rendered multi-view batch does not belong to this step")
        for info in self._stats_views(outputs):
            info["means2d"].retain_grad()

    def _accumulate(self, info):
        grad = info["means2d"].grad
        if grad is None:
            raise RuntimeError("mvgs-density: no screen-space gradient for the statistics view")
        grad = grad[0].clone()
        # 3DGS accumulates NDC-space gradients: d/d(ndc) = d/d(pixel) * size / 2.
        grad[:, 0] *= info["width"] / 2.0
        grad[:, 1] *= info["height"] / 2.0
        radii = info["radii"][0]
        visible = (radii > 0).all(dim=-1) if radii.dim() == 2 else radii > 0
        self.grad_accum[visible] += torch.linalg.vector_norm(grad[visible], dim=-1)
        self.count[visible] += 1

    # ---- schedule helpers ---------------------------------------------------------------
    def is_densify(self, it):
        c = self.c
        return c["densify_from"] < it < c["densify_until"] and it % c["densify_every"] == 0

    def is_reset(self, it):
        c = self.c
        return it < c["densify_until"] and (it % c["reset_every"] == 0 or it == c["white_reset_at"])

    def means_lr(self, it):
        c = self.c
        t = min(max(it / c["lr_max_steps"], 0.0), 1.0)
        return self.lr0 * c["lr_final_factor"] ** t

    # ---- MVGS densification -------------------------------------------------------------
    def mv_inputs(self, outputs):
        """(cameras, boxes, losses) of the step's views for CRD; cameras for MVAD."""
        views = outputs.get("mv_views") or []
        h_box, w_box = (int(v) for v in self.c["window"])
        cams, boxes, losses = [], [], []
        for v in views:
            l1 = (v["target"] - v["rgb"].detach()).abs()
            loss, box = G.max_loss_window(l1, h_box, w_box)
            cams.append(v["camera"])
            boxes.append(box)
            losses.append(loss)
        return cams, boxes, losses

    @torch.no_grad()
    def densify(self, params, optimizers, it, outputs):
        c = self.c
        grads = self.grad_accum / self.count
        grads[grads.isnan()] = 0.0
        cams, boxes, losses = self.mv_inputs(outputs)
        beta = float(c["grow_grad"])
        if c["mvad"] and len(cams) >= 2:
            centers = torch.stack([G.camera_center(cam["viewmat"]) for cam in cams])
            beta = G.mvad_threshold(centers, beta, c["tau"])
        cuboids = torch.zeros(0, 6)
        if c["crd"] and len(cams) >= 2:
            cuboids = G.cross_ray_boxes(cams, boxes, losses, c["box_index"])
        inbox = G.in_boxes(params["means"], cuboids)
        smax = params["scales"].exp().max(dim=-1).values
        small = smax <= c["percent_dense"] * self.extent
        high = grads >= beta
        clone = (high & small) | inbox
        split = high & ~small
        n_clone = int(clone.sum())
        if n_clone:
            ops.duplicate(params, optimizers, {}, clone)
        split = torch.cat([split, torch.zeros(n_clone, dtype=torch.bool, device=split.device)])
        n_split = int(split.sum())
        if n_split:
            ops.split(params, optimizers, {}, split)
        prune = params["opacities"].sigmoid() < c["min_opacity"]
        if it > c["big_after"]:
            prune |= params["scales"].exp().max(dim=-1).values > c["big_world"] * self.extent
        n_prune = int(prune.sum())
        if n_prune:
            ops.remove(params, optimizers, {}, prune)
        self._reset_stats(params)
        self.history.append(
            {
                "iteration": int(it),
                "threshold": beta,
                "cuboids": int(len(cuboids)),
                "inbox": int(inbox.sum()),
                "cloned": n_clone,
                "split": n_split,
                "pruned": n_prune,
                "gaussians": int(len(params["means"])),
            }
        )

    @torch.no_grad()
    def after_backward(self, params, optimizers, step, outputs):
        c = self.c
        k, it = G.stage_at(self.stages, step)
        if it < c["densify_until"]:
            for info in self._stats_views(outputs):
                self._accumulate(info)
            if self.is_densify(it):
                self.densify(params, optimizers, it, outputs)
        if self.is_reset(it):
            ops.reset_opa(params, optimizers, {}, value=c["reset_value"])
        if step < G.total_steps(self.stages):
            _, next_it = G.stage_at(self.stages, step + 1)
            optimizers["means"].param_groups[0]["lr"] = self.means_lr(next_it)
            if self.active:
                self.batch = self._make_batch(step + 1)
        else:
            self.batch = None
        self.state["stage"] = k
        # The statistics are accumulated: release the rasterizer outputs of the non-last views.
        # The runtime keeps `outputs` alive through the next step's forward, so otherwise M views
        # of per-Gaussian tensors (means2d, radii, conics, ...) would stay on the GPU.
        for v in (outputs.get("mv_views") or [])[:-1]:
            v.pop("info", None)


def create_density(spec):
    return MVGSDensity(spec)
