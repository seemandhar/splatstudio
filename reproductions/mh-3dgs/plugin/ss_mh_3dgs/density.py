"""MH-3DGS density controller: Metropolis-Hastings birth sampler + MCMC relocation.

Port of `combined_mcmc_and_reloc_multiview` and the densification block of train.py of
https://github.com/SNU-VGILab/MH-3DGS (commit 71ef8f9). Replaces 3DGS clone/split/prune and
opacity reset entirely; no SGLD noise and no Gaussian cap (the count emerges from acceptance).
"""

import math
import random
import sys
import time
from pathlib import Path

import torch
from gsplat.strategy.ops import _update_param_with_optimizer, relocate

from . import core
from .renderer import render


def _argv_value(flag):
    argv = sys.argv
    for i, arg in enumerate(argv):
        if arg == flag and i + 1 < len(argv):
            return argv[i + 1]
        if arg.startswith(flag + "="):
            return arg.split("=", 1)[1]
    return None


def append_gaussians(params, optimizers, parents, new_means):
    """densification_postfix: append copies of `parents` (raw parameters) with `new_means`;
    Adam moments of the new rows are zero. Parameters stay trainable leaves."""
    n_new = len(parents)

    def param_fn(name, p):
        rows = new_means.to(p.dtype) if name == "means" else p[parents]
        return torch.nn.Parameter(torch.cat([p.detach(), rows.detach()]), requires_grad=p.requires_grad)

    def optimizer_fn(key, v):
        return torch.cat([v, torch.zeros((n_new, *v.shape[1:]), device=v.device, dtype=v.dtype)])

    _update_param_with_optimizer(param_fn, optimizer_fn, params, optimizers)


def check_trainable(params):
    for name, p in params.items():
        if not (isinstance(p, torch.nn.Parameter) and p.requires_grad and p.is_leaf):
            raise RuntimeError(f"{name}: density op left a non-trainable parameter")


class MHDensity:
    requires = set()

    def __init__(self, spec):
        unknown = set(spec.config) - set(core.DEFAULTS)
        if unknown:
            raise ValueError(f"Unknown mh-density options: {sorted(unknown)}")
        self.c = {**core.DEFAULTS, **spec.config}
        if self.c["pixel_lookup"] != "world_xy":
            raise ValueError("pixel_lookup must be 'world_xy' (official code)")
        if self.c["ssim_kind"] not in ("similarity", "dissimilarity"):
            raise ValueError("ssim_kind must be 'similarity' or 'dissimilarity'")
        self.requires = set()
        self.cameras = None
        self.images = None  # CPU (pinned) uint8 or float32 HWC, aligned with cameras
        self.order = None
        self.denom = None
        self.gen = None
        self._binoms_t = None
        self.history = []
        self.state = {"history": self.history}

    # ---- runtime hooks ------------------------------------------------------------------
    def initialize(self, params, optimizers, scene_scale=1.0):
        for name in ("means", "scales", "quats", "opacities", "sh0", "shN"):
            if name not in params or name not in optimizers:
                raise ValueError(f"mh-density requires the {name} field")
        # Official opacity_lr = 0.025 (arguments/__init__.py); runtime default 0.05.
        optimizers["opacities"].param_groups[0]["lr"] = float(self.c["opacity_lr"])
        dev = params["means"].device
        self.denom = torch.zeros(len(params["means"]), device=dev)
        self.gen = torch.Generator(device=dev)
        self.gen.manual_seed(int(self.c["seed"]))

    def set_training_cameras(self, cameras):
        """Runtime hook: reload the training images (not exposed to components) for the
        multi-view error maps (improving-densification / mvgs precedent)."""
        root = self.c["data"] or _argv_value("--data")
        if root is None:
            raise RuntimeError("mh-density needs the training images: set density config "
                               "'data' or run through ss-train --data")
        downscale = int(self.c["downscale"] or _argv_value("--downscale") or 1)
        from splatstudio.data import load_dataset

        t0 = time.perf_counter()
        dataset = load_dataset(Path(root), downscale)
        names = dataset["splits"]["train"]
        if len(names) != len(cameras):
            raise RuntimeError("Reloaded training split does not match the runtime cameras")
        images = []
        for name, cam in zip(names, cameras):
            frame = dataset["frames"][name]
            dev = cam["K"].device
            same = (frame["width"] == cam["width"] and frame["height"] == cam["height"]
                    and torch.allclose(frame["K"].to(dev), cam["K"])
                    and torch.allclose(frame["viewmat"].to(dev), cam["viewmat"]))
            if not same:
                raise RuntimeError(f"Reloaded training view {name} differs from runtime camera")
            images.append(frame["image"])
        del dataset
        self.set_training_frames(cameras, images)
        self.state["image_load_seconds"] = time.perf_counter() - t0

    def set_training_frames(self, cameras, images):
        if len(cameras) != len(images) or not cameras:
            raise ValueError("cameras and images must be non-empty and aligned")
        self.cameras = [dict(c) for c in cameras]
        pin = torch.cuda.is_available()
        self.images = []
        for im in images:
            im = im.float().cpu()
            q = (im * 255).round().clamp(0, 255).to(torch.uint8)
            store = q if torch.equal(q.float() / 255, im) else im  # exact 8-bit storage only
            self.images.append(store.pin_memory() if pin else store)
        order = list(range(len(cameras)))
        if self.c["shuffle_cameras"]:
            random.Random(int(self.c["seed"])).shuffle(order)  # 3DGS Scene(shuffle=True)
        self.order = order

    def before_backward(self, params, optimizers, step, outputs):
        pass

    def after_backward(self, params, optimizers, step, outputs):
        event = core.schedule(step, self.c)
        if event is None:
            return
        if self.cameras is None:
            raise RuntimeError("mh-density has no training views (set_training_cameras not called)")
        # add_densification_stats on densification iterations: denom += 1 where visible.
        radii = outputs["info"]["radii"][0]
        visible = (radii > 0).all(dim=-1) if radii.dim() == 2 else radii > 0
        if len(self.denom) != len(params["means"]):
            raise RuntimeError("mh-density statistics are misaligned with the population")
        self.denom[visible] += 1
        with torch.no_grad():
            self._step(params, optimizers, step, event)
        check_trainable(params)

    # ---- algorithm ----------------------------------------------------------------------
    def _image(self, i, device):
        im = self.images[i].to(device, non_blocking=True)
        return im.float() / 255 if im.dtype == torch.uint8 else im

    def error_maps(self, params, views, sh_degree):
        """accumulate_multiview_errors: mean SSIM map and mean |I - I_hat| map, CHW."""
        dev = params["means"].device
        bg = torch.tensor(self.c["background"], dtype=torch.float32, device=dev)
        ssim_sum = l1_sum = None
        for v in views:
            cam = self.cameras[v]
            # Official render() clamps to [0,1] (gaussian_renderer/__init__.py:119).
            rgb = render(params, cam, sh_degree, bg)["rgb"].clamp(0, 1).permute(2, 0, 1)
            gt = self._image(v, dev).permute(2, 0, 1)
            s = core.ssim_map(rgb, gt)
            if self.c["ssim_kind"] == "dissimilarity":
                s = 1 - s
            e = (rgb - gt).abs()
            ssim_sum = s if ssim_sum is None else ssim_sum + s
            l1_sum = e if l1_sum is None else l1_sum + e
        return ssim_sum / len(views), l1_sum / len(views)

    def _step(self, params, optimizers, step, ev):
        t0 = time.perf_counter()
        n_views = len(self.cameras)
        k = core.subset_size(n_views, ev["progress"])
        views = [self.order[i] for i in core.round_robin(n_views, step, k)]
        sh_degree = min(step // 1000, 3)  # official oneupSHdegree at it % 1000 == 0
        ssim_agg, l1_agg = self.error_maps(params, views, sh_degree)

        # Step 1: relocate dead Gaussians (identical to gsplat.strategy.ops.relocate).
        dead = params["opacities"].sigmoid() <= self.c["opacity_threshold"]
        n_dead = int(dead.sum())
        if 0 < n_dead < len(dead):
            relocate(params, optimizers, {}, dead, self._binoms(params["means"].device),
                     min_opacity=self.c["opacity_threshold"])

        # Step 2: importance and MH birth proposals.
        means = params["means"]
        ssim_s, l1_s = core.lookup_world_xy(means, ssim_agg, l1_agg)
        imp = core.importance(params["opacities"].sigmoid(), ssim_s, l1_s, self.c)
        scales = params["scales"].exp()
        existing = means.detach()
        parents, new_means, stats = [], [], {}
        for phase in ("coarse", "fine"):
            idx, prop, n_prop, rho = core.propose(
                existing, scales, self.denom, imp, ev[f"{phase}_batch"], ev[f"{phase}_scale"],
                existing, ev["voxel_size"], self.c, self.gen)
            parents.append(idx)
            new_means.append(prop)
            stats[phase] = {"proposed": n_prop, "accepted": int(idx.numel()),
                            "mean_rho": float(rho.mean()) if rho.numel() else 0.0}
        parents = torch.cat(parents)
        new_means = torch.cat(new_means)
        if len(parents) > 0:
            append_gaussians(params, optimizers, parents, new_means)
            # densification_postfix resets the statistics.
            self.denom = torch.zeros(len(params["means"]), device=means.device)
        self.history.append({
            "step": step, "views": k, "dead_relocated": n_dead, "born": int(len(parents)),
            "gaussians": len(params["means"]), **stats,
            "seconds": round(time.perf_counter() - t0, 3)})

    def _binoms(self, device):
        if self._binoms_t is None or self._binoms_t.device != device:
            n_max = 51
            b = torch.zeros((n_max, n_max), device=device)
            for n in range(n_max):
                for kk in range(n + 1):
                    b[n, kk] = float(math.comb(n, kk))
            self._binoms_t = b
        return self._binoms_t


def create_density(spec):
    return MHDensity(spec)
