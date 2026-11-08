"""AuGS density controller: the base 3DGS run (splatstudio's gsplat DefaultStrategy wrapper, with
the 3DGS opacity reset) for `base_steps`, then the post-hoc enhancement at step `base_steps`:
render the training views (RGB + T-threshold depth), error-driven 2D Gaussian training per view,
inverse splatting into augmented 3D Gaussians with view-dependent opacity lobes, and the joint
optimisation set-up of official refine.py (sparse Adam without bias correction, original
Gaussians: opacity only).

ss-train runs both phases in one job: --steps = base_steps + refine_iters_per_view * |train|.
The training images are reloaded from ss-train's --data (the runtime exposes cameras only).
"""

import json
import math
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

from .fit2d import fit_view, image_loss
from .inverse import inverse_splat
from .optim import SparseRowAdam
from .renderer import AugRenderer, median_depth

DEFAULTS = {
    # base 3DGS (identical to the built-in `gsplat` method's density config)
    "absgrad": False,
    "grow_grad2d": 0.0002,
    "opacity_reset": True,
    # schedule
    "base_steps": 30000,
    "refine_iters_per_view": 30,  # §4.1: joint iterations = 30 x |train views|
    "strict_steps": True,
    # 2D stage (supp. §A, train_2d.py)
    "ratio": 0.1,
    "depth_t_threshold": 0.7,  # official gsplat rasterize_to_pixels_fwd.cu:173 (paper: 0.5)
    "xy_lr": 0.001,
    "color_lr": 0.01,
    "opacity_lr": 0.02,
    "scale_lr": 0.001,
    "rotation_lr": 0.02,
    "add_fraction": 0.2,
    "coarse_steps": 200,
    "finetune_steps": 1000,
    "init_opacity": 0.02,
    "init_scale": 0.005,
    # lobe initialisation (supp. §A, refine.py:51)
    "t_scale": 7.0,
    "t_min": 0.01,
    "t_max": 1.0,
    "beta_init": 0.0,
    # joint optimisation learning rates (supp. §A, benchmark.py)
    "lr_means": 1.6e-5,  # x scene extent, constant
    "lr_sh0": 0.001,
    "lr_shN": 0.001 / 20,
    "lr_opacities": 0.02,
    "lr_scales": 0.002,
    "lr_quats": 0.0005,
    "lr_lobe_dir": 0.001,
    "lr_lobe_t": 0.0002,
    "lr_lobe_beta": 0.002,
    "seed": 0,
    "measure_base": True,  # write <output>/base_test.json (base model, same run)
    "data": None,
    "downscale": None,
}

BASE_KEYS = ("absgrad", "grow_grad2d", "opacity_reset")
FIELD_NAMES = ("lobe_dir", "lobe_t", "lobe_beta", "aug_flag")


def _argv_value(flag):
    argv = sys.argv
    for i, arg in enumerate(argv):
        if arg == flag and i + 1 < len(argv):
            return argv[i + 1]
        if arg.startswith(flag + "="):
            return arg.split("=", 1)[1]
    return None


def lobe_t_init(theta, cfg):
    """T = clip(c theta_bar / pi, 0.01, 1.0), c = 7 (supp. §A; refine.py:51)."""
    return torch.clip(theta * cfg["t_scale"] / math.pi, min=cfg["t_min"], max=cfg["t_max"])


def allocate(losses, total):
    """train_2d_kernels: kernels per view = ceil(loss_i / sum(loss) * total)."""
    w = torch.as_tensor(losses, dtype=torch.float64).clamp(min=0)
    w = w / w.sum()
    return torch.ceil(w * total).int().tolist()


class AugsDensity:
    def __init__(self, spec):
        unknown = set(spec.config) - set(DEFAULTS)
        if unknown:
            raise ValueError(f"Unknown augs-density options: {sorted(unknown)}")
        self.c = {**DEFAULTS, **spec.config}
        from splatstudio.backends.gsplat import Density

        self.base = Density(SimpleNamespace(config={k: self.c[k] for k in BASE_KEYS}))
        self.requires = set(self.base.requires)
        self.cameras = None
        self.phase = "base"
        self.n_base = None
        self.stats = {}
        self.state = {"phase": self.phase, "stats": self.stats}

    def initialize(self, params, optimizers, scene_scale=1.0):
        for name in FIELD_NAMES:
            if name not in params:
                raise ValueError(f"augs-density requires the {name} field")
        self.base.initialize(params, optimizers, scene_scale)
        self.state["base_state"] = self.base.state
        self.extent = float(scene_scale)
        self.lr0 = optimizers["means"].param_groups[0]["lr"]  # already x scene scale

    def set_training_cameras(self, cameras):
        self.cameras = [dict(c) for c in cameras]
        if sys.argv and Path(sys.argv[0]).name == "ss-train" and self.c["strict_steps"]:
            steps = int(_argv_value("--steps") or 30000)
            need = self.c["base_steps"] + self.c["refine_iters_per_view"] * len(cameras)
            if steps != need:
                raise ValueError(
                    f"AuGS runs {self.c['base_steps']} base steps + "
                    f"{self.c['refine_iters_per_view']} x {len(cameras)} joint steps: pass "
                    f"--steps {need} (got {steps})")

    # ---- training hooks -----------------------------------------------------------------
    def before_backward(self, params, optimizers, step, outputs):
        if self.phase == "base":
            self.base.before_backward(params, optimizers, step, outputs)

    def after_backward(self, params, optimizers, step, outputs):
        if self.phase == "base":
            self.base.after_backward(params, optimizers, step, outputs)
            # 3DGS position schedule over the base run (the runtime decays over --steps)
            frac = min(step, self.c["base_steps"]) / self.c["base_steps"]
            optimizers["means"].param_groups[0]["lr"] = self.lr0 * 0.01**frac
            if step == self.c["base_steps"]:
                self.augment(params, optimizers)
        else:
            optimizers["means"].param_groups[0]["lr"] = self.c["lr_means"] * self.extent

    # ---- enhancement --------------------------------------------------------------------
    def _load_images(self, device):
        root = self.c["data"] or _argv_value("--data")
        if root is None:
            raise RuntimeError("augs-density needs the training images: set config 'data' or run "
                               "through ss-train --data")
        downscale = int(self.c["downscale"] or _argv_value("--downscale") or 1)
        from splatstudio.data import load_dataset

        dataset = load_dataset(Path(root), downscale)
        names = dataset["splits"]["train"]
        if self.cameras is None or len(names) != len(self.cameras):
            raise RuntimeError("Reloaded training split does not match the runtime's cameras")
        for name, cam in zip(names, self.cameras):
            frame = dataset["frames"][name]
            if not (frame["width"] == cam["width"] and frame["height"] == cam["height"]
                    and torch.allclose(frame["viewmat"].to(device), cam["viewmat"])
                    and torch.allclose(frame["K"].to(device), cam["K"])):
                raise RuntimeError(f"Reloaded training view {name} differs from the runtime camera")
        return dataset, [dataset["frames"][n]["image"] for n in names]

    @torch.no_grad()
    def render_views(self, params, images, device):
        renderer = AugRenderer(SimpleNamespace(config={}))
        renders, depths, losses = [], [], []
        for cam, img in zip(self.cameras, images):
            rgb = renderer.render(params, cam, 3)["rgb"].permute(2, 0, 1).contiguous()
            gt = img.to(device).permute(2, 0, 1).contiguous()
            losses.append(float(image_loss(rgb, gt).clamp(min=0)))
            depths.append(median_depth(params, cam, self.c["depth_t_threshold"]).cpu())
            renders.append(rgb.cpu())
        return renders, depths, losses

    def augment(self, params, optimizers):
        device = params["means"].device
        t0 = time.perf_counter()
        n_base = len(params["means"])
        dataset, images = self._load_images(device)
        if self.c["measure_base"]:
            out = _argv_value("--output")
            if out is not None:
                from splatstudio.runtime import measure, write_json

                renderer = AugRenderer(SimpleNamespace(config={}))
                metrics, _ = measure(renderer, params, dataset, "test", 3, device)
                write_json(Path(out) / "base_test.json",
                           {"step": self.c["base_steps"], "gaussians": n_base, "split": "test",
                            "metrics": metrics, "note": "base model before AuGS enhancement; "
                            "measurement only, never used for selection"})
                self.stats["base_test"] = metrics
                print(json.dumps({"augs_base_test": metrics}), flush=True)
        del dataset
        renders, depths, losses = self.render_views(params, images, device)
        counts = allocate(losses, n_base * self.c["ratio"])
        torch.cuda.empty_cache()
        t1 = time.perf_counter()
        gen = torch.Generator(device=device).manual_seed(int(self.c["seed"]))
        new = {k: [] for k in ("means", "sh0", "opacities", "lobe_dir", "theta", "scales", "quats")}
        totals = {"kernels": 0, "skipped": 0, "cluster_retries": 0, "scale_fit_failures": 0}
        fit_seconds = proj_seconds = 0.0
        for i, cam in enumerate(self.cameras):
            ta = time.perf_counter()
            bg = renders[i].to(device)
            gt = images[i].to(device).permute(2, 0, 1).contiguous()
            dm = depths[i].to(device)
            g, info = fit_view(bg, gt, dm, counts[i], self.c, gen)
            tb = time.perf_counter()
            if g.size():
                p = {k: v.detach() for k, v in g.p.items()}
                res = inverse_splat(p["xy"], p["scale"], p["rotation"], p["opacity"], p["color"],
                                    dm, i, self.cameras)
                for k, lst in new.items():
                    lst.append(res[k])
                for k in totals:
                    totals[k] += res["stats"][k]
            tc = time.perf_counter()
            fit_seconds += tb - ta
            proj_seconds += tc - tb
            if i % 10 == 0 or i == len(self.cameras) - 1:
                print(json.dumps({"augs_view": i, "target": counts[i], **info,
                                  "fit_s": round(tb - ta, 2), "proj_s": round(tc - tb, 2)}),
                      flush=True)
            del g, bg, gt, dm
        cat = {k: torch.cat(v) for k, v in new.items()}
        n_aug = len(cat["means"])
        if n_aug == 0:
            raise RuntimeError("AuGS produced no augmented Gaussians")
        rows = {
            "means": cat["means"],
            "scales": cat["scales"],
            "quats": cat["quats"],
            "opacities": cat["opacities"],
            "sh0": cat["sh0"],
            "shN": torch.zeros((n_aug,) + tuple(params["shN"].shape[1:]), device=device),
            "lobe_dir": cat["lobe_dir"],
            "lobe_t": lobe_t_init(cat["theta"], self.c),
            "lobe_beta": torch.full((n_aug,), float(self.c["beta_init"]), device=device),
            "aug_flag": torch.ones(n_aug, device=device),
        }
        with torch.no_grad():
            for name, extra in rows.items():
                old = params[name].detach()
                params[name] = nn.Parameter(torch.cat((old, extra.to(old.dtype))).contiguous(),
                                            requires_grad=True)
        lrs = {"means": self.c["lr_means"] * self.extent, "sh0": self.c["lr_sh0"],
               "shN": self.c["lr_shN"], "opacities": self.c["lr_opacities"],
               "scales": self.c["lr_scales"], "quats": self.c["lr_quats"]}
        for name, lr in lrs.items():
            optimizers[name] = SparseRowAdam(params[name], lr, n_base,
                                             include_used_base=(name == "opacities"))
        for name in ("lobe_dir", "lobe_t", "lobe_beta"):
            optimizers[name] = torch.optim.Adam([params[name]], lr=self.c[f"lr_{name}"], eps=1e-15)
        optimizers["aug_flag"] = torch.optim.Adam([params["aug_flag"]], lr=1e-12)
        for name, p in params.items():  # requires_grad pitfall guard
            if not (p.is_leaf and p.requires_grad):
                raise RuntimeError(f"{name} is not a trainable leaf after augmentation")
        self.n_base = n_base
        self.phase = "refine"
        self.stats.update(totals)
        self.stats.update({
            "n_base": n_base, "n_aug": n_aug, "views": len(self.cameras),
            "render_seconds": t1 - t0, "fit2d_seconds": fit_seconds,
            "projection_seconds": proj_seconds, "total_seconds": time.perf_counter() - t0,
            "lobe_t_mean": float(rows["lobe_t"].mean()),
        })
        self.state["phase"] = self.phase
        print(json.dumps({"augs_augmented": self.stats}), flush=True)
        torch.cuda.empty_cache()


def create_density(spec):
    return AugsDensity(spec)
