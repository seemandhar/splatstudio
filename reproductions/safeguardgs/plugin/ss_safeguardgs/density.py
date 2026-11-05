"""SafeguardGS density: vanilla 3DGS densification plus one pixel-level rank pruning event.

Official `scene/gaussian_model_safeguard_gs.py` (https://github.com/ASU-ESIC-FAN-Lab/SafeguardGS,
commit 539ca22e) with the paper's sweep settings (`benchmark/benchmark.yaml`: prune_iterations
[20000], safeguard_gs_purne_topk 1, safeguard_gs_score_function 0x24):

* every step: the unchanged SplatStudio 3DGS density (gsplat DefaultStrategy + opacity-reset
  fix), and the target image of the rendered training view is cached (uint8, lossless for 8-bit
  sources) for the colour-similarity term;
* step in prune_steps: render every training view once with the top-k kernel against its
  ground truth, OR the per-view keep masks, remove all other Gaussians
  (`topk_gaussians` + `prune_points`; survivors keep their Adam moments).
"""

import json
import time
from dataclasses import replace

import torch
from gsplat.strategy import ops
from splatstudio.backends.gsplat import Density

from . import core, kernels
from .renderer import TARGET_CAPABILITY

DEFAULTS = {
    "base": {"absgrad": False, "grow_grad2d": 0.0002},
    "prune_steps": [20000],  # paper Sec. 4.1 / supp. Sec. 7, benchmark.yaml
    "top_k": 1,  # paper Sec. 3.4 / 4.3
    "score_function": core.SG_36,  # 0x24 = Score_SG_36
    "selection": "tile",  # official kernel semantics; "pixel" = paper-text per-ray top-k
    "p_dist_coef": 1.0,  # safeguard_gs_p_dist_activation_coef
    "c_dist_coef": 1.0,  # safeguard_gs_c_dist_activation_coef
}


def runtime_sh_degree(step):
    """SH degree the native runtime renders at `step` (+1 per 1000 steps, max 3)."""
    return min((int(step) - 1) // 1000, 3)


class SafeguardDensity:
    def __init__(self, spec):
        unknown = set(spec.config) - set(DEFAULTS)
        if unknown:
            raise ValueError(f"Unknown sg-density options: {sorted(unknown)}")
        self.cfg = {**DEFAULTS, **spec.config}
        if self.cfg["selection"] not in kernels.SELECTIONS:
            raise ValueError(f"selection must be one of {kernels.SELECTIONS}")
        if not core.valid_func_id(self.cfg["score_function"]):
            raise ValueError("invalid score_function bit mask")
        if int(self.cfg["top_k"]) < 1 or (self.cfg["selection"] == "pixel" and self.cfg["top_k"] > 16):
            raise ValueError("top_k must be >= 1 (<= 16 for selection 'pixel')")
        self.base = Density(replace(spec, config=dict(self.cfg["base"])))
        self.requires = set(self.base.requires) | {TARGET_CAPABILITY}
        self.cameras = None
        self.images = None
        self.history = []

    @property
    def state(self):
        return {"strategy": self.base.state, "safeguard_prunes": self.history}

    def set_training_cameras(self, cameras):
        self.cameras = list(cameras)
        self.viewmats = torch.stack([c["viewmat"] for c in self.cameras]).flatten(1)
        self.images = [None] * len(self.cameras)

    def initialize(self, params, optimizers, scene_scale=1.0):
        self.base.initialize(params, optimizers, scene_scale)

    def before_backward(self, params, optimizers, step, outputs):
        self.base.before_backward(params, optimizers, step, outputs)

    def after_backward(self, params, optimizers, step, outputs):
        self.base.after_backward(params, optimizers, step, outputs)
        self._cache_target(outputs)
        if step in self.cfg["prune_steps"]:
            self.prune(params, optimizers, step)

    def _cache_target(self, outputs):
        if self.cameras is None or outputs.get("target_image") is None:
            return
        d = (self.viewmats - outputs["viewmat"].flatten()[None]).abs().sum(dim=1)
        k = int(d.argmin())
        if float(d[k]) > 1e-5:
            raise RuntimeError("Rendered frame is not one of the training cameras")
        if self.images[k] is None:
            self.images[k] = (outputs["target_image"].clamp(0, 1) * 255).round().to(torch.uint8)

    @torch.no_grad()
    def keep_mask(self, params, step):
        """Union over training views of the per-view top-k keep masks (official topk_gaussians)."""
        if not self.cameras:
            raise RuntimeError("sg-density needs set_training_cameras() before pruning")
        missing = [k for k, img in enumerate(self.images) if img is None]
        if missing:
            raise RuntimeError(
                f"{len(missing)} training views were never rendered before step {step}; "
                "their ground-truth images are unavailable for the colour score"
            )
        c = self.cfg
        degree = runtime_sh_degree(step)
        keep = torch.zeros(len(params["means"]), dtype=torch.bool, device=params["means"].device)
        for cam, img in zip(self.cameras, self.images):
            meta, colors = kernels.project(params, cam, degree)
            keep |= kernels.view_keep(
                meta, colors, img.float() / 255.0, c["score_function"], c["top_k"],
                c["selection"], c["p_dist_coef"], c["c_dist_coef"],
            )
        return keep

    @torch.no_grad()
    def prune(self, params, optimizers, step):
        start = time.perf_counter()
        keep = self.keep_mask(params, step)
        before = len(params["means"])
        ops.remove(params, optimizers, self.base.state, ~keep)
        torch.cuda.empty_cache()
        record = {
            "safeguard_prune_step": int(step),
            "before": before,
            "after": len(params["means"]),
            "pruned_fraction": round(1 - len(params["means"]) / max(before, 1), 4),
            "views": len(self.cameras),
            "seconds": round(time.perf_counter() - start, 2),
        }
        self.history.append(record)
        print(json.dumps(record), flush=True)


def create_density(spec):
    return SafeguardDensity(spec)
