"""BOGausS optimizer + densification control (arXiv 2504.01844, sections 3-4).

- replaces every per-field Adam by BOGausSAdam (unbiased sparse Adam, Delta-scaled position and
  linear-size updates, minimal size);
- per-Gaussian precision Delta_i from the training cameras (recomputed every `delta_every`);
- per training step, custom-CUDA statistics of the rendered view: effective opacity
  P_i^(c) = sum_p w / S (running max) and SE_{c,i} / sum_j SE_{c,j} (running sum = P_SNR);
- every 500 steps until 15000: effective-opacity pruning (< 0.02, at most 1 %), then
  density-preserving splits of the top-P_SNR Gaussians up to the budget growth target.
No cloning, no gradient threshold, no opacity reset.
"""

import json

import torch

from . import core, ops
from .kernels import pixel_stats
from .optim import BOGausSAdam
from .renderer import TARGET_CAPABILITY

DEFAULTS = {
    # C1 precision (section "Precision matrix derivation")
    "precision_alpha": 2.0,
    "sigma_uv": 1.0,
    "frustum_margin": 1.3,
    "near": 0.2,
    "min_size_factor": 1.0,
    "delta_every": 500,
    # C2 optimizer
    "betas": [0.9, 0.999],
    "eps": 1e-15,
    "lr_means": "auto",
    "lr_scales": "auto",
    "lr_ref_means": 1.6e-4,
    "lr_ref_log_scales": 0.005,
    "sparse": True,
    # C3 state inheritance
    "alpha_t": 1.0,
    "alpha_g": 0.2,
    # C4 pruning
    "prune_threshold": 0.02,
    "prune_fraction": 0.01,
    # C5 / C6 split
    "split_offset": 0.3,
    "densify_from": 500,
    "densify_until": 15000,
    "densify_interval": 500,
    "budget": None,
    "budget_table": core.budget_table(1.0),
    "log": True,
}
MODES = {"means": "position", "scales": "size"}


class BOGausSDensity:
    def __init__(self, spec):
        unknown = set(spec.config) - set(DEFAULTS)
        if unknown:
            raise ValueError(f"Unknown BOGausS density options: {sorted(unknown)}")
        self.cfg = {**DEFAULTS, **spec.config}
        for key in ("lr_means", "lr_scales"):
            v = self.cfg[key]
            if v != "auto" and not isinstance(v, (int, float)):
                raise ValueError(f"{key} must be 'auto' or a number")
        self.requires = {TARGET_CAPABILITY}
        self.state = None
        self.cameras = None
        self.params = None
        self.optimizers = None
        self.final_budget = None
        self.history = []
        self.lr_info = {}

    # ---------------------------------------------------------------- lifecycle
    def initialize(self, params, optimizers, scene_scale=1.0):
        c = self.cfg
        self.extent = float(scene_scale)
        self.start_count = len(params["means"])
        self.params, self.optimizers = params, optimizers
        for name, p in params.items():
            old = optimizers[name]
            optimizers[name] = BOGausSAdam(
                [p], lr=old.param_groups[0]["lr"], mode=MODES.get(name, "plain"),
                betas=tuple(c["betas"]), eps=c["eps"], min_size_factor=c["min_size_factor"],
            )
        n, dev = self.start_count, params["means"].device
        self.state = {
            "delta": torch.full((n,), float("nan"), device=dev),
            "pmax": torch.zeros(n, device=dev),
            "psnr": torch.zeros(n, device=dev),
        }

    def set_training_cameras(self, cameras):
        c = self.cfg
        self.cameras = cameras
        self.update_delta()
        eta_m = eta_s = None
        if c["lr_means"] == "auto" or c["lr_scales"] == "auto":
            eta_m, eta_s = core.lr_calibration(
                self.state["delta"], self.params["scales"].detach().exp(), self.extent,
                c["lr_ref_means"], c["lr_ref_log_scales"],
            )
        eta_m = eta_m if c["lr_means"] == "auto" else float(c["lr_means"])
        eta_s = eta_s if c["lr_scales"] == "auto" else float(c["lr_scales"])
        # the runtime reads the means lr after this hook and decays it by 0.01 over training
        self.optimizers["means"].param_groups[0]["lr"] = eta_m
        self.optimizers["scales"].param_groups[0]["lr"] = eta_s
        self.lr_info = {"eta_means": eta_m, "eta_scales": eta_s,
                        "median_delta": float(self.state["delta"].median()),
                        "extent": self.extent}
        if c["log"]:
            print(json.dumps({"bogauss_lr": self.lr_info}), flush=True)

    @torch.no_grad()
    def update_delta(self):
        c = self.cfg
        delta, _ = core.precision_delta(
            self.params["means"].detach(), self.cameras, c["precision_alpha"], c["sigma_uv"],
            c["frustum_margin"], c["near"],
        )
        self.state["delta"] = delta

    def before_backward(self, params, optimizers, step, outputs):
        # screen-space gradients are not used by BOGausS; retained so the renderer's declared
        # gradient.screen.signed capability stays observable (generic contract probe)
        outputs["info"]["means2d"].retain_grad()
        radii = outputs["info"]["radii"][0]
        vis = (radii > 0).all(dim=-1) if radii.dim() == 2 else radii > 0
        if not self.cfg["sparse"]:
            vis = torch.ones_like(vis)
        for name, opt in optimizers.items():
            if not isinstance(opt, BOGausSAdam):
                raise RuntimeError(f"{name}: optimizer is not BOGausSAdam")
            opt.visible = vis
            opt.delta = self.state["delta"]

    def after_backward(self, params, optimizers, step, outputs):
        c = self.cfg
        if step <= c["densify_until"]:
            self._accumulate(outputs)
            if step > c["densify_from"] and (step - c["densify_from"]) % c["densify_interval"] == 0:
                self.densify(params, optimizers, step)
                if self.cameras is not None:
                    self.update_delta()
                return
        if self.cameras is not None and step % c["delta_every"] == 0:
            self.update_delta()

    # ---------------------------------------------------------------- statistics
    @torch.no_grad()
    def _accumulate(self, outputs):
        target = outputs.get("target_image")
        if target is None:  # generic contract probes render without a training image
            return
        err = (outputs["rgb"].detach() - target).square().sum(dim=-1)
        stats = pixel_stats(outputs["info"], err)
        p = core.effective_opacity(stats["blend"], stats["count"])
        self.state["pmax"] = torch.maximum(self.state["pmax"], p)
        self.state["psnr"] += core.snr_priority_increment(stats["se"])

    # ---------------------------------------------------------------- densification
    @torch.no_grad()
    def densify(self, params, optimizers, step):
        c = self.cfg
        if self.final_budget is None:
            self.final_budget = core.resolve_budget(self.start_count, c["budget"],
                                                    c["budget_table"])
        target = core.count_target(step, self.start_count, self.final_budget,
                                   c["densify_from"], c["densify_until"], c["densify_interval"])
        n0 = len(params["means"])
        prune_idx = core.prune_selection(self.state["pmax"], c["prune_threshold"],
                                         c["prune_fraction"])
        if len(prune_idx):
            mask = torch.zeros(n0, dtype=torch.bool, device=prune_idx.device)
            mask[prune_idx] = True
            ops.remove(params, optimizers, self.state, mask)
        n1 = len(params["means"])
        n_split = max(0, min(target - n1, n1))
        if n_split:
            sel = torch.topk(self.state["psnr"], n_split).indices
            ops.split(params, optimizers, self.state, sel, c["split_offset"], c["alpha_t"],
                      c["alpha_g"], c["min_size_factor"])
        n = len(params["means"])
        dev = params["means"].device
        self.state["pmax"] = torch.zeros(n, device=dev)
        self.state["psnr"] = torch.zeros(n, device=dev)
        row = {"step": step, "target": target, "pruned": n0 - n1, "splits": n_split, "count": n}
        self.history.append(row)
        if c["log"]:
            print(json.dumps({"bogauss_densify": row}), flush=True)
        if dev.type == "cuda":
            torch.cuda.empty_cache()


def create_density(spec):
    return BOGausSDensity(spec)
