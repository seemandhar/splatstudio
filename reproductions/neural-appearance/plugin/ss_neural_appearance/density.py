"""Density / optimisation driver = the official FasterGSVDA trainer with ``USE_MCMC: true`` and
``PPISP.USE: true`` (paper_configs/default/<appearance>/<scene>.yaml), on top of the Faster-GS
FusedAdam + Morton ordering from ``ss_faster_gs`` (reproductions/faster-gs).

Step mapping (as in ss_faster_gs.density): an official callback at 0-based iteration i runs before
training step i, i.e. after i optimizer steps = runtime ``after_backward(step=i)``.
  initialize        : MCMC init (opacity 0.5, scale 0.1 x RMS-3NN), FusedAdam swap with the
                      config lrs, Morton (iteration 0), lrs of step 1
  after_backward(s) : Bank step (MLP Adam, PPISP Adam + scheduler)
                      s <= T: MCMC noise with the lr of step s (post_optimizer_step)
                      MCMC relocate/add if 600 <= s <= 24900 and s % 100 == 0
                      Morton if s % 5000 == 0 and s <= 25000
                      lrs of step s+1 (means LRDecayPolicy over T, cosine 7k->37k for base colour
                      and features, MLP warm-up policy; the runtime's own means schedule is
                      overwritten)
                      s == T: training_cleanup (prune opacity < 1/255 or degenerate, Morton)
                      s == T + distill: pack MLP + PPISP into ``na_blob``
Steps T+1 .. T+distill (5000) are the PPISP controller distillation (official ``finalize``): the
renderer rasterises without gradients, so no Gaussian field receives a gradient and Adam skips them.
"""

import math

import torch
from gsplat.strategy.ops import _update_param_with_optimizer
from ss_faster_gs.density import permute_gaussians
from ss_faster_gs.morton import morton_order
from ss_faster_gs.optim import replace_optimizers
from torch import nn

from . import bank as bank_mod
from .appearance import DEFAULTS as APPEARANCE_DEFAULTS
from .appearance import SH_C0
from .mcmc import mcmc_step, noise_delta
from .schedule import cosine_decay, lr_decay

DEFAULTS = {
    "appearance": "neural",
    "cap_max": 1_000_000,  # MAX_PRIMITIVES (scene-specific, paper_configs)
    "densify_from": 600,  # DENSIFICATION_START_ITERATION
    "densify_until": 24900,  # DENSIFICATION_END_ITERATION (MCMC)
    "densify_every": 100,  # DENSIFICATION_INTERVAL
    "min_opacity": 0.005,  # mcmc_densification(min_opacity=0.005)
    "noise_lr": 5e5,  # add_noise(..., 5e5 * lr_means)
    "init_opacity": 0.5,  # initialize_from_point_cloud(use_mcmc=True)
    "init_scale_factor": 0.1,  # distances * 0.1 if use_mcmc
    "morton_every": 5000,  # MORTON_ORDERING_INTERVAL
    "morton_until": 25000,  # MORTON_ORDERING_END_ITERATION
    "final_prune_opacity": 1.0 / 255.0,  # MIN_OPACITY_AFTER_TRAINING
    "train_steps": 30000,  # NUM_ITERATIONS
    "lr_means_init": 0.00016,  # LEARNING_RATE_MEANS_INIT (x camera extent)
    "lr_means_final": 0.0000016,  # LEARNING_RATE_MEANS_FINAL
    "lr_scales": 0.005,
    "lr_quats": 0.001,
    "lr_opacities": 0.05,  # MCMC
    # base colour lr in colour space (official LEARNING_RATE_BASE_COLORS); applied to sh0 / Y0
    "lr_base": 0.00025,  # neural 0.00025; SH 0.0025 * Y0
    "lr_residual": 0.008,  # neural features 0.008; SH coefficients 0.000125
    "cosine_decay": True,  # APPEARANCE_COSINE_DECAY.USE (neural: true, SH: false)
    "cosine_start": 7000,
    "cosine_end": 37000,
    "lr_mlp": 0.003,  # LEARNING_RATE_MLP_WEIGHTS
    "mlp_delay_steps": 3000,  # Neural.lr_schedulers
    "mlp_delay_mult": 1e-8,
    "mlp_final_factor": 0.1,
    "mlp_beta2": 0.99,
    "ppisp": True,  # MODEL.PPISP.USE
    "ppisp_controller_steps": 5000,  # CONTROLLER_TRAINING_STEPS (distillation after T)
    "fused_adam": True,
    "seed": 1337,  # tcnn default seed for the MLP init
    **APPEARANCE_DEFAULTS,
}
REQUIRES = {"appearance.neural-ppisp"}
BASE_FIELDS = ("means", "scales", "quats", "opacities", "sh0", "shN")
EXTRA_FIELDS = ("na_features", "na_blob")


def append_rows(params, optimizers, new):
    """Optim.adam_utils.extend_param_groups: concatenate rows; new Adam moments are zero."""
    n_new = len(new["means"])

    def param_fn(name, p):
        return nn.Parameter(torch.cat([p.detach(), new[name].to(p.dtype)]),
                            requires_grad=p.requires_grad)

    def optimizer_fn(key, v):
        return torch.cat([v, torch.zeros((n_new, *v.shape[1:]), dtype=v.dtype, device=v.device)])

    _update_param_with_optimizer(param_fn, optimizer_fn, params, optimizers)


def remove_rows(params, optimizers, mask):
    keep = ~mask

    def param_fn(name, p):
        return nn.Parameter(p.detach()[keep], requires_grad=p.requires_grad)

    def optimizer_fn(key, v):
        return v[keep]

    _update_param_with_optimizer(param_fn, optimizer_fn, params, optimizers)


class NeuralAppearanceDensity:
    def __init__(self, spec):
        unknown = set(spec.config) - set(DEFAULTS)
        if unknown:
            raise ValueError(f"Unknown neural-appearance density options: {sorted(unknown)}")
        self.cfg = {**DEFAULTS, **spec.config}
        self.requires = set(REQUIRES)
        self.state = None
        self.extent = 1.0
        self.bank = None
        self.generator = None

    # ------------------------------------------------------------------ schedules
    def lrs(self, iteration):
        """Official update_learning_rate(iteration) (called with iteration + 1 before step)."""
        c = self.cfg
        out = {"means": lr_decay(iteration, c["lr_means_init"] * self.extent,
                                 c["lr_means_final"] * self.extent, c["train_steps"])}
        base, res = c["lr_base"] / SH_C0, c["lr_residual"]  # sh0 = c0 / Y0 (Adam scale-invariant)
        if c["cosine_decay"]:
            base = cosine_decay(iteration, base, c["cosine_start"], c["cosine_end"])
            res = cosine_decay(iteration, res, c["cosine_start"], c["cosine_end"])
        out["sh0"] = base
        out["na_features" if c["appearance"] == "neural" else "shN"] = res
        out["mlp"] = lr_decay(iteration, c["lr_mlp"], c["lr_mlp"] * c["mlp_final_factor"],
                              c["train_steps"], c["mlp_delay_steps"], c["mlp_delay_mult"])
        return out

    def set_lrs(self, optimizers, iteration):
        lrs = self.lrs(iteration)
        for name, lr in lrs.items():
            if name == "mlp":
                if self.bank is not None:
                    self.bank.set_mlp_lr(lr)
            else:
                optimizers[name].param_groups[0]["lr"] = float(lr)
        return lrs

    # ------------------------------------------------------------------ lifecycle
    def initialize(self, params, optimizers, scene_scale=1.0):
        extra = set(params) - set(BASE_FIELDS) - set(EXTRA_FIELDS)
        if extra or not set(EXTRA_FIELDS) <= set(params):
            raise ValueError(f"neural-appearance density expects fields {EXTRA_FIELDS}")
        c = self.cfg
        self.extent = float(scene_scale)
        with torch.no_grad():
            # MCMC initialisation; runtime --recipe 3dgs already set scales = log RMS-3NN
            params["scales"].add_(math.log(c["init_scale_factor"]))
            params["opacities"].fill_(math.log(c["init_opacity"] / (1 - c["init_opacity"])))
        static = {"scales": c["lr_scales"], "quats": c["lr_quats"],
                  "opacities": c["lr_opacities"], "na_blob": 0.0}
        static["shN" if c["appearance"] == "neural" else "na_features"] = 0.0
        if c["fused_adam"]:
            replace_optimizers(params, optimizers, static)
        else:
            for name, lr in static.items():
                optimizers[name].param_groups[0]["lr"] = float(lr)
        self.set_lrs(optimizers, 1)
        self.state = {"events": [], "counters": {"noise_steps": 0, "mcmc_calls": 0}}
        self.morton(params, optimizers, 0)

    def set_training_cameras(self, cameras):
        c = self.cfg
        ppisp_cfg = {"use": c["ppisp"], "controller_training_steps": c["ppisp_controller_steps"],
                     "controller_distillation": True}
        appearance_cfg = {k: c[k] for k in APPEARANCE_DEFAULTS}
        device = cameras[0]["viewmat"].device if cameras else "cuda"
        self.bank = bank_mod.Bank(c["appearance"], appearance_cfg, ppisp_cfg, cameras,
                                  c["train_steps"], mlp_lr=self.lrs(1)["mlp"],
                                  mlp_betas=(0.9, c["mlp_beta2"]), seed=c["seed"], device=device)
        bank_mod.ACTIVE = self.bank

    def total_steps(self):
        return self.cfg["train_steps"] + (self.cfg["ppisp_controller_steps"] if self.cfg["ppisp"]
                                          else 0)

    def before_backward(self, params, optimizers, step, outputs):
        pass

    @torch.no_grad()
    def morton(self, params, optimizers, step):
        permute_gaussians(params, optimizers, morton_order(params["means"].detach()))
        self.state["events"].append({"step": step, "op": "morton"})

    @torch.no_grad()
    def after_backward(self, params, optimizers, step, outputs):
        c = self.cfg
        T = c["train_steps"]
        if self.bank is None:
            raise RuntimeError("set_training_cameras was not called")
        self.bank.optimizer_step(step)
        if step <= T:
            lr_means = self.lrs(step)["means"]  # lr used by the step just taken
            params["means"].data.add_(noise_delta(params["scales"], params["quats"],
                                                  params["opacities"], c["noise_lr"] * lr_means,
                                                  self.generator))
            self.state["counters"]["noise_steps"] += 1
        if (c["densify_from"] <= step <= c["densify_until"]
                and step % c["densify_every"] == 0):
            event = mcmc_step(params, optimizers,
                              lambda rows: append_rows(params, optimizers, rows),
                              int(c["cap_max"]), c["min_opacity"], self.generator)
            self.state["counters"]["mcmc_calls"] += 1
            if step % 1000 == 0:
                self.state["events"].append({"step": step, "op": "mcmc", **event})
        if step <= c["morton_until"] and step % c["morton_every"] == 0:
            self.morton(params, optimizers, step)
        if step < T:
            self.set_lrs(optimizers, step + 1)
        if step == T:
            m = c["final_prune_opacity"]
            prune = params["opacities"] < math.log(m / (1 - m))
            prune |= params["quats"].square().sum(dim=1) < 1e-8
            if prune.any() and not prune.all():
                remove_rows(params, optimizers, prune)
            self.morton(params, optimizers, step)
            self.state["events"].append({"step": step, "op": "cleanup", "pruned": int(prune.sum()),
                                         "gaussians": len(params["means"])})
            for name in optimizers:  # optimisation of the scene ends (official teardown)
                optimizers[name].param_groups[0]["lr"] = 0.0
            self.bank.set_mlp_lr(0.0)
        # keep the packed state current after every step (density ops above move blob rows), so
        # the checkpoint of any --steps restores the MLP and PPISP; mandatory at the final step
        self.export(params, required=step == self.total_steps())

    @torch.no_grad()
    def export(self, params, required=True):
        flat = self.bank.pack()
        blob = params["na_blob"]
        if flat.numel() > blob.numel():
            if required:
                raise RuntimeError(f"na_blob too small ({blob.numel()} < {flat.numel()} values)")
            self.state["blob_values"] = -1
            return
        new = torch.zeros(blob.numel(), device=blob.device)
        new[:flat.numel()] = flat.to(blob.device)
        blob.data.copy_(new.reshape(blob.shape))
        self.state["blob_values"] = int(flat.numel())


def create_density(spec):
    return NeuralAppearanceDensity(spec)
