"""HDGS density / schedule component.

Stage 1 (steps 1..stage1_steps) = official 2DGS densification (scene/gaussian_model.py,
train.py with run_360.py arguments) and the 3DGS exponential position LR (30000 steps).
At step ``stage1_steps``: Fisher pruning of ``prune_ratio`` (paper §3.3/§4.1), texture init
(texture_manager.init_texture) and the stage-2 freeze (train_texture.py): xyz, f_dc, scaling,
rotation and opacity get lr 0; f_rest steps only after 1000 stage-2 iterations; the texture
has its own Adam(lr 8e-3) with StepLR(8000, 0.1), stepped here after the runtime's step.
"""

import math

import torch
from gsplat.strategy import ops

from . import texture as tex
from .fisher import fisher_diagonal, prune_mask, sensitivity

DEFAULTS = {
    # official arguments + run_360.py
    "densify_grad_threshold": 0.0002,
    "densification_interval": 150,
    "densify_from_iter": 500,
    "densify_until_iter": 20000,
    "opacity_cull": 0.01,
    "opacity_reset_interval": 3000,
    "percent_dense": 0.01,
    "max_screen_size_after": 3000,  # size_threshold = 20 if it > 3000 (world-size prune only)
    "skip_densify_after": 19000,  # train.py: not (it > 19000 and N > 8e6)
    "skip_densify_count": 8e6,
    "position_lr_init": 0.00016,
    "position_lr_final": 0.0000016,
    "position_lr_max_steps": 30000,
    "init_third_log_scale": -30.0,  # runtime keeps 3 scale columns; 2DGS has 2
    # stage switch
    "stage1_steps": 40000,  # run_360.py --iteration 40000; texture branch resumes chkpnt40000
    "prune_ratio": 0.1,  # paper §4.1 "we prune 10% of the surfels after the 2DGS training"
    "fisher_probes": 1,  # Rademacher probes per training view (Hutchinson estimate)
    "fisher_sort_window": 16,
    # stage 2 (train_texture.py, code-texture/run_360.py)
    "tex_res": 200,
    "max_tex": 500,
    "tex_lr": 0.008,
    "tex_eps": 1e-8,  # torch.optim.Adam default (texture_opt)
    "tex_step_size": 8000,
    "tex_gamma": 0.1,
    "shn_lr": 0.0025 / 20,
    "shn_delay": 1000,  # `if run_iter > 1000: gaussians.optimizer.step()`
}
FROZEN = ("means", "scales", "quats", "opacities", "sh0")


def expon_lr(step, lr_init, lr_final, max_steps):
    """get_expon_lr_func with lr_delay_steps = 0."""
    t = min(max(step / max_steps, 0.0), 1.0)
    return math.exp(math.log(lr_init) * (1 - t) + math.log(lr_final) * t)


class Density:
    requires = {"gradient.hdgs"}

    def __init__(self, spec):
        unknown = set(spec.config) - set(DEFAULTS)
        if unknown:
            raise ValueError(f"Unknown HDGS density options: {sorted(unknown)}")
        self.cfg = {**DEFAULTS, **spec.config}
        self.state = {}
        self.extent = 1.0
        self.cameras = None
        self.accum = None
        self.denom = None
        self.events = []

    # ------------------------------------------------------------------ setup
    def initialize(self, params, optimizers, scene_scale=1.0):
        tex.reset_store()
        self.extent = float(scene_scale)
        n = len(params["means"])
        dev = params["means"].device
        with torch.no_grad():
            # create_from_pcd: rots = torch.rand(N, 4); 2 scale columns (3rd pinned, unused)
            params["quats"].copy_(torch.rand(n, 4, device=dev))
            params["scales"][:, 2] = float(self.cfg["init_third_log_scale"])
            params[tex.FLAG].zero_()
        self._zero_stats(n, dev)
        self._set_means_lr(optimizers, 0)

    def set_training_cameras(self, cameras):
        self.cameras = cameras

    def _zero_stats(self, n, dev):
        self.accum = torch.zeros(n, device=dev)
        self.denom = torch.zeros(n, device=dev)

    def _set_means_lr(self, optimizers, step):
        c = self.cfg
        lr = expon_lr(step, c["position_lr_init"] * self.extent,
                      c["position_lr_final"] * self.extent, c["position_lr_max_steps"])
        optimizers["means"].param_groups[0]["lr"] = lr

    # ------------------------------------------------------------------ hooks
    def before_backward(self, params, optimizers, step, outputs):
        return None  # means2D.retain_grad() is done by the renderer

    def after_backward(self, params, optimizers, step, outputs):
        c = self.cfg
        s1 = int(c["stage1_steps"])
        if step <= s1:
            if step < c["densify_until_iter"]:
                self._accumulate(outputs)
                n = len(params["means"])
                if (step > c["densify_from_iter"] and step % c["densification_interval"] == 0
                        and not (step > c["skip_densify_after"] and n > c["skip_densify_count"])):
                    self._densify_and_prune(params, optimizers, step)
                if step % c["opacity_reset_interval"] == 0:
                    ops.reset_opa(params, optimizers, {}, value=0.01)
                    self.events.append(("reset_opacity", step))
            # update_learning_rate(iteration) is called before each render
            self._set_means_lr(optimizers, step + 1)
        if step == s1:
            self._switch_to_texture(params, optimizers)
        if step >= s1:
            self._stage2_lrs(optimizers, step + 1 - s1)
        if step > s1:
            self._texture_step(step - s1)

    # ------------------------------------------------------------------ stage 1
    @torch.no_grad()
    def _accumulate(self, outputs):
        info = outputs["info"]
        g = info["means2d"].grad
        if g is None:
            return
        vis = info["radii"] > 0
        # add_densification_stats: norm of the full (3-vector) viewspace gradient
        self.accum[vis] += torch.linalg.vector_norm(g[vis], dim=-1)
        self.denom[vis] += 1

    @torch.no_grad()
    def _densify_and_prune(self, params, optimizers, step):
        c = self.cfg
        grads = self.accum / self.denom
        grads[grads.isnan()] = 0.0
        smax = params["scales"][:, :2].exp().max(dim=1).values
        big = smax > c["percent_dense"] * self.extent
        n0 = len(grads)
        clone = (grads >= c["densify_grad_threshold"]) & ~big
        split = (grads >= c["densify_grad_threshold"]) & big
        state = {}
        if n0 < 1.8e7:
            ops.duplicate(params, optimizers, state, clone)  # appended at the end
            pad = torch.zeros(int(clone.sum()), dtype=torch.bool, device=split.device)
            # 2DGS split: samples ~ N(0, diag(s_u, s_v, 0)) rotated; scales / (0.8 * 2)
            # (gsplat split with the pinned third scale exp(-30) is the same operation)
            ops.split(params, optimizers, state, torch.cat([split, pad]))
        prune = torch.sigmoid(params["opacities"]) < c["opacity_cull"]
        if step > c["max_screen_size_after"]:
            # max_radii2D was zeroed by densification_postfix -> only the world-size test acts
            prune |= params["scales"][:, :2].exp().max(dim=1).values > 0.1 * self.extent
        if prune.any():
            ops.remove(params, optimizers, state, prune)
        self._zero_stats(len(params["means"]), params["means"].device)
        self.events.append(("densify", step, int(clone.sum()), int(split.sum()),
                            int(prune.sum()), len(params["means"])))

    # ------------------------------------------------------------------ switch
    def _switch_to_texture(self, params, optimizers):
        c = self.cfg
        if self.cameras is None:
            raise RuntimeError("HDGS Fisher pruning needs the training cameras")
        n0 = len(params["means"])
        if c["prune_ratio"] > 0:
            diag = fisher_diagonal(params, self.cameras, int(c["fisher_sort_window"]),
                                   sh_degree=3, probes=int(c["fisher_probes"]))
            mask = prune_mask(sensitivity(diag), float(c["prune_ratio"]))
            with torch.no_grad():
                ops.remove(params, optimizers, {}, mask)
        with torch.no_grad():
            params[tex.FLAG].fill_(1.0)
            index, total = tex.texture_index(params["scales"][:, :2].exp(), c["tex_res"],
                                             int(c["max_tex"]))
            buf = torch.nn.Parameter(tex.initial_texture(params["sh0"], index, total))
        opt = torch.optim.Adam([buf], lr=c["tex_lr"], eps=c["tex_eps"])
        tex.STORE.update(texture=buf, index=index, optimizer=opt, key=tex.store_key(params))
        self.state = {"texture": buf.data, "texture_key": tex.STORE["key"],
                      "tex_res": torch.tensor(float(c["tex_res"]))}
        self.events.append(("fisher_prune_texture", n0, len(params["means"]), total))

    def _stage2_lrs(self, optimizers, t):
        """Learning rates used at stage-2 iteration t (1-based)."""
        for name in FROZEN:
            optimizers[name].param_groups[0]["lr"] = 0.0
        # run_iter = t - 1; gaussian optimiser steps only when run_iter > shn_delay
        optimizers["shN"].param_groups[0]["lr"] = (
            self.cfg["shn_lr"] if t - 1 > self.cfg["shn_delay"] else 0.0)
        opt = tex.STORE["optimizer"]
        if opt is not None:
            opt.param_groups[0]["lr"] = tex.tex_lr(t, self.cfg["tex_lr"],
                                                   self.cfg["tex_step_size"],
                                                   self.cfg["tex_gamma"])

    def _texture_step(self, t):
        opt = tex.STORE["optimizer"]
        opt.step()
        opt.zero_grad(set_to_none=True)
        # keep the persisted reference current (Adam updates in place)
        self.state["texture"] = tex.STORE["texture"].data


def create_density(spec):
    return Density(spec)
