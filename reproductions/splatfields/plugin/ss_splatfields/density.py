"""SplatFields density components.

`SplatFieldsDensity` (stage 2, the paper's method) owns the SplatFields networks (global
learned state the runtime cannot hold) and their Adam, registered in `network.ACTIVE`. Per step,
after the runtime's per-Gaussian optimizer step (the anchors get no gradient, official
`get_xyz.detach()`):
1. network Adam step + zero_grad, then LR(step) = exp-lerp 8e-4 -> 1.6e-6 over 40k (official
   SplatFieldsModel.update_learning_rate(iteration) after deform.optimizer.step());
2. 3DGS clone/split/prune on the anchors with gsplat's DefaultStrategy: densify every 100 steps
   after 500, last refine at 39900 (official saves before the 40000 densify), screen-space gradients of the *predicted*
   centres (official viewspace_points of the network splats), split sampled from the stored
   base Gaussian (base scale, identity rotation), prune on the stored opacity (constant 0.1 ->
   never below 0.005) and on world size > 0.1 extent after step 3000; no opacity reset
   (official train.py never calls reset_opacity); screen-size pruning is dead code in the
   official 3DGS loop (max_radii2D is zeroed by densification_postfix right before the prune),
   matching gsplat's refine_scale2d_stop_iter = 0;
3. at total_steps: bake the predicted Gaussians into the base fields (means, log |scale|, quats,
   logit opacity, sh0 = (rgb - 0.5)/C0, shN = 0) and mark the network baked, so validation, the
   checkpoint and `ss-eval` measure the plain-3DGS model the paper renders after discarding
   the networks.

`Stage1Density` (`splatfields-3dgs`, the official `--is_static` 3DGS run whose cloud
initialises SplatFields and which is the paper's 3DGS row): the same DefaultStrategy schedule on
ordinary Gaussians, no opacity reset, and the official xyz LR 1.6e-4 * 5 -> 1.6e-6 * 5 over 30k
steps (GaussianModel.training_setup hard-codes spatial_lr_scale = 5), held after 30k.
"""

import json
import sys

import torch
from gsplat import DefaultStrategy

from . import network

C0 = 0.28209479177387814

# refine_stop_iter 40000: official densify_until_iter is 45000, but train.py saves the model
# (saving_iterations) BEFORE the densification block of iteration 40000, so the evaluated set
# is the one after the 39900 refine; gsplat returns for step >= refine_stop_iter.
# reset_every 3000 only gates gsplat's world-size prune (step > reset_every, official
# iteration > opacity_reset_interval); NoResetStrategy never resets opacity.
BASE = {"absgrad": False, "grow_grad2d": 0.0002, "grow_scale3d": 0.01, "prune_scale3d": 0.1,
        "prune_opa": 0.005, "refine_start_iter": 500, "refine_stop_iter": 40000,
        "refine_every": 100, "reset_every": 3000}

DEFAULTS = {
    "total_steps": 40000,
    "lr_init": 0.00016 * 5,   # position_lr_init * spatial_lr_scale (deform_model.py)
    "lr_final": 0.0000016,    # position_lr_final
    "lr_max_steps": 40000,    # deform_lr_max_steps
    "adam_eps": 1e-15,
    "net_seed": 0,
    "net": network.NET_DEFAULTS,
    "base": BASE,
}

STAGE1_DEFAULTS = {
    "total_steps": 40000,
    "means_lr_init": 0.00016 * 5,
    "means_lr_final": 0.0000016 * 5,
    "means_lr_max_steps": 30000,
    "base": BASE,
}


class NoResetStrategy(DefaultStrategy):
    """gsplat DefaultStrategy.step_post_backward without the opacity reset (official SplatFields
    train.py never calls reset_opacity). Does not rely on gsplat 1.5.x's precedence bug."""

    def step_post_backward(self, params, optimizers, state, step, info, packed=False):
        if step >= self.refine_stop_iter:
            return
        self._update_state(params, state, info, packed=packed)
        if (step > self.refine_start_iter and step % self.refine_every == 0
                and step % self.reset_every >= self.pause_refine_after_reset):
            self._grow_gs(params, optimizers, state, step)
            self._prune_gs(params, optimizers, state, step)
            state["grad2d"].zero_()
            state["count"].zero_()
            if self.refine_scale2d_stop_iter > 0:
                state["radii"].zero_()
            torch.cuda.empty_cache()


def argv_steps():
    argv = sys.argv
    for i, a in enumerate(argv):
        if a == "--steps" and i + 1 < len(argv):
            return int(argv[i + 1])
        if a.startswith("--steps="):
            return int(a.split("=", 1)[1])
    return None


def check_steps(step, total):
    """Fail at step 1 if ss-train --steps differs from the schedule's total (bake step)."""
    if step == 1:
        steps = argv_steps()
        if steps is not None and steps != int(total):
            raise ValueError(f"ss-train --steps {steps} != total_steps {total}: the "
                             "SplatFields schedule/bake assumes the paper's 40000 iterations")


def _strategy(base):
    s = NoResetStrategy(**base)
    if s.refine_scale2d_stop_iter != 0:
        raise ValueError("screen-size pruning must stay disabled (dead in the official loop)")
    return s


class SplatFieldsDensity:
    def __init__(self, spec):
        unknown = set(spec.config) - set(DEFAULTS)
        if unknown:
            raise ValueError(f"Unknown splatfields-density options: {sorted(unknown)}")
        self.cfg = {**DEFAULTS, **spec.config}
        self.strategy = _strategy(self.cfg["base"])
        self.requires = {"gradient.screen.signed"}
        self.net = None
        self.strategy_state = None
        self.history = []

    @property
    def state(self):
        return {"strategy": self.strategy_state,
                "splatfields": {"events": self.history,
                                "network": ({k: v.detach().cpu() for k, v in
                                             self.net.state_dict().items()}
                                            if self.net is not None else None)}}

    def lr(self, step):
        c = self.cfg
        return network.exp_lr(step, c["lr_init"], c["lr_final"], c["lr_max_steps"])

    def initialize(self, params, optimizers, scene_scale=1.0):
        self.strategy.check_sanity(params, optimizers)
        self.strategy_state = self.strategy.initialize_state(scene_scale=scene_scale)
        device = params["means"].device
        self.net = network.SplatFieldsNet(self.cfg["net"], self.cfg["net_seed"]).to(device)
        self.net.optimizer = torch.optim.Adam(
            [{"params": list(self.net.parameters()), "lr": self.lr(0), "name": "deform"}],
            lr=0.0, eps=float(self.cfg["adam_eps"]))
        network.ACTIVE["net"] = self.net

    def before_backward(self, params, optimizers, step, outputs):
        check_steps(step, self.cfg["total_steps"])
        self.strategy.step_pre_backward(params, optimizers, self.strategy_state, step,
                                        outputs["info"])

    def after_backward(self, params, optimizers, step, outputs):
        net = self.net
        if net.baked:
            return
        net.optimizer.step()
        net.optimizer.zero_grad(set_to_none=True)
        for group in net.optimizer.param_groups:
            group["lr"] = self.lr(step)
        before = len(params["means"])
        self.strategy.step_post_backward(params, optimizers, self.strategy_state, step,
                                         outputs["info"], packed=False)
        if len(params["means"]) != before:
            self.history.append({"step": step, "before": before, "after": len(params["means"])})
        if step == int(self.cfg["total_steps"]):
            self.bake(params)
            record = {"splatfields": "bake", "step": step, "gaussians": len(params["means"])}
            self.history.append(record)
            print(json.dumps(record), flush=True)

    @torch.no_grad()
    def bake(self, params):
        """Write the network-predicted splats into the base fields (paper: networks discarded)."""
        pred = self.net(params["means"], params["scales"].exp())
        params["means"].copy_(pred["means"])
        params["scales"].copy_(pred["scales"].abs().clamp_min(1e-12).log())  # Sigma uses s s^T
        params["quats"].copy_(pred["quats"])
        op = pred["opacities"].clamp(1e-6, 1 - 1e-6)
        params["opacities"].copy_(torch.log(op / (1 - op)))
        params["sh0"].copy_(((pred["rgb"] - 0.5) / C0)[:, None, :])
        params["shN"].zero_()
        self.net.baked = True


class Stage1Density:
    def __init__(self, spec):
        unknown = set(spec.config) - set(STAGE1_DEFAULTS)
        if unknown:
            raise ValueError(f"Unknown sf3dgs-density options: {sorted(unknown)}")
        self.cfg = {**STAGE1_DEFAULTS, **spec.config}
        self.strategy = _strategy(self.cfg["base"])
        self.requires = {"gradient.screen.signed"}
        self.state = None

    def means_lr(self, step):
        c = self.cfg
        return network.exp_lr(step, c["means_lr_init"], c["means_lr_final"],
                              c["means_lr_max_steps"])

    def initialize(self, params, optimizers, scene_scale=1.0):
        self.strategy.check_sanity(params, optimizers)
        self.state = self.strategy.initialize_state(scene_scale=scene_scale)
        optimizers["means"].param_groups[0]["lr"] = self.means_lr(0)

    def before_backward(self, params, optimizers, step, outputs):
        check_steps(step, self.cfg["total_steps"])
        self.strategy.step_pre_backward(params, optimizers, self.state, step, outputs["info"])

    def after_backward(self, params, optimizers, step, outputs):
        self.strategy.step_post_backward(params, optimizers, self.state, step, outputs["info"],
                                         packed=False)
        # official update_learning_rate(iteration) after the step; replaces the runtime's
        # 0.01^(step/steps) schedule (set just before this hook) for the next step
        optimizers["means"].param_groups[0]["lr"] = self.means_lr(step)


def create_density(spec):
    return SplatFieldsDensity(spec)


def create_stage1_density(spec):
    return Stage1Density(spec)
