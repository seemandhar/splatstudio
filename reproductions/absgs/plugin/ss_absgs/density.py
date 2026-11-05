"""AbsGS adaptive density control (arXiv 2404.10484, §3.3 and §4.1 "Implementation Details").

Clone follows 3DGS (signed view-space gradient, tau=0.0002); split uses the homodirectional
gradient (per-pixel absolute components summed, gsplat ``means2d.absgrad``) with a larger
threshold (0.0004 for AbsGS-0004). Scale threshold tau_S = 0.001 x scene extent.

Official reference: github.com/TY424/AbsGS ``scene/gaussian_model.py`` (densify_and_prune,
densify_and_clone, densify_and_split, add_densification_stats) and
``submodules/diff-gaussian-rasterization-abs/cuda_rasterizer/backward.cu:549-550``.
"""

from dataclasses import dataclass

import torch
from gsplat import DefaultStrategy
from gsplat.strategy.ops import duplicate, reset_opa, split

DEFAULTS = {
    "clone_grad2d": 0.0002,  # paper §4.1: clone follows 3D-GS; code arguments:89
    "split_grad2d": 0.0004,  # AbsGS-0004 (Table 1, code default arguments:90)
    "grow_scale3d": 0.001,  # tau_S, paper §4.1 Baselines; code percent_dense arguments:83
    "prune_opa": 0.005,  # 3DGS, code train.py:130
    "prune_scale3d": 0.1,  # 3DGS big_points_ws, gaussian_model.py:474
    "refine_start_iter": 500,
    "refine_stop_iter": 15000,
    "refine_every": 100,
    "reset_every": 3000,
    "opacity_reset": True,  # 3DGS opacity reset (see AbsGSStrategy.step_post_backward)
    "reduce_every": 0,  # code-only `use_reduce` (opacity <= 0.8 every 500); not used in paper
}


@dataclass
class AbsGSStrategy(DefaultStrategy):
    clone_grad2d: float = DEFAULTS["clone_grad2d"]
    split_grad2d: float = DEFAULTS["split_grad2d"]
    opacity_reset: bool = True
    reduce_every: int = 0

    def initialize_state(self, scene_scale: float = 1.0):
        state = super().initialize_state(scene_scale)
        state["grad2d_abs"] = None
        return state

    def step_post_backward(self, params, optimizers, state, step, info, packed=False):
        # gsplat 1.5.3 DefaultStrategy writes `step % reset_every == 0 & step > 0`, which Python
        # parses as `(step % reset_every == 0) and (0 > 0)` -> the opacity reset never fires.
        # 3DGS/AbsGS reset opacities to 0.01 every 3000 iterations while densifying
        # (train.py:135), so this strategy performs the reset itself (`opacity_reset`).
        if step >= self.refine_stop_iter:
            return
        self._update_state(params, state, info, packed=packed)
        if (
            step > self.refine_start_iter
            and step % self.refine_every == 0
            and step % self.reset_every >= self.pause_refine_after_reset
        ):
            self._grow_gs(params, optimizers, state, step)
            self._prune_gs(params, optimizers, state, step)
            state["grad2d"].zero_()
            state["grad2d_abs"].zero_()
            state["count"].zero_()
            if self.refine_scale2d_stop_iter > 0:
                state["radii"].zero_()
            torch.cuda.empty_cache()
        if self.reduce_every > 0 and step % self.reduce_every == 0:
            # official gaussian_model.py:260 reduce_opacity (clamp to 0.8, reset Adam moments)
            reset_opa(params=params, optimizers=optimizers, state=state, value=0.8)
        if self.opacity_reset and step % self.reset_every == 0:
            reset_opa(params=params, optimizers=optimizers, state=state, value=self.prune_opa * 2.0)

    def _update_state(self, params, state, info, packed=False):
        # Accumulate both ||g|| (signed sum, for clone) and ||g_hat|| (homodirectional, for
        # split), same NDC scaling and visibility as 3DGS add_densification_stats.
        means2d = info[self.key_for_gradient]
        if means2d.grad is None or getattr(means2d, "absgrad", None) is None:
            raise RuntimeError("AbsGS density needs signed and absolute means2d gradients")
        scale = means2d.grad.new_tensor(
            [info["width"] / 2.0 * info["n_cameras"], info["height"] / 2.0 * info["n_cameras"]]
        )
        grads = means2d.grad * scale
        grads_abs = means2d.absgrad * scale
        n = len(params["means"])
        for key in ("grad2d", "grad2d_abs", "count"):
            if state[key] is None:
                state[key] = torch.zeros(n, device=grads.device)
        if packed:
            gs_ids = info["gaussian_ids"]
        else:
            sel = (info["radii"] > 0.0).all(dim=-1)  # [C, N]
            gs_ids = torch.where(sel)[1]
            grads, grads_abs = grads[sel], grads_abs[sel]
        state["grad2d"].index_add_(0, gs_ids, grads.norm(dim=-1))
        state["grad2d_abs"].index_add_(0, gs_ids, grads_abs.norm(dim=-1))
        state["count"].index_add_(0, gs_ids, torch.ones_like(gs_ids, dtype=torch.float32))

    def decisions(self, params, state):
        """Return (is_clone, is_split) masks; official densify_and_clone / densify_and_split."""
        count = state["count"].clamp_min(1)
        grads = state["grad2d"] / count
        grads_abs = state["grad2d_abs"] / count
        is_small = (
            torch.exp(params["scales"]).max(dim=-1).values
            <= self.grow_scale3d * state["scene_scale"]
        )
        is_clone = (grads >= self.clone_grad2d) & is_small
        is_split = (grads_abs >= self.split_grad2d) & ~is_small
        return is_clone, is_split

    @torch.no_grad()
    def _grow_gs(self, params, optimizers, state, step):
        is_clone, is_split = self.decisions(params, state)
        n_clone = int(is_clone.sum().item())
        n_split = int(is_split.sum().item())
        if n_clone > 0:
            duplicate(params=params, optimizers=optimizers, state=state, mask=is_clone)
        # Clones are appended and are never split in the same round (official padded_grad).
        is_split = torch.cat([is_split, torch.zeros(n_clone, dtype=torch.bool, device=is_split.device)])
        if n_split > 0:
            split(
                params=params,
                optimizers=optimizers,
                state=state,
                mask=is_split,
                revised_opacity=self.revised_opacity,
            )
        state["last_clone"], state["last_split"] = n_clone, n_split
        return n_clone, n_split


class Density:
    requires = {"gradient.screen.signed", "gradient.screen.absolute"}

    def __init__(self, spec):
        cfg = {**DEFAULTS, **spec.config}
        self.strategy = AbsGSStrategy(absgrad=True, **cfg)
        self.state = None

    def initialize(self, params, optimizers, scene_scale=1.0):
        self.strategy.check_sanity(params, optimizers)
        self.state = self.strategy.initialize_state(scene_scale=scene_scale)

    def before_backward(self, params, optimizers, step, outputs):
        self.strategy.step_pre_backward(params, optimizers, self.state, step, outputs["info"])

    def after_backward(self, params, optimizers, step, outputs):
        self.strategy.step_post_backward(
            params, optimizers, self.state, step, outputs["info"], packed=False
        )


def create_density(spec):
    return Density(spec)
