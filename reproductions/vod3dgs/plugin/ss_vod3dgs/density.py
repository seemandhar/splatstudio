"""3DGS adaptive density control with VoD-3DGS changes: max-over-training-views opacity prune
(Eq. 10), eigen reset of Ŝ at every opacity reset (Eq. 11), opacity lr_γ/4 (§5.1)."""

import torch
from gsplat import DefaultStrategy
from gsplat.strategy.ops import remove, reset_opa
from splatstudio.backends.gsplat import GSPLAT_RESET_BROKEN, resets_opacity

from .vod import _CONTEXT, camera_center, eigen_reset, max_view_logits

DEFAULTS = {"variant": "S", "opacity_lr": 0.05 / 4}


class VoDStrategy(DefaultStrategy):
    train_centers = None

    @torch.no_grad()
    def _prune_gs(self, params, optimizers, state, step):
        if self.train_centers is None:
            raise RuntimeError("VoD-3DGS prune needs the training cameras (set_training_cameras)")
        alpha_max = torch.sigmoid(
            max_view_logits(params["opacities"], params["vod_S"], params["means"], self.train_centers)
        )
        is_prune = alpha_max < self.prune_opa  # Eq. 10
        if step > self.reset_every:
            is_too_big = torch.exp(params["scales"]).max(dim=-1).values > self.prune_scale3d * state["scene_scale"]
            if step < self.refine_scale2d_stop_iter:
                is_too_big |= state["radii"] > self.prune_scale2d
            is_prune = is_prune | is_too_big
        n_prune = int(is_prune.sum().item())
        if n_prune > 0:
            remove(params=params, optimizers=optimizers, state=state, mask=is_prune)
        return n_prune


@torch.no_grad()
def reset_vod_S(params, optimizers, variant):
    """Eq. 11 in place (keeps the Parameter object and requires_grad); zero its Adam moments."""
    p = params["vod_S"]
    p.data.copy_(eigen_reset(p.data, variant))
    st = optimizers["vod_S"].state.get(p, {})
    for key in ("exp_avg", "exp_avg_sq"):
        if key in st:
            st[key].zero_()


class Density:
    requires = frozenset({"gradient.screen.signed"})

    def __init__(self, spec):
        config = {**DEFAULTS, **spec.config}
        self.variant = config.pop("variant")
        if self.variant not in ("S", "L"):
            raise ValueError("variant must be 'S' or 'L'")
        self.opacity_lr = float(config.pop("opacity_lr"))
        self.opacity_reset = bool(config.pop("opacity_reset", True))
        self.strategy = VoDStrategy(**config)
        self.state = None
        self.resets = []

    def initialize(self, params, optimizers, scene_scale=1.0):
        self.strategy.check_sanity(params, optimizers)
        self.state = self.strategy.initialize_state(scene_scale=scene_scale)
        optimizers["opacities"].param_groups[0]["lr"] = self.opacity_lr  # §5.1: lr_γ / 4

    def set_training_cameras(self, cameras):
        centers = torch.stack([camera_center(c["viewmat"]) for c in cameras])
        self.strategy.train_centers = centers
        _CONTEXT["train_centers"] = centers

    def before_backward(self, params, optimizers, step, outputs):
        self.strategy.step_pre_backward(params, optimizers, self.state, step, outputs["info"])

    def after_backward(self, params, optimizers, step, outputs):
        self.strategy.step_post_backward(params, optimizers, self.state, step, outputs["info"], packed=False)
        if self.opacity_reset and resets_opacity(self.strategy, step):
            if GSPLAT_RESET_BROKEN:
                reset_opa(params, optimizers, self.state, value=self.strategy.prune_opa * 2.0)
            reset_vod_S(params, optimizers, self.variant)  # Eq. 11, same schedule as the γ reset
            self.resets.append(step)
        for name, p in params.items():
            if not (p.is_leaf and p.requires_grad):
                raise RuntimeError(f"{name} is no longer a trainable leaf after density control")


def create_density(spec):
    return Density(spec)
