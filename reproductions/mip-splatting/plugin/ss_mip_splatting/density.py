"""3DGS densification (gsplat DefaultStrategy) + Mip-Splatting 3D filter updates and reset.

Official train.py (commit 1ecf7d5): compute_3D_filter(trainCameras) at iteration 0, after
every densify_and_prune, and every 100 iterations after densification until total - 100;
reset_opacity is filter-aware (clamps the filtered opacity to 0.01).

This controller wraps gsplat's DefaultStrategy directly (not splatstudio's base Density
wrapper, whose own unfiltered reset would double the reset). The only opacity reset is the
filter-aware one below; gsplat 1.5.x's built-in DefaultStrategy reset never fires.
"""

import torch
from gsplat import DefaultStrategy
from gsplat.strategy.ops import _update_param_with_optimizer
from splatstudio.backends.gsplat import GSPLAT_RESET_BROKEN

from .filters import compute_filter_3d, is_filter_step, is_reset_step, reset_logits

OWN_KEYS = {
    "filter_coef",
    "near",
    "frustum_margin",
    "filter_interval",
    "densify_until",
    "total_steps",
    "opacity_reset_every",
    "opacity_reset_value",
}


class MipDensity:
    def __init__(self, spec):
        config = dict(spec.config)
        own = {k: config.pop(k) for k in OWN_KEYS}
        self.filter_coef = float(own["filter_coef"])
        self.near = float(own["near"])
        self.margin = float(own["frustum_margin"])
        self.interval = int(own["filter_interval"])
        self.densify_until = int(own["densify_until"])
        self.total_steps = int(own["total_steps"])
        self.reset_every = int(own["opacity_reset_every"])
        self.reset_value = float(own["opacity_reset_value"])
        self.strategy = DefaultStrategy(**config)
        if self.strategy.absgrad:
            raise ValueError("Mip-Splatting uses 3DGS's signed screen-space gradients")
        if self.strategy.refine_stop_iter != self.densify_until:
            raise ValueError("densify_until must equal the strategy's refine_stop_iter")
        if not GSPLAT_RESET_BROKEN:
            # A gsplat whose DefaultStrategy reset works would reset a second, unfiltered time.
            raise RuntimeError("gsplat DefaultStrategy resets opacity itself; adapt MipDensity")
        self.requires = {"gradient.screen.signed"}
        self.state = None
        self.params = None
        self.cameras = None
        self.filter_updates = 0
        self.resets = 0

    def set_training_cameras(self, cameras):
        """Runtime hook (called right after initialize, training split only)."""
        cameras = list(cameras)
        if not cameras:
            raise ValueError("Mip-Splatting needs at least one training camera")
        self.cameras = cameras
        if self.params is not None:
            self.update_filter(self.params)  # official: compute_3D_filter before iteration 1

    def _require_cameras(self):
        if not self.cameras:
            raise RuntimeError(
                "Mip-Splatting 3D filter needs the training cameras: the runtime must call "
                "density.set_training_cameras(...) before training"
            )

    def update_filter(self, params):
        self._require_cameras()
        device = params["means"].device
        if self.cameras[0]["viewmat"].device != device:
            self.cameras = [
                {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in c.items()}
                for c in self.cameras
            ]
        f = compute_filter_3d(
            params["means"], self.cameras, self.filter_coef, self.near, self.margin
        )
        params["filter_3d"].data.copy_(f.to(params["filter_3d"].dtype))
        self.filter_updates += 1

    def reset_opacity(self, params, optimizers):
        with torch.no_grad():
            logits = reset_logits(
                params["scales"], params["opacities"], params["filter_3d"], self.reset_value
            )

        def param_fn(name, p):
            return torch.nn.Parameter(logits.to(p.dtype), requires_grad=p.requires_grad)

        _update_param_with_optimizer(
            param_fn, lambda key, v: torch.zeros_like(v), params, optimizers, ["opacities"]
        )
        self.resets += 1

    def initialize(self, params, optimizers, scene_scale=1.0):
        self.strategy.check_sanity(params, optimizers)
        self.state = self.strategy.initialize_state(scene_scale=scene_scale)
        self.params = params
        if self.cameras:
            self.update_filter(params)

    def before_backward(self, params, optimizers, step, outputs):
        # The step's render already used filter_3d: it must come from the training cameras.
        self._require_cameras()
        self.strategy.step_pre_backward(params, optimizers, self.state, step, outputs["info"])

    def after_backward(self, params, optimizers, step, outputs):
        # gsplat grow/prune (new Gaussians copy filter_3d; it is recomputed right below).
        self.strategy.step_post_backward(
            params, optimizers, self.state, step, outputs["info"], packed=False
        )
        if is_filter_step(
            step, self.interval, self.strategy.refine_start_iter, self.densify_until,
            self.total_steps,
        ):
            self.update_filter(params)
        if is_reset_step(step, self.reset_every, self.densify_until):
            self.reset_opacity(params, optimizers)


def create_density(spec):
    return MipDensity(spec)
