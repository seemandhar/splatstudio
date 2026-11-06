"""3DGS adaptive density control driven by the active loop's per-round iteration (cur_iter).

Wraps ``splatstudio.backends.gsplat.Density`` (gsplat DefaultStrategy + 3DGS opacity reset every
3000 steps) and adds the official white-background reset at cur_iter == densify_from_iter
(active_train.py: ``cur_iter % opacity_reset_interval == 0 or (white_background and
cur_iter == densify_from_iter)``). Skipped steps are ignored; after each acquisition the
controller restores the initial Gaussians and calls ``reset`` (fresh densification stats).
"""

from gsplat.strategy.ops import reset_opa
from splatstudio.backends.gsplat import Density as GsplatDensity

from .active import ActiveController
from .renderer import latest_controller

DEFAULTS = {
    "grow_grad2d": 0.0002,  # densify_grad_threshold
    "prune_opa": 0.005,  # --min_opacity
    "grow_scale3d": 0.01,  # percent_dense
    "prune_scale3d": 0.1,
    "refine_start_iter": 500,  # densify_from_iter
    "refine_stop_iter": 10000,  # densify_until_iter (FisherRF arguments/__init__.py)
    "reset_every": 3000,  # opacity_reset_interval
    "refine_every": 100,  # densification_interval
    "white_background_reset": True,  # --white_background extra reset at densify_from_iter
}
REQUIRES = ("gradient.screen.signed",)


class Density:
    def __init__(self, spec):
        config = {**DEFAULTS, **spec.config}
        self.white_reset = bool(config.pop("white_background_reset"))
        self.inner = GsplatDensity(type(spec)(spec.name, spec.kind, spec.factory, spec.requires,
                                              spec.provides, config))
        self.requires = set(self.inner.requires)
        self.controller = latest_controller() or ActiveController({})
        self.scene_scale = 1.0

    @property
    def state(self):
        return {"strategy": self.inner.state, "fisherrf": self.controller.summary()}

    def set_training_cameras(self, cameras):
        self.controller.register_cameras(cameras)

    def initialize(self, params, optimizers, scene_scale=1.0):
        self.scene_scale = scene_scale
        self.inner.initialize(params, optimizers, scene_scale)
        self.controller.register_state(params, optimizers, self)

    def reset(self, params, optimizers):
        self.inner.initialize(params, optimizers, self.scene_scale)

    def before_backward(self, params, optimizers, step, outputs):
        if outputs.get("fisherrf_skip"):
            return
        # outside the active loop (probes) the runtime step is the iteration
        self.inner.before_backward(params, optimizers, outputs.get("fisherrf_cur_iter", step),
                                   outputs)

    def after_backward(self, params, optimizers, step, outputs):
        if outputs.get("fisherrf_skip"):
            return
        cur = outputs.get("fisherrf_cur_iter", step)
        self.inner.after_backward(params, optimizers, cur, outputs)
        strategy = self.inner.strategy
        if (self.white_reset and cur == strategy.refine_start_iter
                and cur < strategy.refine_stop_iter):
            reset_opa(params, optimizers, self.inner.state, value=strategy.prune_opa * 2.0)


def create_density(spec):
    return Density(spec)
