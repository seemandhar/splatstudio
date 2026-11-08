"""3DGS adaptive density control (runtime gsplat DefaultStrategy + 3DGS opacity reset) with the
"3DGS + Ours*" settings of Opti3DGS Sec. 4: scaling_lr = 0.01, densification_interval = 500
(gsplat `refine_every`), densify_grad_threshold = 0.0001 (gsplat `grow_grad2d`)."""

from dataclasses import replace

from splatstudio.backends.gsplat import Density

OURS_STAR = {"scales_lr": 0.01, "refine_every": 500, "grow_grad2d": 0.0001}
PLAIN_3DGS = {"scales_lr": 0.005, "refine_every": 100, "grow_grad2d": 0.0002}


class OptiDensity(Density):
    def __init__(self, spec):
        config = dict(spec.config)
        if "scales_lr" not in config:
            raise ValueError("opti-density needs an explicit scales_lr")
        self.scales_lr = float(config.pop("scales_lr"))
        if self.scales_lr <= 0:
            raise ValueError("scales_lr must be positive")
        super().__init__(replace(spec, config=config))

    def initialize(self, params, optimizers, scene_scale=1.0):
        super().initialize(params, optimizers, scene_scale)
        # Only the scales optimizer's lr changes (3DGS scaling_lr); gsplat ops preserve
        # param_groups when they replace parameters during densification.
        for group in optimizers["scales"].param_groups:
            group["lr"] = self.scales_lr


def create_density(spec):
    return OptiDensity(spec)
