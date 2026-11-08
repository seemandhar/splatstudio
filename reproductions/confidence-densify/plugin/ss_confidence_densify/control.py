"""3DGS-MCMC control at the same budget (the paper's baseline row): the built-in gsplat-mcmc
density (gsplat MCMCStrategy: opacity sampling, opacity-threshold relocation, position noise)
plus the 3DGS-MCMC initial opacity 0.5; the regulariser is the separate `mcmc-reg` loss."""

import math
from dataclasses import replace

import torch
from splatstudio.backends.gsplat import MCMCDensity


class ControlDensity(MCMCDensity):
    def __init__(self, spec):
        cfg = dict(spec.config)
        self.init_opacity = float(cfg.pop("init_opacity", 0.5))
        super().__init__(replace(spec, config=cfg))

    def initialize(self, params, optimizers, scene_scale=1.0):
        with torch.no_grad():
            o = self.init_opacity
            params["opacities"].fill_(math.log(o / (1 - o)))
        super().initialize(params, optimizers, scene_scale)


def create_control_density(spec):
    return ControlDensity(spec)
