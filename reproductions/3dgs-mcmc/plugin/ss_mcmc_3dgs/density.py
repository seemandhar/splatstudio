"""3DGS-MCMC density: gsplat MCMCStrategy (relocation of opacity<=0.005 Gaussians onto alive ones
sampled by opacity, +5 % births up to cap_max, every 100 steps in (500, 25000); SGLD position
noise every step with noise_lr 5e5), which matches the official train.py / gaussian_model.py.
Adds what the built-in `gsplat-mcmc` lacks: per-scene cap (official configs) and the official
initialisation (opacity 0.5, scale = 0.1 * sqrt(mean 3-NN squared distance)).
Cap lookup pattern from reproductions/mh-3dgs/plugin/ss_mh_3dgs/control.py."""

import math

import torch
from splatstudio.backends.gsplat import MCMCDensity
from splatstudio.runtime import knn_log_scales

from .scenes import CAPS, scene_name

DEFAULTS = {
    "cap_max": None,  # None -> official configs/<scene>.json cap for the --data scene
    "noise_lr": 5e5,
    "refine_start_iter": 500,
    "refine_stop_iter": 25000,
    "refine_every": 100,
    "min_opacity": 0.005,
    "init_opacity": 0.5,
    "init_scale": 0.1,
}
_STRATEGY_KEYS = ("noise_lr", "refine_start_iter", "refine_stop_iter", "refine_every",
                  "min_opacity")


class _Spec:
    def __init__(self, config):
        self.config = config


class OfficialMCMCDensity(MCMCDensity):
    def __init__(self, spec):
        unknown = set(spec.config) - set(DEFAULTS)
        if unknown:
            raise ValueError(f"Unknown mcmc-3dgs-density options: {sorted(unknown)}")
        self.c = {**DEFAULTS, **spec.config}
        super().__init__(_Spec({k: self.c[k] for k in _STRATEGY_KEYS} | {"cap_max": 1}))
        self.cap_source = None

    def resolve_cap(self):
        if self.c["cap_max"] is not None:
            return int(self.c["cap_max"]), "config"
        scene = scene_name()
        if scene is None:
            return 1_000_000, "no-data-default"  # contract probes only (no training run)
        if scene not in CAPS:
            raise RuntimeError(f"mcmc-3dgs: no official cap for scene {scene!r}; set cap_max")
        return CAPS[scene], f"official-config:{scene}"

    def initialize(self, params, optimizers, scene_scale=1.0):
        cap, self.cap_source = self.resolve_cap()
        self.strategy.cap_max = cap
        with torch.no_grad():
            o = float(self.c["init_opacity"])
            params["opacities"].fill_(math.log(o / (1 - o)))
            params["scales"].copy_(
                knn_log_scales(params["means"].detach()) + math.log(float(self.c["init_scale"]))
            )
        super().initialize(params, optimizers, scene_scale)


def create_density(spec):
    return OfficialMCMCDensity(spec)
