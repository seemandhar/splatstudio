"""2DGS renderer component: official surfel rasteriser (raster.py), black background
(official ``white_background`` False), surface depth = expected depth (``depth_ratio`` 0, the
official default for unbounded scenes; 1 = median depth as used for DTU)."""

import torch

from .raster import render

CAPABILITIES = {
    "render.rgb", "render.depth.surface", "render.alpha", "render.normal.world.unnormalized",
    "render.normal_from_depth.world.alpha", "render.distortion.ndc_l2", "render.depth.median",
    "gradient.screen.projected_center",
}
DEFAULTS = {"background": [0.0, 0.0, 0.0], "depth_ratio": 0.0}


class Renderer:
    def __init__(self, spec):
        unknown = set(spec.config) - set(DEFAULTS)
        if unknown:
            raise ValueError(f"Unknown 2DGS renderer options: {sorted(unknown)}")
        cfg = {**DEFAULTS, **spec.config}
        self.background = [float(v) for v in cfg["background"]]
        self.depth_ratio = float(cfg["depth_ratio"])
        if not 0.0 <= self.depth_ratio <= 1.0:
            raise ValueError("depth_ratio must be in [0, 1]")
        self.capabilities = set(CAPABILITIES)
        self._bg = {}

    def bg(self, device):
        if device not in self._bg:
            self._bg[device] = torch.tensor(self.background, dtype=torch.float32, device=device)
        return self._bg[device]

    def render(self, params, camera, sh_degree=3):
        return render(params, camera, sh_degree, self.bg(params["means"].device),
                      self.depth_ratio)


def create_renderer(spec):
    return Renderer(spec)
