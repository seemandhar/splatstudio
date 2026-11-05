"""AtomGS renderer component: official feature rasterizer (raster.py), black background
(official ``white_background`` False)."""

import torch

from .raster import render

CAPABILITIES = {
    "render.rgb", "render.depth.expected_z", "render.depth.mean_accumulated",
    "render.depth.median", "render.alpha", "render.normal_from_depth.world",
    "gradient.screen.inria",
}
DEFAULTS = {"background": [0.0, 0.0, 0.0]}


class Renderer:
    capabilities = CAPABILITIES

    def __init__(self, spec):
        unknown = set(spec.config) - set(DEFAULTS)
        if unknown:
            raise ValueError(f"Unknown AtomGS renderer options: {sorted(unknown)}")
        cfg = {**DEFAULTS, **spec.config}
        self.background = [float(v) for v in cfg["background"]]
        self.capabilities = set(CAPABILITIES)
        self._bg = {}

    def render(self, params, camera, sh_degree=3):
        dev = params["means"].device
        if dev not in self._bg:
            self._bg[dev] = torch.tensor(self.background, dtype=torch.float32, device=dev)
        return render(params, camera, sh_degree, self._bg[dev])


def create_renderer(spec):
    return Renderer(spec)
