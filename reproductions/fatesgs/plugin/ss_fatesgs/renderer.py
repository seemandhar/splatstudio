"""FatesGS renderer component: the official 2DGS rasterizer (see raster.py), black background
(official white_background False). Adds ``camera`` to the outputs so the prior losses can look
up the training view."""

import torch

from .raster import render

CAPABILITIES = {
    "render.rgb", "render.depth.expected_z", "render.normal.world",
    "render.normal_from_depth.world", "render.distortion.2dgs", "gradient.surfel",
}
DEFAULTS = {"background": [0.0, 0.0, 0.0]}


class Renderer:
    capabilities = CAPABILITIES

    def __init__(self, spec):
        unknown = set(spec.config) - set(DEFAULTS)
        if unknown:
            raise ValueError(f"Unknown FatesGS renderer options: {sorted(unknown)}")
        self.background = [float(v) for v in {**DEFAULTS, **spec.config}["background"]]
        self.capabilities = set(CAPABILITIES)
        self._bg = {}

    def render(self, params, camera, sh_degree=3):
        dev = params["means"].device
        if dev not in self._bg:
            self._bg[dev] = torch.tensor(self.background, dtype=torch.float32, device=dev)
        out = render(params, camera, sh_degree, self._bg[dev])
        out["camera"] = camera
        return out


def create_renderer(spec):
    return Renderer(spec)
