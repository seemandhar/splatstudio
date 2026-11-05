"""RaDe-GS renderer component: official rasterized-depth/normal kernel (raster.py), black
background (official white_background False), no 2D filter (kernel_size = 0 in the kernel)."""

import torch

from .raster import render

CAPABILITIES = {
    "render.rgb", "render.depth", "render.depth.median", "render.alpha",
    "render.normal.camera", "render.normal_from_depth.camera", "render.distortion.radegs",
    "gradient.screen.gof",
}
DEFAULTS = {"background": [0.0, 0.0, 0.0]}


class Renderer:
    capabilities = CAPABILITIES

    def __init__(self, spec):
        unknown = set(spec.config) - set(DEFAULTS)
        if unknown:
            raise ValueError(f"Unknown RaDe-GS renderer options: {sorted(unknown)}")
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
