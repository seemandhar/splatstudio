"""GOF renderer component: official ray-tracing rasterizer (raster.py), black background
(official white_background False), kernel_size 0 (no 2D Mip filter in GOF)."""

import torch

from .raster import render

CAPABILITIES = {
    "render.rgb", "render.depth.median_ray_z", "render.alpha", "render.normal.world",
    "render.normal_from_depth.world", "render.distortion.gof", "gradient.screen.gof",
}
DEFAULTS = {"background": [0.0, 0.0, 0.0], "kernel_size": 0.0}


class Renderer:
    capabilities = CAPABILITIES

    def __init__(self, spec):
        unknown = set(spec.config) - set(DEFAULTS)
        if unknown:
            raise ValueError(f"Unknown GOF renderer options: {sorted(unknown)}")
        cfg = {**DEFAULTS, **spec.config}
        self.background = [float(v) for v in cfg["background"]]
        self.kernel_size = float(cfg["kernel_size"])
        self.capabilities = set(CAPABILITIES)
        self._bg = {}

    def render(self, params, camera, sh_degree=3):
        dev = params["means"].device
        if dev not in self._bg:
            self._bg[dev] = torch.tensor(self.background, dtype=torch.float32, device=dev)
        return render(params, camera, sh_degree, self._bg[dev], self.kernel_size)


def create_renderer(spec):
    return Renderer(spec)
