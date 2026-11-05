"""Renderer component: the vendored official rasterizer (raster.py), black background
(official ``white_background`` false)."""

import torch

from .raster import rasterize

CAPABILITIES = {"render.rgb", "render.depth.inverse", "gradient.screen.inria_abs"}
DEFAULTS = {"background": [0.0, 0.0, 0.0]}


class Renderer:
    def __init__(self, spec):
        unknown = set(spec.config) - set(DEFAULTS)
        if unknown:
            raise ValueError(f"Unknown improvedgs-raster options: {sorted(unknown)}")
        cfg = {**DEFAULTS, **spec.config}
        self.background = [float(v) for v in cfg["background"]]
        if len(self.background) != 3:
            raise ValueError("background must have 3 values")
        self.capabilities = set(CAPABILITIES)
        self._bg = {}

    def render(self, params, camera, sh_degree=3):
        dev = params["means"].device
        if dev not in self._bg:
            self._bg[dev] = torch.tensor(self.background, dtype=torch.float32, device=dev)
        color, radii, invdepth, _, means2D = rasterize(params, camera, sh_degree, self._bg[dev])
        out = {"rgb": color.permute(1, 2, 0), "info": {"means2d": means2D, "radii": radii}}
        if invdepth.numel():
            out["inverse_depth"] = invdepth[0][..., None]
        return out


def create_renderer(spec):
    return Renderer(spec)
