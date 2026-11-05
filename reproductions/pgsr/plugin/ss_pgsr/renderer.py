"""PGSR renderer component: official plane rasterizer (raster.py), black background (official
``white_background`` False). Plane/normal maps are rendered when the link asks for them
(official ``return_plane = iteration > single_view_weight_from_iter``; always outside training).
"""

import torch

from .link import LINK
from .raster import render

CAPABILITIES = {
    "render.rgb", "render.depth.plane", "render.normal.camera", "render.distance.plane",
    "render.alpha", "render.normal_from_depth.camera", "gradient.screen.signed",
    "gradient.screen.absolute", "render.observe_count", "render.camera",
}
DEFAULTS = {"background": [0.0, 0.0, 0.0]}


class Renderer:
    capabilities = CAPABILITIES

    def __init__(self, spec):
        unknown = set(spec.config) - set(DEFAULTS)
        if unknown:
            raise ValueError(f"Unknown PGSR renderer options: {sorted(unknown)}")
        cfg = {**DEFAULTS, **spec.config}
        self.background = [float(v) for v in cfg["background"]]
        self.capabilities = set(CAPABILITIES)
        self._bg = {}
        LINK.renderer = self

    def bg(self, device):
        if device not in self._bg:
            self._bg[device] = torch.tensor(self.background, dtype=torch.float32, device=device)
        return self._bg[device]

    def render(self, params, camera, sh_degree=3, render_geo=None):
        if render_geo is None:
            render_geo = LINK.render_geo
        out = render(params, camera, sh_degree, self.bg(params["means"].device), render_geo)
        out["sh_degree"] = sh_degree
        return out


def create_renderer(spec):
    return Renderer(spec)
