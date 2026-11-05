"""Analytic-Splatting renderer: vendored official rasterizer (raster.py), black background
(official `white_background` False for Mip-NeRF 360 / T&T / DB).

outputs["info"]: `means2d` [N,3] (retained grad: Inria NDC-space screen gradient in [:, :2]),
`radii` [N], image `width`/`height`.
"""

import torch

from .raster import render

CAPABILITIES = {"render.rgb", "gradient.screen.signed"}
DEFAULTS = {"background": [0.0, 0.0, 0.0]}


class Renderer:
    def __init__(self, spec):
        unknown = set(spec.config) - set(DEFAULTS)
        if unknown:
            raise ValueError(f"Unknown Analytic-Splatting renderer options: {sorted(unknown)}")
        cfg = {**DEFAULTS, **spec.config}
        self.background = [float(v) for v in cfg["background"]]
        self.capabilities = set(CAPABILITIES)
        self._bg = {}

    def render(self, params, camera, sh_degree=3):
        dev = params["means"].device
        if dev not in self._bg:
            self._bg[dev] = torch.tensor(self.background, dtype=torch.float32, device=dev)
        color, means2d, radii = render(params, camera, sh_degree, self._bg[dev])
        return {
            "rgb": color.permute(1, 2, 0),
            "info": {"means2d": means2d, "radii": radii,
                     "width": int(camera["width"]), "height": int(camera["height"])},
        }


def create_renderer(spec):
    return Renderer(spec)
