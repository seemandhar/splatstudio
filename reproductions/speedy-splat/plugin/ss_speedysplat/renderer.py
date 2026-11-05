"""Speedy-Splat renderer: vendored official rasterizer (raster.py), black background (official
`white_background` False). `rasterizer: stock` selects the unculled 3D-GS rasterizer (control).

outputs["info"]: `means2d` [N,3] (retained grad: Inria NDC-space screen gradient in [:, :2]),
`radii` [N] (3-sigma radius; > 0 iff the Gaussian touches a tile), `num_rendered`
(tile-Gaussian pairs), `kernel_ms` (speedy forward CUDA-event time), width/height.
"""

import torch

from .raster import RASTERIZERS, render

CAPABILITIES = {"render.rgb", "gradient.screen.signed"}
DEFAULTS = {"background": [0.0, 0.0, 0.0], "rasterizer": "speedy"}


class Renderer:
    def __init__(self, spec):
        unknown = set(spec.config) - set(DEFAULTS)
        if unknown:
            raise ValueError(f"Unknown Speedy-Splat renderer options: {sorted(unknown)}")
        cfg = {**DEFAULTS, **spec.config}
        if cfg["rasterizer"] not in RASTERIZERS:
            raise ValueError(f"rasterizer must be one of {RASTERIZERS}")
        self.rasterizer = cfg["rasterizer"]
        self.background = [float(v) for v in cfg["background"]]
        self.capabilities = set(CAPABILITIES)
        self._bg = {}

    def background_tensor(self, device):
        if device not in self._bg:
            self._bg[device] = torch.tensor(self.background, dtype=torch.float32,
                                            device=device)
        return self._bg[device]

    def render(self, params, camera, sh_degree=3):
        bg = self.background_tensor(params["means"].device)
        color, means2d, radii, stats = render(params, camera, sh_degree, bg, self.rasterizer)
        return {
            "rgb": color.permute(1, 2, 0),
            "info": {"means2d": means2d, "radii": radii, **stats,
                     "width": int(camera["width"]), "height": int(camera["height"])},
        }


def create_renderer(spec):
    return Renderer(spec)
