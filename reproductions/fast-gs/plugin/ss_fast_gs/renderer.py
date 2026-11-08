"""FastGS renderer: vendored official rasterizer (raster.py) with the Compact Box factor
``mult`` (beta; official default 0.5, T&T / Deep Blending scripts 0.7) used for training and
evaluation (official render.py passes the same --mult), black background (official
``white_background`` False).

outputs["info"]: ``means2d`` [N,4] (retained grad: Inria NDC-space screen gradient, columns
0-1 signed, 2-3 AbsGS absolute sums), ``radii`` [N], ``num_rendered``, width/height.
Like the SplatStudio Taming-3DGS port, the frame's ground-truth image and camera are forwarded
(``target_image``, ``camera``) so the density can score arbitrary cached training views.
"""

import torch

from .raster import render

TARGET_CAPABILITY = "training.target_image"
FASTGS_GRADIENT = "gradient.screen.fastgs4"
CAPABILITIES = {"render.rgb", "gradient.screen.signed", FASTGS_GRADIENT, TARGET_CAPABILITY}
DEFAULTS = {"background": [0.0, 0.0, 0.0], "mult": 0.5}


class Renderer:
    def __init__(self, spec):
        unknown = set(spec.config) - set(DEFAULTS)
        if unknown:
            raise ValueError(f"Unknown FastGS renderer options: {sorted(unknown)}")
        cfg = {**DEFAULTS, **spec.config}
        self.mult = float(cfg["mult"])
        if not self.mult > 0:
            raise ValueError("mult must be > 0")
        self.background = [float(v) for v in cfg["background"]]
        self.capabilities = set(CAPABILITIES)
        self._bg = {}

    def background_tensor(self, device):
        key = str(device)
        if key not in self._bg:
            self._bg[key] = torch.tensor(self.background, dtype=torch.float32, device=device)
        return self._bg[key]

    def render(self, params, camera, sh_degree=3):
        bg = self.background_tensor(params["means"].device)
        color, means2d, radii, stats = render(params, camera, sh_degree, bg, self.mult)
        return {
            "rgb": color.permute(1, 2, 0),
            "info": {"means2d": means2d, "radii": radii, "num_rendered": stats["num_rendered"],
                     "width": int(camera["width"]), "height": int(camera["height"])},
            "target_image": camera.get("image"),
            "camera": {k: camera[k] for k in ("K", "viewmat", "width", "height")},
        }


def create_renderer(spec):
    return Renderer(spec)
