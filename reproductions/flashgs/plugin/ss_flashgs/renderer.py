"""SplatStudio renderer: gsplat while gradients are needed, FlashGS for inference renders.

The official FlashGS release has no backward pass, so it cannot train. Under
``torch.no_grad()`` (SplatStudio's ``measure``: validation at the end of training and
``ss-eval``) this renderer returns the FlashGS image instead of gsplat's. Inference outputs
contain only ``rgb`` (uint8 -> float, as FlashGS writes it): FlashGS produces no depth/alpha.
"""

import torch
from splatstudio.backends.gsplat import Renderer as GsplatRenderer
from splatstudio.specs import ComponentSpec

from . import core


class FlashGSRenderer:
    def __init__(self, spec):
        config = dict(spec.config)
        self.inference_backend = config.pop("inference_backend", "flashgs")
        self.tile = int(config.pop("tile", core.TILE))
        if self.inference_backend not in {"flashgs", "gsplat"}:
            raise ValueError(f"inference_backend must be flashgs|gsplat, got {self.inference_backend}")
        if config:
            raise ValueError(f"Unknown FlashGS renderer options: {sorted(config)}")
        self.training = GsplatRenderer(ComponentSpec(spec.name, "renderer", "", (), (), {}))
        self.capabilities = set(self.training.capabilities)
        self._raster = None
        self._cache = None

    def _prepared(self, params, sh_degree):
        key = (id(params), params["means"].data_ptr(), len(params["means"]), sh_degree,
               params["means"]._version, params["sh0"]._version)
        if self._cache is None or self._cache[0] != key:
            self._cache = (key, core.prepare(params, sh_degree))
        return self._cache[1]

    def render(self, params, camera, sh_degree=3):
        if torch.is_grad_enabled() or self.inference_backend == "gsplat":
            return self.training.render(params, camera, sh_degree)
        if self._raster is None:
            self._raster = core.FlashGSRasterizer(params["means"].device, self.tile)
        prep = self._prepared(params, sh_degree)
        image = self._raster.forward(prep, camera, torch.ones(3))
        return {
            "rgb": image.float() / 255.0,
            "info": {"backend": "flashgs", **self._raster.last},
        }


def create_renderer(spec):
    return FlashGSRenderer(spec)
