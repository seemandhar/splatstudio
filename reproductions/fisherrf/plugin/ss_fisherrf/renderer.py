"""gsplat training renderer + the FisherRF active-selection gate (see active.py)."""

import torch
from splatstudio.backends.gsplat import Renderer as GsplatRenderer

from .active import ActiveController

CAPABILITIES = ("gradient.screen.signed", "render.depth.expected_z", "render.rgb")

# The density and loss factories of the same build pick up the renderer's controller
# (components.build instantiates components in declaration order: renderer first).
_LATEST = {"controller": None}


def latest_controller():
    return _LATEST["controller"]


class Renderer:
    def __init__(self, spec):
        self.inner = GsplatRenderer(type(spec)(spec.name, spec.kind, spec.factory, spec.requires,
                                               spec.provides, {}))
        self.capabilities = set(self.inner.capabilities)
        self.controller = ActiveController(dict(spec.config))
        _LATEST["controller"] = self.controller

    def render(self, params, camera, sh_degree=3):
        mode = self.controller.begin(params, camera)
        if mode == "skip":
            dummy = torch.zeros((), device=params["means"].device, requires_grad=True)
            return {"fisherrf_skip": True, "fisherrf_dummy": dummy}
        if mode == "eval":
            return self.inner.render(params, camera, sh_degree)
        out = self.inner.render(params, camera, self.controller.sh_degree)
        out["fisherrf_skip"] = False
        out["fisherrf_cur_iter"] = self.controller.cur_iter
        return out


def create_renderer(spec):
    return Renderer(spec)
