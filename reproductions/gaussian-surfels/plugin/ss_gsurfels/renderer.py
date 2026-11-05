"""Gaussian Surfels renderer component (official CUDA rasterizer, raster.py) with the training-
time parts of the official train.py loop that live next to rendering:

* coarse-to-fine resolution: training renders 1..2016 at 1/4, 2017..5040 at 1/2, then full
  resolution (train.py:73-80 switches the scale at it 2002 / 5002, but the camera stack is only
  refilled at the next epoch boundary: k*N+1 = 2017 / 5041 for both N = 42 and N = 56 DTU
  training views). Training renders are counted by this component (grad-enabled calls).
* random background per training step (ModelParams.random_background = True); evaluation
  renders use the official test background (black, white_background = False).
* per-view sidecar data (mask, Omnidata normal, PIL-downscaled image) for the losses.

Outputs: rgb/depth/alpha HWC for the runtime, plus ``gs`` with the official CHW tensors.
"""

import hashlib

import torch

from . import sidecar
from .raster import render, scaled_camera

CAPABILITIES = {"render.rgb", "render.depth.expected_z", "render.normal.camera", "render.alpha",
                "gradient.gsurfels"}
DEFAULTS = {
    "resolution_schedule": [[2016, 4], [5040, 2]],  # (last training render, downscale factor)
    "random_background": True,
    "eval_background": [0.0, 0.0, 0.0],
}


def view_key(viewmat):
    return hashlib.sha1(viewmat.detach().float().cpu().numpy().round(5).tobytes()).hexdigest()


class Renderer:
    capabilities = CAPABILITIES

    def __init__(self, spec):
        unknown = set(spec.config) - set(DEFAULTS)
        if unknown:
            raise ValueError(f"Unknown Gaussian Surfels renderer options: {sorted(unknown)}")
        self.cfg = {**DEFAULTS, **spec.config}
        self.capabilities = set(CAPABILITIES)
        self.train_calls = 0
        self._table = False

    def table(self):
        if self._table is False:
            self._table = sidecar.table()
        return self._table

    def factor(self, call):
        for last, f in self.cfg["resolution_schedule"]:
            if call <= last:
                return int(f)
        return 1

    def render(self, params, camera, sh_degree=3):
        dev = params["means"].device
        training = torch.is_grad_enabled() and params["means"].requires_grad
        tab = self.table()
        idx = tab.index(camera["viewmat"]) if (tab is not None and training) else None
        if training:
            self.train_calls += 1
            factor = self.factor(self.train_calls) if idx is not None else 1
            bg = (torch.rand(3, device=dev) if self.cfg["random_background"]
                  else torch.zeros(3, device=dev))
        else:
            factor = 1
            bg = torch.tensor(self.cfg["eval_background"], dtype=torch.float32, device=dev)
        cam = scaled_camera(camera, factor)
        out = render(params, cam, sh_degree, bg)
        view = tab.view(idx, factor, dev) if idx is not None else None
        return {
            "rgb": out["color"].permute(1, 2, 0),
            "depth": out["depth"].permute(1, 2, 0),
            "alpha": out["opac"].permute(1, 2, 0),
            "info": {"means2d": out["means2d"], "radii": out["radii"]},
            "camera": camera,
            "view_id": idx if idx is not None else view_key(camera["viewmat"]),
            "gs": {"color": out["color"], "normal": out["normal"], "depth": out["depth"],
                   "opac": out["opac"], "camera": cam, "background": bg, "factor": factor,
                   "view": view, "train_call": self.train_calls if training else None},
        }


def create_renderer(spec):
    return Renderer(spec)
