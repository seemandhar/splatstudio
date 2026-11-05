"""Hierarchical 3DGS renderer (vendored hierarchy rasterizer, see raster.py).

Before the hierarchy exists (h3d flags 0): official `render` of train_single.py, all leaves,
random background per training iteration (train_single.py:58), black at evaluation.
After the hierarchy is built: official `render_post`; the cut is selected per view
(`expand_to_size`) and interpolation weights computed (`get_interpolation_weights`):
  training   : target size sampled log-uniformly in [limit_min, limit_max) (train_post.py:89),
               black background (train_post.py).
  evaluation : target granularity `eval_tau` pixels converted with render_hierarchy.py:57;
               eval_tau = -0.5 gives threshold 0, i.e. every leaf with t = 1 ("leaves").
SH degree is the full degree 3 in the hierarchy phase (train_post / render_hierarchy set
active_sh_degree = sh_degree).

outputs["info"]: means2d (retained screen gradient), radii, width, height, cut_size.
"""

import random

import torch

from . import hier
from .raster import render_cut, render_leaves

CAPABILITIES = {"render.rgb", "render.antialiased_2d", "gradient.screen.signed"}
DEFAULTS = {"eval_tau": -0.5, "limit_min": 0.005, "limit_max": 0.1, "max_sh_degree": 3,
            "random_background": True}


class Renderer:
    def __init__(self, spec):
        unknown = set(spec.config) - set(DEFAULTS)
        if unknown:
            raise ValueError(f"Unknown hierarchical-3dgs renderer options: {sorted(unknown)}")
        self.cfg = {**DEFAULTS, **spec.config}
        self.capabilities = set(CAPABILITIES)
        self.cut = hier.Cut()
        self._tree = None
        self.rng = random.Random(0)

    def tree(self, params):
        blob = params[hier.FIELD]
        key = (blob.data_ptr(), blob.shape[0], blob._version)
        if self._tree is None or self._tree[0] != key:
            self._tree = (key, *hier.unpack(params))
        return self._tree[1], self._tree[2]

    def render(self, params, camera, sh_degree=3):
        dev = params["means"].device
        training = torch.is_grad_enabled()
        info = {"width": int(camera["width"]), "height": int(camera["height"])}
        if not hier.is_built(params):
            if training and self.cfg["random_background"]:
                bg = torch.rand(3, device=dev)
            else:
                bg = torch.zeros(3, device=dev)
            color, means2d, radii = render_leaves(params, camera, sh_degree, bg)
            info.update(means2d=means2d, radii=radii, cut_size=len(params["means"]))
            return {"rgb": color.permute(1, 2, 0), "info": info}
        nodes, boxes = self.tree(params)
        campos = torch.linalg.inv(camera["viewmat"].float())[:3, 3]
        if training:
            threshold = hier.sample_limit(self.cfg["limit_min"], self.cfg["limit_max"],
                                          self.rng.random())
        else:
            threshold = hier.pixel_threshold(float(self.cfg["eval_tau"]), camera)
        r, p, w, k = self.cut.select(nodes, boxes, len(params["means"]), threshold, campos)
        bg = torch.zeros(3, device=dev)
        color, means2d, radii = render_cut(params, camera, int(self.cfg["max_sh_degree"]), bg,
                                           r, p, w, k)
        info.update(means2d=means2d, radii=radii, cut_size=len(r), threshold=threshold)
        return {"rgb": color.permute(1, 2, 0), "info": info}


def create_renderer(spec):
    return Renderer(spec)
