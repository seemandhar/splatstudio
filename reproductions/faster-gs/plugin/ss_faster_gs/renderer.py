"""Faster-GS renderer: the vendored official CUDA rasterizer (raster.py), black background
(official config), near/far 0.2/10000.

outputs["info"]["fastergs"] is the persistent densification-info holder (official
`Gaussians._densification_info`): the renderer (re)allocates a zero `[2, N]` buffer whenever the
population size changes and passes it to the backward while `holder.enabled`; the density reads,
zeroes, re-orders and finally disables it. `n_instances` = Gaussian/tile instances of the frame.
"""

import torch

from .raster import FAR, NEAR, rasterize

CAPABILITIES = {"render.rgb", "densification.fastergs"}
DEFAULTS = {"background": [0.0, 0.0, 0.0], "near": NEAR, "far": FAR}


class FasterGSInfo:
    """Persistent [2, N] buffer: row 0 visibility count, row 1 sum of NDC ||dL/d mu_2D||."""

    def __init__(self):
        self.buffer = None
        self.enabled = True

    def get(self, n, device):
        if self.buffer is None or self.buffer.shape[1] != n or self.buffer.device != device:
            self.buffer = torch.zeros((2, n), dtype=torch.float32, device=device)
        return self.buffer


class Renderer:
    def __init__(self, spec):
        unknown = set(spec.config) - set(DEFAULTS)
        if unknown:
            raise ValueError(f"Unknown Faster-GS renderer options: {sorted(unknown)}")
        cfg = {**DEFAULTS, **spec.config}
        self.background = [float(v) for v in cfg["background"]]
        self.near, self.far = float(cfg["near"]), float(cfg["far"])
        self.capabilities = set(CAPABILITIES)
        self.densification = FasterGSInfo()
        self._bg = {}

    def background_tensor(self, device):
        if device not in self._bg:
            self._bg[device] = torch.tensor(self.background, dtype=torch.float32, device=device)
        return self._bg[device]

    def render(self, params, camera, sh_degree=3):
        means = params["means"]
        training = torch.is_grad_enabled() and means.requires_grad
        info_buffer = None
        if training and self.densification.enabled:
            info_buffer = self.densification.get(len(means), means.device)
        stats = {}
        image = rasterize(params, camera, sh_degree, self.background_tensor(means.device),
                          info_buffer, self.near, self.far, stats)
        return {
            "rgb": image.permute(1, 2, 0),
            "info": {"fastergs": self.densification, **stats,
                     "width": int(camera["width"]), "height": int(camera["height"])},
        }


def create_renderer(spec):
    return Renderer(spec)
