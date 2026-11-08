"""gsplat renderer with LiteGS's training-time conventions.

- SH degree schedule: min(epoch // 5, 3) (official trainer.py) while training; the runtime's degree
  at evaluation (no grad; final degree 3 on both).
- The rendered image is clamped to [0, 1] before the loss (official render/__init__.py `img.clamp`).
- Exposes the pre-clamp image (`rgb_raw`, retained by the density controller for dL/dC) and the
  camera, which the opacity-gradient statistics kernel needs. Image formation is gsplat's.
"""

from dataclasses import replace

import torch
from splatstudio.backends.gsplat import Renderer

from . import schedule

META_CAPABILITY = "litegs.raster_meta"


class LiteRenderer(Renderer):
    def __init__(self, spec):
        config = dict(spec.config)
        self.sh_interval = int(config.pop("sh_epoch_interval", 5))
        self.clamp_training_rgb = bool(config.pop("clamp_training_rgb", True))
        super().__init__(replace(spec, config=config))
        self.capabilities = set(self.capabilities) | {META_CAPABILITY}
        self.last_degree = None

    def render(self, params, camera, sh_degree=3):
        n_train = schedule.SHARED["n_train"]
        if torch.is_grad_enabled() and n_train is not None:
            sh_degree = schedule.sh_degree(schedule.SHARED["step"] + 1, n_train, self.sh_interval,
                                           3)
        self.last_degree = sh_degree
        outputs = super().render(params, camera, sh_degree)
        raw = outputs["rgb"]
        outputs["rgb_raw"] = raw
        outputs["sh_degree"] = sh_degree
        outputs["camera"] = {k: camera[k] for k in ("K", "viewmat", "width", "height")}
        if self.clamp_training_rgb and torch.is_grad_enabled():
            outputs["rgb"] = raw.clamp(0, 1)
        return outputs


def create_renderer(spec):
    return LiteRenderer(spec)
