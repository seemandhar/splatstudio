"""LightGaussian renderer: the unchanged gsplat renderer plus distillation pseudo views (C2).

`distill_train.py` + `utils/pose_utils.py:gaussian_poses(cam, 0, std_dev_translation=0.05,
std_dev_rotation=0)`: on distillation iterations with `iteration % 3 != 0` the camera centre is
moved by N(0, 0.05^2) per world axis (rotation unchanged) and both student and teacher are
rendered there. The camera actually used is returned as `outputs["lg_camera"]`.
"""

import torch
from splatstudio.backends.gsplat import Renderer

from . import schedule
from .shared import STATE


def shift_camera(camera, offset):
    """getWorld2View2(R, T, translate): C2W[:3, 3] += offset, i.e. t' = t - R offset."""
    viewmat = camera["viewmat"].clone()
    viewmat[:3, 3] = viewmat[:3, 3] - viewmat[:3, :3] @ offset.to(viewmat)
    return {**camera, "viewmat": viewmat}


class LightGaussianRenderer:
    def __init__(self, spec):
        self.base = Renderer(spec)
        self.capabilities = self.base.capabilities

    def render(self, params, camera, sh_degree=3):
        cam = camera
        if STATE.active and schedule.pseudo_view(STATE.next_step, STATE.cfg):
            std = float(STATE.cfg["pseudo_translation_std"])
            offset = torch.randn(3, generator=STATE.generator) * std
            STATE.last_offset = offset
            cam = shift_camera(camera, offset)
        out = self.base.render(params, cam, sh_degree)
        out["lg_camera"] = cam
        return out


def create_renderer(spec):
    return LightGaussianRenderer(spec)
