"""LightGaussian losses.

* `photometric`: 3DGS L1 + 0.2 D-SSIM against the ground truth in every stage except SH
  distillation (`distill_train.py` never uses the ground truth image).
* `lightgaussian-distill`: during distillation L1 + 0.2 D-SSIM between the student render and
  the frozen SH-3 teacher render at the same (pseudo or training) camera (`distill_train.py`;
  the paper's Eq. 5 writes an L2 loss, the code uses this one).
Inactive terms return an attached zero so the runtime's loss checks pass.
"""

from types import SimpleNamespace

import torch
import torch.nn.functional as F
from splatstudio.losses import ssim

from .shared import STATE


def _dssim_l1(rgb, target, weight):
    return (1 - weight) * F.l1_loss(rgb, target) + weight * (1 - ssim(rgb, target))


def _zero(outputs):
    return outputs["rgb"].sum() * 0.0


def create_photometric(spec):
    weight = float(spec.config.get("ssim_weight", 0.2))

    def loss(params, outputs, target):
        if STATE.active and STATE.stage() == "distill":
            return _zero(outputs)
        return _dssim_l1(outputs["rgb"], target, weight)

    return loss


def teacher_render(renderer, camera, sh_degree=3):
    with torch.no_grad():
        return renderer.render(STATE.teacher, camera, sh_degree)["rgb"].detach()


def create_distill(spec):
    from .renderer import Renderer

    weight = float(spec.config.get("ssim_weight", 0.2))
    # Teacher renderer: the same gsplat renderer configuration as the student's.
    plain = Renderer(SimpleNamespace(config=dict(spec.config.get("renderer", {}))))

    def loss(params, outputs, target):
        if not (STATE.active and STATE.stage() == "distill"):
            return _zero(outputs)
        if STATE.teacher is None:
            raise RuntimeError("distillation stage without a teacher model")
        teacher = teacher_render(plain, outputs["lg_camera"], 3)
        return _dssim_l1(outputs["rgb"], teacher, weight)

    return loss
