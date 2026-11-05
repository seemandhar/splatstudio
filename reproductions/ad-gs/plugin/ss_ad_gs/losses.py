"""AD-GS losses: edge-aware depth smoothness (Eq. 7) and two-field co-training (Eq. 6, 9)."""

import torch
import torch.nn.functional as F
from splatstudio.losses import ssim

from .renderer import rasterize
from .schedule import Schedule, schedule_config


def photometric(image, target, lambda_ssim=0.2):
    # Eq. 5: (1 - l) L1 + l (1 - SSIM)  (official utils/loss_utils.py:loss_photometric)
    return (1 - lambda_ssim) * F.l1_loss(image, target) + lambda_ssim * (1 - ssim(image, target))


def edge_aware_smoothness(depth, image):
    """Eq. 7 first term in the official form (train.py:75-92). depth [H,W,1], image [H,W,3]."""
    dx = depth[:, 1:] - depth[:, :-1]
    dy = depth[1:] - depth[:-1]
    wx = torch.exp(-(image[:, 1:] - image[:, :-1]).abs().mean(dim=-1, keepdim=True))
    wy = torch.exp(-(image[1:] - image[:-1]).abs().mean(dim=-1, keepdim=True))
    return (wx * dx.abs()).mean() + (wy * dy.abs()).mean()


def depth_regularizer(depth, image, weight, range_weight):
    # train.py:236,244: w * (L_smooth + lambda_r * (-(d_max - d_min)))
    range_loss = -(depth.max() - depth.min())
    return weight * (edge_aware_smoothness(depth, image) + range_weight * range_loss)


def zero_like_graph(outputs):
    return outputs["rgb"].sum() * 0.0


def create_depth_smoothness(spec):
    cfg = dict(spec.config)
    schedule = Schedule.from_config(schedule_config(cfg))
    weight = float(cfg.get("weight", 0.01))  # omega_1 / --depth_smooth_weight
    range_weight = float(cfg.get("range_weight", 0.001))  # lambda_r

    def loss(params, outputs, target):
        if not schedule.regularize(outputs["step"]):
            return zero_like_graph(outputs)
        # Edge weights from the ground-truth training image (imagex = original_image).
        return depth_regularizer(outputs["depth"], target, weight, range_weight)

    return loss


def create_cotraining(spec):
    cfg = dict(spec.config)
    schedule = Schedule.from_config(schedule_config(cfg))
    weight = float(cfg.get("depth_weight", 0.01))
    range_weight = float(cfg.get("range_weight", 0.001))
    pseudo_weight = float(cfg.get("pseudo_weight", 1.0))  # lambda_3
    start_pseudo = int(cfg.get("start_sample_pseudo", 500))
    lambda_ssim = float(cfg.get("lambda_ssim", 0.2))

    def loss(params, outputs, target):
        co = outputs.get("adgs")
        if co is None:
            raise RuntimeError("adgs-cotrain needs the adgs-density component")
        if co.partner is None:
            return zero_like_graph(outputs)
        step, degree, bg = outputs["step"], outputs["sh_degree"], outputs["background"]
        g2 = co.partner.params
        # G2 on the same training view and background as G1 (train.py:221-228).
        r2 = rasterize(g2, outputs["camera"], degree, bg)
        r2["info"]["means2d"].retain_grad()
        regularize = schedule.regularize(step)
        loss2 = photometric(r2["rgb"], target, lambda_ssim)
        if regularize:
            loss2 = loss2 + depth_regularizer(r2["depth"], target, weight, range_weight)
        g1_term = zero_like_graph(outputs)
        if regularize and step >= start_pseudo:
            camera = co.next_pseudo_camera()
            u1 = rasterize(params, camera, degree, bg)["rgb"]
            u2 = rasterize(g2, camera, degree, bg)["rgb"]
            # Eq. 6 with the other field's render detached (train.py:270-273), / (N-1) = 1.
            g1_term = pseudo_weight * photometric(u1, u2.detach(), lambda_ssim)
            loss2 = loss2 + pseudo_weight * photometric(u2, u1.detach(), lambda_ssim)
        loss2.backward()
        co.partner_info = r2["info"]
        co.partner_loss = float(loss2.detach())
        return g1_term

    return loss
