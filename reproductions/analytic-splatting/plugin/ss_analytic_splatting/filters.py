"""Mip-Splatting 3D-filter bookkeeping that the official Analytic-Splatting code keeps even with
`--filter3d` off: its `reset_opacity` (scene/gaussian_model.py:431) clamps the *filtered*
opacity, i.e. sets o <- min(o, 0.01 / coef) with coef = sqrt(|S^2| / |S^2 + f^2|).

`compute_3D_filter` (gaussian_model.py:188) is copied from Mip-Splatting; the functions below
are the same as the native Mip-Splatting port (reproductions/mip-splatting, log-space coef to
avoid the official float32 0/0).
"""

import math

import torch


@torch.no_grad()
def compute_filter_3d(means, cameras, coef=0.2, near=0.2, margin=0.15):
    """Per-Gaussian 3D filter std: min visible depth / max focal_x * sqrt(coef).

    cameras: iterable of dicts with OpenCV world-to-camera `viewmat` [4,4], pixel `K` [3,3],
    `width`, `height`. Returns [N] on `means.device`.
    """
    cameras = list(cameras)
    if not cameras:
        raise ValueError("compute_filter_3d needs at least one training camera")
    xyz = means.detach().float()
    distance = torch.full((len(xyz),), 100000.0, device=xyz.device)
    valid_points = torch.zeros(len(xyz), dtype=torch.bool, device=xyz.device)
    focal_length = 0.0
    for camera in cameras:
        viewmat = camera["viewmat"].to(xyz.device, torch.float32)
        K = camera["K"].to(xyz.device, torch.float32)
        width, height = float(camera["width"]), float(camera["height"])
        xyz_cam = xyz @ viewmat[:3, :3].T + viewmat[:3, 3]
        valid_depth = xyz_cam[:, 2] > near
        z = xyz_cam[:, 2].clamp(min=0.001)
        x = xyz_cam[:, 0] / z * K[0, 0] + K[0, 2]
        y = xyz_cam[:, 1] / z * K[1, 1] + K[1, 2]
        in_screen = ((x >= -margin * width) & (x <= (1 + margin) * width)
                     & (y >= -margin * height) & (y <= (1 + margin) * height))
        valid = valid_depth & in_screen
        distance[valid] = torch.minimum(distance[valid], z[valid])
        valid_points |= valid
        focal_length = max(focal_length, float(K[0, 0]))
    if valid_points.any():
        distance[~valid_points] = distance[valid_points].max()
    return distance / focal_length * math.sqrt(coef)


def log_filter_coefficient(log_scales, filter_3d):
    """log sqrt(|S^2| / |S^2 + f^2 I|) = sum(log s) - 0.5 * sum(log(s^2 + f^2))."""
    log_f2 = 2 * torch.log(filter_3d.detach())[:, None]
    log_filtered_sq = torch.logaddexp(2 * log_scales, log_f2.expand_as(log_scales))
    return log_scales.sum(dim=1) - 0.5 * log_filtered_sq.sum(dim=1)


def reset_logits(log_scales, opacity_logits, filter_3d, value=0.01, eps=1e-6):
    """Official reset_opacity: logit(min(sigmoid(o) * coef, value) / coef)
    = logit(min(sigmoid(o), value / coef)), evaluated without 0/0."""
    ceiling = value * torch.exp(-log_filter_coefficient(log_scales, filter_3d))
    unfiltered = torch.minimum(opacity_logits.sigmoid(), ceiling)
    return torch.logit(unfiltered.clamp(eps, 1 - eps))
