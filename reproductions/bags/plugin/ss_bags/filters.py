"""Mip-Splatting 3D filter math (BAGS builds on Mip-Splatting). Copied from the SPLATIFY
mip-splatting port (reproductions/mip-splatting/plugin/ss_mip_splatting/filters.py), which
matches autonomousvision/mip-splatting scene/gaussian_model.py; BAGS uses the same code
(source/code/scene/gaussian_model.py compute_3D_filter / get_*_with_3D_filter / reset_opacity).
"""
import math

import torch


@torch.no_grad()
def compute_filter_3d(means, cameras, coef=0.2, near=0.2, margin=0.15):
    """Per-Gaussian 3D smoothing filter std (paper Eq. 8 + code `compute_3D_filter`).

    nu_k = max_n 1_n(p_k) f_n / d_n (Eq. 8). The official code implements it as
    min visible depth / max focal_x over cameras, and the filter std as sqrt(coef) * d / f,
    i.e. filter variance coef / nu^2 (paper's s/nu with s = 0.2, squared in the code).

    cameras: iterable of dicts with OpenCV world-to-camera `viewmat` [4,4], pixel `K` [3,3],
    integer `width`, `height`. Returns a [N] tensor on `means.device`.
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
        # Official code projects with principal point W/2, H/2; we use K (cx = W/2 here).
        x = xyz_cam[:, 0] / z * K[0, 0] + K[0, 2]
        y = xyz_cam[:, 1] / z * K[1, 1] + K[1, 2]
        in_screen = (
            (x >= -margin * width)
            & (x <= (1 + margin) * width)
            & (y >= -margin * height)
            & (y <= (1 + margin) * height)
        )
        valid = valid_depth & in_screen
        distance[valid] = torch.minimum(distance[valid], z[valid])
        valid_points |= valid
        focal_length = max(focal_length, float(K[0, 0]))
    if valid_points.any():
        distance[~valid_points] = distance[valid_points].max()
    return distance / focal_length * math.sqrt(coef)


def _log_filtered_sq(log_scales, filter_3d):
    """log(s^2 + f^2) per axis, computed without forming s^2 (no float32 underflow)."""
    log_f2 = 2 * torch.log(filter_3d.detach())[:, None]  # -inf for f = 0 is handled
    return torch.logaddexp(2 * log_scales, log_f2.expand_as(log_scales))


def log_filter_coefficient(log_scales, filter_3d):
    """log sqrt(|Sigma| / |Sigma + f^2 I|) = sum(log s) - 0.5 * sum(log(s^2 + f^2)).

    The official code forms prod(s^2) / prod(s^2 + f^2) directly; in float32 prod(s^2)
    underflows to 0 for small Gaussians (s ~ 1e-8), giving 0/0 = NaN in reset_opacity.
    Log space is the same quantity without the underflow (documented deviation).
    """
    return log_scales.sum(dim=1) - 0.5 * _log_filtered_sq(log_scales, filter_3d).sum(dim=1)


def filtered_scales_opacities(log_scales, opacity_logits, filter_3d):
    """Apply the 3D smoothing filter (Eq. 9): Sigma + f^2 I and sqrt(|Sigma|/|Sigma+f^2 I|).

    Code: get_scaling_with_3D_filter / get_opacity_with_3D_filter. Because Sigma = R S^2 R^T,
    Sigma + f^2 I = R (S^2 + f^2) R^T, so the filter acts axis-wise on the scales.
    `filter_3d` is a constant (no gradient), as in the official code.
    """
    log_filtered_sq = _log_filtered_sq(log_scales, filter_3d)
    coef = torch.exp(log_scales.sum(dim=1) - 0.5 * log_filtered_sq.sum(dim=1))
    return torch.exp(0.5 * log_filtered_sq), opacity_logits.sigmoid() * coef


def filter_coefficient(log_scales, filter_3d):
    return torch.exp(log_filter_coefficient(log_scales, filter_3d))


def reset_logits(log_scales, opacity_logits, filter_3d, value=0.01, eps=1e-6):
    """Official `reset_opacity`: clamp the *filtered* opacity to `value`, then undo the filter.

    Official: logit(min(sigmoid(o) * coef, value) / coef). Evaluated here as the algebraically
    identical logit(min(sigmoid(o), value / coef)) with value / coef = value * exp(-log coef)
    (finite or +inf, never 0/0), and the unfiltered opacity clamped to [eps, 1 - eps] before
    the logit so saturated sigmoid(o) = 1.0 cannot give +inf. Deviation from the official code,
    which has the same latent NaN/inf issue.
    """
    ceiling = value * torch.exp(-log_filter_coefficient(log_scales, filter_3d))
    unfiltered = torch.minimum(opacity_logits.sigmoid(), ceiling)
    return torch.logit(unfiltered.clamp(eps, 1 - eps))


