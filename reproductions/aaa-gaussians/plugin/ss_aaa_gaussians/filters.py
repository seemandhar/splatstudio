"""Pure tensor functions of AAA-Gaussians (arXiv 2504.12811) matching the official code.

- ``train_filter_3d``: ``utils/mip_filter.py:mip_filter_3d`` of DerThomy/AAA-Gaussians @ dc5b3f2
  (Mip-Splatting's sampling-rate bound, k = 0.3; Gaussians seen by no camera get 0, unlike
  Mip-Splatting which uses the max distance).
- ``perpendicular_scaling`` / ``volume_scaling``: paper Eq. (11) vs Eq. (8) amplitude factors
  (``consistent_common.cuh:compute_gauss2screen``).
"""

import math

import torch


@torch.no_grad()
def train_filter_3d(means, cameras, kernel=0.3, near=0.2, margin=0.15):
    """Per-Gaussian filter std sqrt(k) * z_min / f_max over training cameras ([N]).

    cameras: dicts with OpenCV world-to-camera ``viewmat`` [4,4], pixel ``K`` [3,3], ``width``,
    ``height``. Official projection uses principal point (W/2, H/2) and focal_x for both axes'
    max; we use K (cx = W/2 is enforced by the renderer)."""
    cameras = list(cameras)
    if not cameras:
        raise ValueError("train_filter_3d needs at least one training camera")
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
    distance[~valid_points] = 0  # official: unseen Gaussians get no filter
    return distance / focal_length * math.sqrt(kernel)


def _dilated(scales, variance):
    s2 = scales * scales
    return s2, s2 + variance[..., None]


def perpendicular_scaling(scales, rotmats, view_dirs, variance):
    """Eq. (11): sqrt(sum_i d'_i^2 prod_{j!=i} s_j^2 / sum_i d'_i^2 prod_{j!=i} s^_j),
    d' = R^T d (columns of R are the Gaussian axes in world space)."""
    s2, sd = _dilated(scales, variance)
    dp = torch.einsum("nji,nj->ni", rotmats, view_dirs)
    r = dp * dp

    def pair(a):
        return torch.stack([a[:, 1] * a[:, 2], a[:, 0] * a[:, 2], a[:, 0] * a[:, 1]], -1)

    return torch.sqrt((r * pair(s2)).sum(-1) / (r * pair(sd)).sum(-1))


def volume_scaling(scales, variance):
    """Eq. (8) (Mip-Splatting): sqrt(prod s_i^2 / prod (s_i^2 + k/v^2))."""
    s2, sd = _dilated(scales, variance)
    return torch.sqrt(s2.prod(-1) / sd.prod(-1))


def view_filter_variance(z_view, focal, kernel=0.3, filter_3d=None):
    """Eq. (12) as coded: max(filter_3d^2, k z^2 / f^2), f = max(fx, fy)."""
    v = kernel * (z_view / focal) ** 2
    if filter_3d is not None:
        v = torch.maximum(filter_3d ** 2, v)
    return v
