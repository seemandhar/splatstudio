"""Initialisation and Mip-Splatting helpers of WildGaussians (official method.py):
3D smoothing filter (`compute_3D_filter`, `get_gaussians`), filter-aware `reset_opacity`, sky
points (`get_uniform_points_on_sphere_fibonacci`, `get_sky_points`), 3-NN initial scales.

Filter helpers are adapted from reproductions/mip-splatting (log-space evaluation of the same
quantities to avoid float32 underflow of prod(s^2)); the 3D filter here uses the WildGaussians
projection x / z * fx + W / 2 (principal point ignored, as in the official code).
"""

import math

import torch


@torch.no_grad()
def compute_filter_3d(means, cameras, coef=0.2, near=0.2, margin=0.15):
    """Official compute_3D_filter: min depth over cameras in which the point is in front
    (z > 0.2) and inside the screen expanded by 15 %, divided by the largest fx, * sqrt(0.2)."""
    xyz = means.detach().float()
    distance = torch.full((len(xyz),), 100000.0, device=xyz.device)
    valid_points = torch.zeros(len(xyz), dtype=torch.bool, device=xyz.device)
    focal = 0.0
    for cam in cameras:
        vm = cam["viewmat"].to(xyz.device, torch.float32)
        K = cam["K"].to(xyz.device, torch.float32)
        width, height = float(cam["width"]), float(cam["height"])
        xc = xyz @ vm[:3, :3].T + vm[:3, 3]
        valid_depth = xc[:, 2] > near
        z = xc[:, 2].clamp(min=0.001)
        x = xc[:, 0] / z * K[0, 0] + width / 2.0
        y = xc[:, 1] / z * K[1, 1] + height / 2.0
        in_screen = ((x >= -margin * width) & (x <= width * (1 + margin))
                     & (y >= -margin * height) & (y <= (1 + margin) * height))
        valid = valid_depth & in_screen
        distance[valid] = torch.minimum(distance[valid], z[valid])
        valid_points |= valid
        focal = max(focal, float(K[0, 0]))
    if valid_points.any():
        distance[~valid_points] = distance[valid_points].max()
    return distance / focal * math.sqrt(coef)


def _log_filtered_sq(log_scales, filter_3d):
    return torch.logaddexp(2 * log_scales, (2 * torch.log(filter_3d.detach()))[:, None]
                           .expand_as(log_scales))


def log_filter_coefficient(log_scales, filter_3d):
    """log sqrt(prod s^2 / prod(s^2 + f^2))."""
    return log_scales.sum(dim=1) - 0.5 * _log_filtered_sq(log_scales, filter_3d).sum(dim=1)


def filtered_scales_opacities(log_scales, opacity_logits, filter_3d):
    """get_gaussians: scales = sqrt(s^2 + f^2), opacity = sigmoid(o) * sqrt(det1/det2)."""
    lsq = _log_filtered_sq(log_scales, filter_3d)
    coef = torch.exp(log_scales.sum(dim=1) - 0.5 * lsq.sum(dim=1))
    return torch.exp(0.5 * lsq), opacity_logits.sigmoid() * coef


def reset_logits(log_scales, opacity_logits, filter_3d, value=0.01, eps=1e-6):
    """Official GaussianModel.reset_opacity (method.py:1249-1266), including its quirk: the
    filtered opacity sigmoid(o) * coef is clamped to `value`, then divided by
    coef' = sqrt(prod(s_f^2) / prod(s_f^2 + f^2)) computed from the ALREADY filtered scales
    s_f = sqrt(s^2 + f^2) (the filter is added a second time), i.e.
    new = logit(min(sigmoid(o) * coef, value) / coef'). Evaluated in log space (no float32
    underflow of prod(s^2)); the result is clamped to [eps, 1 - eps] before the logit (the
    official code would give NaN/inf there)."""
    lsf2 = _log_filtered_sq(log_scales, filter_3d)  # log(s^2 + f^2) = log s_f^2
    log_f2 = (2 * torch.log(filter_3d.detach()))[:, None].expand_as(log_scales)
    lsff2 = torch.logaddexp(lsf2, log_f2)  # log(s_f^2 + f^2) = log(s^2 + 2 f^2)
    log_coef = log_filter_coefficient(log_scales, filter_3d)
    log_coef2 = 0.5 * (lsf2.sum(dim=1) - lsff2.sum(dim=1))
    filtered = opacity_logits.sigmoid() * torch.exp(log_coef)
    new = torch.minimum(filtered, torch.full_like(filtered, value)) * torch.exp(-log_coef2)
    return torch.logit(new.clamp(eps, 1 - eps))


def fibonacci_sphere(n, dtype=torch.float32):
    """get_uniform_points_on_sphere_fibonacci (arXiv 0912.4540)."""
    phi = math.pi * (3.0 - math.sqrt(5.0))
    half = (n - 1) / 2
    i = torch.linspace(-half, half, n, dtype=dtype)
    lat = torch.arcsin(2.0 * i / (2 * half + 1))
    lon = phi * i
    return torch.stack([torch.cos(lon) * torch.cos(lat), torch.sin(lon) * torch.cos(lat),
                        torch.sin(lat)], -1)


@torch.no_grad()
def sky_points(n, points, cameras):
    """get_sky_points: sphere at 10 * q97(|x - mean|) around the SfM mean; a point is kept if
    some training camera's `camera_project` gives v < 2/3 H. Official quirk kept: points with
    camera z <= eps project to (cx, cy) (always kept) and u / v >= 0 are not checked."""
    sph = fibonacci_sphere(n).to(points.device)
    mean = points.mean(0, keepdim=True)
    radius = torch.quantile(torch.linalg.vector_norm(points - mean, dim=-1), 0.97) * 10
    sph = sph * radius + mean
    keep = torch.zeros(len(sph), dtype=torch.bool, device=sph.device)
    eps = torch.finfo(sph.dtype).eps
    for cam in cameras:
        vm = cam["viewmat"].to(sph.device, sph.dtype)
        K = cam["K"].to(sph.device, sph.dtype)
        rest = ~keep
        uvw = sph[rest] @ vm[:3, :3].T + vm[:3, 3]
        z = uvw[:, 2:]
        uv = torch.where(z > eps, uvw[:, :2] / z, torch.zeros_like(uvw[:, :2]))
        v = K[1, 1] * uv[:, 1] + K[1, 2]
        ok = ~torch.isnan(uv).any(-1) & (v < 2 / 3 * float(cam["height"]))
        keep[rest] = ok
    return sph[keep], float(radius)


def knn_log_scales(points, k=3, budget=2.5e8):
    """log sqrt(mean squared distance to the 3 nearest other points) (distCUDA2, exact KNN)."""
    n = len(points)
    chunk = max(1, int(budget // max(n, 1)))
    out = torch.empty(n, device=points.device)
    for s in range(0, n, chunk):
        d2 = torch.cdist(points[s:s + chunk], points).square()
        d2[torch.arange(d2.shape[0]), torch.arange(s, s + d2.shape[0])] = float("inf")
        out[s:s + chunk] = d2.topk(k, dim=1, largest=False).values.mean(dim=1)
    return out.clamp_min(1e-7).sqrt().log()[:, None].expand(-1, 3).contiguous()
