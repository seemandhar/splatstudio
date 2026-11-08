"""Pure Pixel-GS formulas (CPU-testable)."""

import torch


def depth_factor(depths, gamma_depth, radius):
    """Scaled gradient field, paper Eq. 10: f = clip((z_cam / (gamma * radius))^2, 0, 1).
    Official: diff_gaussian_rasterization/__init__.py:144 `min(1, (depth/depth_threshold)**2)`
    with depth_threshold = 0.37 * cameras_extent (train.py:91)."""
    return torch.clamp((depths / (gamma_depth * radius)) ** 2, min=0.0, max=1.0)


def view_contribution(grad_norm, pixels, depths, *, pixel_aware, depth_scaling, gamma_depth,
                      radius):
    """Per-view (numerator, denominator) increments of Eq. 11 for visible Gaussians.

    numerator   = m * f * ||dL/dmu_ndc||   (Eq. 7 with Eq. 10)
    denominator = m                        (official gaussian_model.py:406-407)
    With pixel_aware=False, m := 1 (3DGS view count, Eq. 4); with depth_scaling=False, f := 1.
    """
    w = pixels if pixel_aware else torch.ones_like(grad_norm)
    n = grad_norm
    if depth_scaling:
        n = n * depth_factor(depths, gamma_depth, radius)
    return n * w, w
