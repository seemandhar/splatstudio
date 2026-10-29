"""Loss components return scalar differentiable tensors."""

import torch
import torch.nn.functional as F


def ssim(image, target):
    # HWC inputs, Gaussian 11x11 window, valid convolution, population covariance.
    if min(image.shape[:2]) < 11:
        raise ValueError("SSIM requires images at least 11 x 11")
    coords = torch.arange(11, device=image.device, dtype=image.dtype) - 5
    gaussian = torch.exp(-(coords**2) / (2 * 1.5**2))
    gaussian /= gaussian.sum()
    window = (gaussian[:, None] * gaussian[None, :])[None, None].expand(3, 1, 11, 11)
    x, y = (t.permute(2, 0, 1)[None] for t in (image, target))

    def conv(t):
        return F.conv2d(t, window, groups=3)

    mx, my = conv(x), conv(y)
    vx, vy, cov = conv(x * x) - mx * mx, conv(y * y) - my * my, conv(x * y) - mx * my
    return (
        ((2 * mx * my + 0.01**2) * (2 * cov + 0.03**2))
        / ((mx * mx + my * my + 0.01**2) * (vx + vy + 0.03**2))
    ).mean()


def create_photometric(spec):
    weight = float(spec.config.get("ssim_weight", 0.2))
    if not 0 <= weight <= 1:
        raise ValueError("ssim_weight must be in [0, 1]")

    def loss(params, outputs, target):
        rgb = outputs["rgb"]
        return (1 - weight) * F.l1_loss(rgb, target) + weight * (1 - ssim(rgb, target))

    return loss


def check_loss(value, name):
    if not isinstance(value, torch.Tensor) or value.numel() != 1:
        raise ValueError(f"{name}: loss must be a scalar tensor")
    if not torch.isfinite(value).all():
        raise ValueError(f"{name}: nonfinite loss")
    if not value.requires_grad:
        raise ValueError(f"{name}: loss is detached from the computation graph")


def create_surfel_geometry(spec):
    normal_weight = float(spec.config["normal_weight"])
    distortion_weight = float(spec.config["distortion_weight"])
    normal_start = int(spec.config.get("normal_start", 7000))
    distortion_start = int(spec.config.get("distortion_start", 3000))

    def loss(params, outputs, target):
        # Both gsplat outputs use world coordinates. Alpha excludes empty pixels.
        normal = F.normalize(outputs["normal"], dim=-1, eps=1e-8)
        surface = F.normalize(outputs["surface_normal"], dim=-1, eps=1e-8)
        normal_error = ((1 - (normal * surface).sum(-1)) * outputs["alpha"][..., 0].detach()).mean()
        step = outputs["step"]
        return (
            normal_weight * (step > normal_start) * normal_error
            + distortion_weight * (step > distortion_start) * outputs["distortion"].mean()
        )

    return loss
