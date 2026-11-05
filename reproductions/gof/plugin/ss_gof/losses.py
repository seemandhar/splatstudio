"""GOF objective (train.py of GOF @5245b20).

photometric  (1 - 0.2) L1 + 0.2 (1 - SSIM), official utils/loss_utils.ssim (11x11 Gaussian
             window sigma 1.5, same padding).
gof-reg      lambda_distortion * mean(distortion map) [from step 15000]
           + lambda_depth_normal * mean(1 - n_render . N_depth) [from step 15000]
             with n_render the renormalised blended intersection-plane normal (world) and
             N_depth the 2DGS finite-difference normal of the median depth (paper L_d, L_n).
"""

import torch
import torch.nn.functional as F

PHOTO_DEFAULTS = {"lambda_dssim": 0.2}
REG_DEFAULTS = {"lambda_distortion": 100.0, "lambda_depth_normal": 0.05,
                "distortion_from_iter": 15000, "depth_normal_from_iter": 15000}


def _window(channels, device, dtype, size=11, sigma=1.5):
    x = torch.arange(size, device=device, dtype=dtype) - size // 2
    g = torch.exp(-(x ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    return (g[:, None] @ g[None, :])[None, None].expand(channels, 1, size, size).contiguous()


def ssim_same(img, target):
    """Official _ssim with padding = window_size // 2, mean over all pixels. HWC inputs."""
    x, y = (t.permute(2, 0, 1)[None] for t in (img, target))
    c = x.shape[1]
    w = _window(c, x.device, x.dtype)

    def conv(t):
        return F.conv2d(t, w, padding=5, groups=c)

    mu1, mu2 = conv(x), conv(y)
    s1, s2, s12 = conv(x * x) - mu1 ** 2, conv(y * y) - mu2 ** 2, conv(x * y) - mu1 * mu2
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    m = ((2 * mu1 * mu2 + c1) * (2 * s12 + c2)) / ((mu1 ** 2 + mu2 ** 2 + c1) * (s1 + s2 + c2))
    return m.mean()


def create_photometric(spec):
    cfg = {**PHOTO_DEFAULTS, **spec.config}
    lam = float(cfg["lambda_dssim"])

    def loss(params, outputs, target):
        rgb = outputs["rgb"]
        return (1 - lam) * (rgb - target).abs().mean() + lam * (1 - ssim_same(rgb, target))

    return loss


def regularizer_terms(outputs):
    distortion = outputs["distortion"].mean()
    normal_error = 1 - (outputs["normal"] * outputs["surface_normal"]).sum(dim=-1)
    return distortion, normal_error.mean()


def create_regularizer(spec):
    unknown = set(spec.config) - set(REG_DEFAULTS)
    if unknown:
        raise ValueError(f"Unknown GOF regularizer options: {sorted(unknown)}")
    cfg = {**REG_DEFAULTS, **spec.config}

    def loss(params, outputs, target):
        step = outputs["step"]
        distortion, normal = regularizer_terms(outputs)
        lam_d = cfg["lambda_distortion"] if step >= cfg["distortion_from_iter"] else 0.0
        lam_n = cfg["lambda_depth_normal"] if step >= cfg["depth_normal_from_iter"] else 0.0
        return distortion * lam_d + normal * lam_n

    return loss
