"""RaDe-GS objective (train.py of RaDe-GS @2d4bc08, lines 126-161).

photometric  (1 - 0.2) L1 + 0.2 (1 - SSIM), official utils/loss_utils.ssim (11x11 Gaussian
             window sigma 1.5, same padding).
radegs-reg   from step regularization_from_iter (15000):
             L_d = mean(edge * where(alpha > 0, distortion / alpha^2 (detached), 0))  (paper Eq. 23)
             L_n = 0.4 mean(1 - n.N(depth_exp)) + 0.6 mean(1 - n.N(depth_med))       (paper Eq. 24)
             loss = 100 L_d + 0.05 L_n                                              (paper Eq. 25)
             n = renormalised rasterized normal (camera frame), N(.) = finite-difference normal
             (depth_double_to_normal, zero border -> error 1 there), edge = exp(-max |grad I|)
             over the 4 neighbours of the GT image (zero border), as in scene/cameras.py.
"""

import torch
import torch.nn.functional as F

PHOTO_DEFAULTS = {"lambda_dssim": 0.2}
REG_DEFAULTS = {"lambda_distortion": 100.0, "lambda_depth_normal": 0.05, "depth_ratio": 0.6,
                "regularization_from_iter": 15000, "edge_weight": True}


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


@torch.no_grad()
def edge_map(target):
    """scene/cameras.py (adopted from GOF): exp(-max 4-neighbour mean-|dI|), zero-padded.
    target HWC -> [H, W]."""
    img = target.clamp(0.0, 1.0).permute(2, 0, 1)
    c = img[:, 1:-1, 1:-1]
    grads = [torch.mean(torch.abs(c - n), 0) for n in (
        img[:, 1:-1, :-2], img[:, 1:-1, 2:], img[:, :-2, 1:-1], img[:, 2:, 1:-1])]
    max_grad = torch.exp(-torch.stack(grads, dim=-1).max(dim=-1)[0])
    return F.pad(max_grad, (1, 1, 1, 1), mode="constant", value=0)


def regularizer_terms(outputs, target, depth_ratio=0.6, edge_weight=True):
    alpha = outputs["alpha"][..., 0]
    dist = outputs["distortion"][..., 0]
    dist = torch.where(alpha > 0, dist / (alpha * alpha).detach(), torch.zeros_like(dist))
    if edge_weight:
        dist = dist * edge_map(target)
    distortion = dist.mean()
    n = outputs["normal"]
    err_exp = 1 - (n * outputs["surface_normal_expected"]).sum(dim=-1)
    err_med = 1 - (n * outputs["surface_normal_median"]).sum(dim=-1)
    normal = (1 - depth_ratio) * err_exp.mean() + depth_ratio * err_med.mean()
    return distortion, normal


def create_regularizer(spec):
    unknown = set(spec.config) - set(REG_DEFAULTS)
    if unknown:
        raise ValueError(f"Unknown RaDe-GS regularizer options: {sorted(unknown)}")
    cfg = {**REG_DEFAULTS, **spec.config}

    def loss(params, outputs, target):
        if outputs["step"] < cfg["regularization_from_iter"]:
            # official: both weights 0 (no graph); keep a graph-attached exact zero
            return outputs["rgb"].sum() * 0.0
        distortion, normal = regularizer_terms(outputs, target, cfg["depth_ratio"],
                                               cfg["edge_weight"])
        return cfg["lambda_distortion"] * distortion + cfg["lambda_depth_normal"] * normal

    return loss
