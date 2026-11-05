"""PGSR objective (official train.py @de24f1a; paper Eq. numbers refer to arXiv 2406.06521 v2).

pgsr-rgb          (1-l) L1(I~, I) + l (1 - SSIM(I_r, I)), I~ = exp(a_i) I_r + b_i when exposure
                  compensation is on, step > 1000 and 1 - SSIM < 0.5, else I_r   (Sec. 3.3)
pgsr-flatten      lambda_1 * mean over visible Gaussians of min(s1, s2, s3)       (L_s, Sec. 3.1)
pgsr-single-view  lambda_2 * mean((1 - grad I)^2 * |N_d - N|_1), step > 7000      (L_svgeom)
pgsr-mv-geo       lambda_4 * mean_{valid} w * phi, step > 7000                    (L_mvgeom)
pgsr-mv-ncc       lambda_3 * mean_{kept} w * (1 - NCC), step > 7000               (L_mvrgb)
"""

import random

import torch
import torch.nn.functional as F

from . import geometry as G
from .link import LINK

RGB_DEFAULTS = {"lambda_dssim": 0.2, "exposure_compensation": False, "exposure_from_iter": 1000,
                "exposure_ssim_gate": 0.5, "exposure_lr": 0.001}
FLATTEN_DEFAULTS = {"scale_loss_weight": 100.0}
SINGLE_DEFAULTS = {"single_view_weight": 0.015, "single_view_weight_from_iter": 7000,
                   "image_weight": True}
MV_SHARED = {"multi_view_weight_from_iter": 7000, "multi_view_patch_size": 3,
             "multi_view_sample_num": 102400, "multi_view_pixel_noise_th": 1.0,
             "use_geo_occ_aware": True, "ncc_scale": 1.0}
GEO_DEFAULTS = {"multi_view_geo_weight": 0.03, **MV_SHARED}
NCC_DEFAULTS = {"multi_view_ncc_weight": 0.15, **MV_SHARED}


def _cfg(spec, defaults, label):
    unknown = set(spec.config) - set(defaults)
    if unknown:
        raise ValueError(f"Unknown {label} options: {sorted(unknown)}")
    return {**defaults, **spec.config}


def _zero(params):
    return params["means"].sum() * 0.0


def _window(channels, device, dtype, size=11, sigma=1.5):
    x = torch.arange(size, device=device, dtype=dtype) - size // 2
    g = torch.exp(-(x ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    return (g[:, None] @ g[None, :])[None, None].expand(channels, 1, size, size).contiguous()


def ssim_same(img, target):
    """Official utils/loss_utils.ssim (11x11 Gaussian sigma 1.5, same padding). HWC inputs."""
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


class Exposure:
    """scene/app_model.AppModel: appear_ab [num_images, 2] zeros, Adam(lr, betas 0.9/0.99)."""

    def __init__(self, lr):
        self.lr = lr
        self.ab = None
        self.optimizer = None

    def coefficients(self, index, count, device):
        if self.ab is None:
            self.ab = torch.nn.Parameter(torch.zeros(count, 2, device=device))
            self.optimizer = torch.optim.Adam([self.ab], lr=self.lr, betas=(0.9, 0.99))
            LINK.extra_optimizers.append(self.optimizer)
        return self.ab[index]


def create_rgb(spec):
    c = _cfg(spec, RGB_DEFAULTS, "pgsr-rgb")
    lam = float(c["lambda_dssim"])
    exposure = Exposure(float(c["exposure_lr"])) if c["exposure_compensation"] else None

    def loss(params, outputs, target):
        image = outputs["rgb"]
        ssim_loss = 1.0 - ssim_same(image, target)
        if (exposure is not None and outputs["step"] > c["exposure_from_iter"]
                and float(ssim_loss) < c["exposure_ssim_gate"]):
            i = LINK.view_index(outputs["camera"])
            ab = exposure.coefficients(i, len(LINK.cameras), image.device)
            l1 = (torch.exp(ab[0]) * image + ab[1] - target).abs().mean()  # Eq. I^a
        else:
            l1 = (image - target).abs().mean()
        return (1.0 - lam) * l1 + lam * ssim_loss

    loss.exposure = exposure
    return loss


def flatten_loss(params, radii, weight):
    vis = radii > 0
    if int(vis.sum()) == 0:
        return _zero(params)
    scale = params["scales"][vis].exp()
    return weight * torch.sort(scale, dim=-1)[0][..., 0].mean()  # L_s = ||min(s1,s2,s3)||_1


def create_flatten(spec):
    c = _cfg(spec, FLATTEN_DEFAULTS, "pgsr-flatten")

    def loss(params, outputs, target):
        return flatten_loss(params, outputs["info"]["radii"], float(c["scale_loss_weight"]))

    return loss


def single_view_loss(normal, depth_normal, target, use_image_weight=True):
    """mean((1 - grad)^2 |N_d - N|_1) over all pixels (HWC maps, camera frame)."""
    diff = (depth_normal - normal).abs().sum(-1)
    if not use_image_weight:
        return diff.mean()
    w = (1.0 - G.image_grad_weight(target)).clamp(0, 1).detach() ** 2
    return (w * diff).mean()


def create_single_view(spec):
    c = _cfg(spec, SINGLE_DEFAULTS, "pgsr-single-view")

    def loss(params, outputs, target):
        if outputs["step"] <= c["single_view_weight_from_iter"]:
            return _zero(params)
        if not outputs.get("geometry"):
            raise RuntimeError("pgsr-single-view: renderer did not produce plane geometry")
        return float(c["single_view_weight"]) * single_view_loss(
            outputs["normal"], outputs["depth_normal"], target, c["image_weight"])

    return loss


def multiview_terms(params, outputs, cfg):
    """Shared neighbour render + both multi-view terms (unweighted), memoised per step."""
    memo = outputs.get("_pgsr_mv")
    if memo is not None:
        if memo["cfg"] != {k: cfg[k] for k in MV_SHARED}:
            raise ValueError("pgsr-mv-geo and pgsr-mv-ncc must share their multi-view options")
        return memo
    shared = {k: cfg[k] for k in MV_SHARED}
    memo = {"cfg": shared, "geo": None, "ncc": None, "neighbour": None, "valid": 0}
    outputs["_pgsr_mv"] = memo
    if not outputs.get("geometry"):
        raise RuntimeError("pgsr multi-view: renderer did not produce plane geometry")
    cam = outputs["camera"]
    i = LINK.view_index(cam)
    if not LINK.neighbours[i]:
        return memo  # official: nearest_cam is None -> no multi-view term
    j = random.choice(LINK.neighbours[i])  # random.sample(nearest_id, 1)
    cam_n = LINK.cameras[j]
    memo["neighbour"] = j
    near = LINK.renderer.render(params, cam_n, outputs.get("sh_degree", 3), render_geo=True)
    depth_r = outputs["depth"][..., 0]
    noise, in_view = G.forward_backward_error(cam, depth_r, cam_n, near["depth"][..., 0])
    weights, d_mask = G.occlusion_weights(noise, in_view, cfg["multi_view_pixel_noise_th"],
                                          cfg["use_geo_occ_aware"])
    count = int(d_mask.sum())
    memo["valid"] = count
    if count == 0:
        return memo
    memo["geo"] = (weights * noise)[d_mask].mean()
    with torch.no_grad():
        valid = torch.arange(d_mask.shape[0], device=d_mask.device)[d_mask]
        if count > cfg["multi_view_sample_num"]:
            pick = torch.randperm(count, device=valid.device)[: cfg["multi_view_sample_num"]]
            valid = valid[pick]
        H, W = depth_r.shape
        iy, ix = torch.meshgrid(torch.arange(H, device=valid.device),
                                torch.arange(W, device=valid.device), indexing="ij")
        pixels = torch.stack([ix, iy], -1).reshape(-1, 2).float()[valid]
        w = weights[valid]
    normal = outputs["normal"].reshape(-1, 3)[valid]
    distance = outputs["distance"].reshape(-1)[valid]
    ncc, keep = G.ncc_terms(cam, cam_n, LINK.gray[i], LINK.gray[j], normal, distance, pixels, w,
                            cfg["multi_view_patch_size"], cfg["ncc_scale"])
    if int(keep.sum()) > 0:
        memo["ncc"] = ncc.mean()
    return memo


def _create_mv(spec, defaults, label, term, weight_key):
    c = _cfg(spec, defaults, label)
    if float(c["ncc_scale"]) != 1.0:
        raise ValueError("ncc_scale != 1 (NCC on a different-resolution grey image) is not "
                         "supported by this port")

    def loss(params, outputs, target):
        if outputs["step"] <= c["multi_view_weight_from_iter"]:
            return _zero(params)
        value = multiview_terms(params, outputs, c)[term]
        return _zero(params) if value is None else float(c[weight_key]) * value

    return loss


def create_mv_geo(spec):
    return _create_mv(spec, GEO_DEFAULTS, "pgsr-mv-geo", "geo", "multi_view_geo_weight")


def create_mv_ncc(spec):
    return _create_mv(spec, NCC_DEFAULTS, "pgsr-mv-ncc", "ncc", "multi_view_ncc_weight")
