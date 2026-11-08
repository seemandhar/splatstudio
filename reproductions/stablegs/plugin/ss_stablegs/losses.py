"""StableGS objective (official train_pair.py:126-176; paper Eq. 2-4 and Sec. 4.1).

pair-color         (1−λ)(L1_0 + L1_1) + λ(2 − SSIM_0 − SSIM_x) on the appearance path,
                   x = 0 in the official code (fused_ssim(image0, gt0) is added twice,
                   train_pair.py:128; "official"), x = 1 with ssim_views "both".
geometric-color    after `start` (15000): λ_s (L1(C^s_0) + L1(C^s_1))  (--single_loss, λ_s = 1;
                   paper Eq. 3 second term).
depth-normal       after `start`: λ (0.4 mean E_exp + 0.6 mean E_med) with
                   E = Σ_views (1 − n·N(D)) on the geometric path (RaDe-GS depth-normal loss,
                   --lambda_depth_normal 0.05; not described in the paper).
depth-consistency  after `start`, only for pairs in P: λ_consis · L_consis (Eq. 4).
                   "official" (train_pair.py:157-176): L = mean relu(D_1∘π_{0→1} − D_0) +
                   mean relu(D_0∘π_{1→0} − D_1) — the sampled other-view depth is compared with
                   the own-view camera z without a frame change, one-sided, mean over all pixels;
                   "paper": two-sided L1 between the warped z (z of the view-i point in view j) and
                   D_j sampled at its projection, mean over validly projected pixels.
Losses not active yet return a graph-attached exact zero.
"""

import torch
import torch.nn.functional as F
from ss_taming_3dgs.core import ssim_same

from .raster import depth_double_to_normal


def _zero(outputs):
    return outputs["rgb"].sum() * 0.0


def _views(outputs):
    pair = outputs.get("pair")
    if pair is None:
        raise RuntimeError("StableGS losses need the pair batch (stablegs-density not active)")
    return pair


def create_pair_color(spec):
    cfg = {"lambda_dssim": 0.2, "ssim_views": "official", **spec.config}
    if cfg["ssim_views"] not in ("official", "both"):
        raise ValueError("ssim_views must be 'official' or 'both'")
    lam = float(cfg["lambda_dssim"])

    def loss(params, outputs, target):
        views = _views(outputs)["views"]
        imgs = [v["app"]["rgb"] for v in views]
        gts = [v["target"] for v in views]
        l1 = sum((im - gt).abs().mean() for im, gt in zip(imgs, gts))
        second = 0 if cfg["ssim_views"] == "official" else 1
        ssim = ssim_same(imgs[0], gts[0]) + ssim_same(imgs[second], gts[second])
        return (1 - lam) * l1 + lam * (2.0 - ssim)

    return loss


def create_geometric_color(spec):
    cfg = {"lambda_s": 1.0, **spec.config}

    def loss(params, outputs, target):
        pair = _views(outputs)
        if not pair["geo"]:
            return _zero(outputs)
        return float(cfg["lambda_s"]) * sum(
            (v["geo"]["rgb"] - v["target"]).abs().mean() for v in pair["views"])

    return loss


def depth_normal_term(views, depth_ratio=0.6):
    err = 0.0
    for v in views:
        g = v["geo"]
        surf = depth_double_to_normal(v["camera"], g["depth"][..., 0], g["median_depth"][..., 0])
        err = err + (1 - (g["normal"][None] * surf).sum(dim=-1))  # [2,H,W]
    return (1 - depth_ratio) * err[0].mean() + depth_ratio * err[1].mean()


def create_depth_normal(spec):
    cfg = {"lambda_depth_normal": 0.05, "depth_ratio": 0.6, **spec.config}

    def loss(params, outputs, target):
        pair = _views(outputs)
        if not pair["geo"]:
            return _zero(outputs)
        return float(cfg["lambda_depth_normal"]) * depth_normal_term(pair["views"],
                                                                     float(cfg["depth_ratio"]))

    return loss


def _camera_points(depth, camera):
    """train_pair.py:161-166: X = (gx·D, gy·D, D) with gx = linspace(−(W−1)/2, (W−1)/2, W)/fx."""
    H, W = depth.shape
    fx, fy = float(camera["K"][0, 0]), float(camera["K"][1, 1])
    dev = depth.device
    gy, gx = torch.meshgrid(torch.linspace(-(H - 1) / 2, (H - 1) / 2, H, device=dev),
                            torch.linspace(-(W - 1) / 2, (W - 1) / 2, W, device=dev),
                            indexing="ij")
    return torch.stack((gx / fx * depth, gy / fy * depth, depth)).reshape(3, -1)


def _project(points_cam, src, dst):
    """Camera-frame points of `src` -> camera frame of `dst` (OpenCV world-to-camera viewmats)."""
    c2w = torch.linalg.inv(src["viewmat"])
    world = c2w[:3, :3] @ points_cam + c2w[:3, 3:]
    return dst["viewmat"][:3, :3] @ world + dst["viewmat"][:3, 3:]


Z_EPS = 1e-6


def safe_z(z):
    """Numerical guard (not in the official code): float32 cancellation can make the transformed
    z exactly 0, then x/z = inf and grid_sample's backward returns NaN (observed on truck at the
    first geometric step). Points with |z| < 1e-6 project > 1e5 image widths away, so replacing
    their z by 1e-6 (gradient 0 through `where`) leaves loss and gradients of all in-view points
    unchanged."""
    return torch.where(z.abs() < Z_EPS, torch.full_like(z, Z_EPS), z)


def _sample(depth, pts, camera):
    """grid_sample(depth, normalized projection) with official normalization 2u/W (centred u)."""
    H, W = depth.shape
    fx, fy = float(camera["K"][0, 0]), float(camera["K"][1, 1])
    z = safe_z(pts[2])
    u = pts[0] / z * fx * 2 / W
    v = pts[1] / z * fy * 2 / H
    grid = torch.stack((u, v), dim=-1)[None, None]
    return F.grid_sample(depth[None, None], grid, align_corners=False)[0, 0, 0]


def consistency_official(d0, d1, cam0, cam1):
    """train_pair.py:157-176 verbatim semantics. d0, d1: [H,W] geometric expected depths."""
    X0, X1 = _camera_points(d0, cam0), _camera_points(d1, cam1)
    X1_0 = _project(X1, cam1, cam0)  # view-1 points in camera 0
    X0_1 = _project(X0, cam0, cam1)
    depth0_1 = _sample(d0, X1_0, cam0).view(d1.shape)
    depth1_0 = _sample(d1, X0_1, cam1).view(d0.shape)
    return (depth1_0 - d0).clamp(0).mean() + (depth0_1 - d1).clamp(0).mean()


def consistency_paper(d0, d1, cam0, cam1):
    """Eq. 4: ‖W_{0→1}(D_0) − D_1‖_1 + ‖W_{1→0}(D_1) − D_0‖_1 over validly projected pixels:
    the warped depth is the z of the view-i point in camera j, compared with D_j at its
    projection (inside the pixel-centre hull, in front of the camera, D_j > 0)."""
    total = 0.0
    for da, db, ca, cb in ((d0, d1, cam0, cam1), (d1, d0, cam1, cam0)):
        Xa = _project(_camera_points(da, ca), ca, cb)
        H, W = db.shape
        fx, fy = float(cb["K"][0, 0]), float(cb["K"][1, 1])
        z = safe_z(Xa[2])
        u = Xa[0] / z * fx * 2 / W
        v = Xa[1] / z * fy * 2 / H
        sampled = _sample(db, Xa, cb)
        # inside the pixel-centre hull, so bilinear sampling never blends with the zero padding
        valid = ((z > 1e-6) & (u.abs() <= (W - 1) / W) & (v.abs() <= (H - 1) / H)
                 & (da.reshape(-1) > 0) & (sampled > 0))
        if bool(valid.any()):
            total = total + (z - sampled)[valid].abs().mean()
    return total


def create_depth_consistency(spec):
    cfg = {"lambda_consis": 0.05, "form": "official", **spec.config}
    if cfg["form"] not in ("official", "paper"):
        raise ValueError("form must be 'official' or 'paper'")
    fn = consistency_official if cfg["form"] == "official" else consistency_paper

    def loss(params, outputs, target):
        pair = _views(outputs)
        if not (pair["geo"] and pair["in_P"]):
            return _zero(outputs)
        v0, v1 = pair["views"]
        value = fn(v0["geo"]["depth"][..., 0], v1["geo"]["depth"][..., 0], v0["camera"],
                   v1["camera"])
        if not torch.is_tensor(value):
            return _zero(outputs)
        return float(cfg["lambda_consis"]) * value

    return loss
