"""Plugin-side distorted-camera loading and reference projections (no SplatStudio core change).

The core v0.1 loader rejects lens distortion, so `ss-train` only feeds pinhole cameras. For
distorted captures (paper §5: ScanNet++ fisheye, Waymo rolling shutter) this module turns a
Nerfstudio transforms.json with `camera_model` OPENCV / OPENCV_FISHEYE and k1..k4/p1/p2 into
camera dicts the 3DGUT renderer accepts (keys documented in renderer.camera_kwargs).
"""

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image


def distortion_keys(cfg):
    model = cfg.get("camera_model", "OPENCV")
    f = {k: float(cfg.get(k, 0.0)) for k in ("k1", "k2", "k3", "k4", "k5", "k6", "p1", "p2")}
    if model == "OPENCV_FISHEYE":
        return {"camera_model": "fisheye", "radial_coeffs": [f["k1"], f["k2"], f["k3"], f["k4"]]}
    if model in ("OPENCV", "FULL_OPENCV", "PINHOLE", "SIMPLE_PINHOLE"):
        radial = [f["k1"], f["k2"], f["k3"], f["k4"], f["k5"], f["k6"]]
        out = {"camera_model": "pinhole"}
        if any(radial) or f["p1"] or f["p2"]:
            out["radial_coeffs"] = radial
            out["tangential_coeffs"] = [f["p1"], f["p2"]]
        return out
    raise ValueError(f"Unsupported camera_model {model!r}")


def load_distorted_frames(root, downscale=1):
    """Return {file_path: camera dict with image} for a distorted transforms.json."""
    root = Path(root).resolve()
    meta = json.loads((root / "transforms.json").read_text())
    frames = {}
    for frame in meta["frames"]:
        cfg = {**meta, **frame}
        path = (root / frame["file_path"]).resolve()
        with Image.open(path) as image:
            width, height = image.size
            w, h = width // downscale, height // downscale
            rgb = np.asarray(image.convert("RGB").resize((w, h), Image.Resampling.LANCZOS),
                             dtype=np.float32) / 255
        sx, sy = w / width, h / height
        K = torch.tensor([[float(cfg["fl_x"]) * sx, 0, float(cfg["cx"]) * sx],
                          [0, float(cfg["fl_y"]) * sy, float(cfg["cy"]) * sy], [0, 0, 1]])
        c2w = torch.tensor(frame["transform_matrix"], dtype=torch.float32)
        c2w[:3, 1:3] *= -1  # OpenGL -> OpenCV
        frames[frame["file_path"]] = {"image": torch.from_numpy(rgb.copy()), "K": K,
                                      "viewmat": torch.linalg.inv(c2w), "width": w, "height": h,
                                      **distortion_keys(cfg)}
    return frames


def project_points(points_cam, K, camera_model="pinhole", radial=None, tangential=None):
    """Independent float64 camera models (OpenCV pinhole k1..k6,p1,p2; OpenCV fisheye k1..k4)."""
    x, y, z = points_cam.unbind(-1)
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    if camera_model == "fisheye":
        r = torch.sqrt(x * x + y * y)
        theta = torch.atan2(r, z)
        k = radial if radial is not None else [0.0] * 4
        t2 = theta * theta
        theta_d = theta * (1 + k[0] * t2 + k[1] * t2**2 + k[2] * t2**3 + k[3] * t2**4)
        scale = torch.where(r > 1e-12, theta_d / r.clamp_min(1e-12), 1 / z)
        u, v = x * scale, y * scale
    else:
        u, v = x / z, y / z
        if radial is not None:
            k = list(radial) + [0.0] * (6 - len(radial))
            p = tangential if tangential is not None else [0.0, 0.0]
            r2 = u * u + v * v
            radial_f = (1 + k[0] * r2 + k[1] * r2**2 + k[2] * r2**3) / (
                1 + k[3] * r2 + k[4] * r2**2 + k[5] * r2**3
            )
            du = 2 * p[0] * u * v + p[1] * (r2 + 2 * u * u)
            dv = p[0] * (r2 + 2 * v * v) + 2 * p[1] * u * v
            u, v = u * radial_f + du, v * radial_f + dv
    return torch.stack((fx * u + cx, fy * v + cy), dim=-1)
