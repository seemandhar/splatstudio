"""Nerfstudio transforms.json input, explicit splits, and a pinhole camera contract."""

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image


def load_dataset(root, downscale=1):
    root = Path(root).resolve()
    if downscale < 1:
        raise ValueError("downscale must be positive")
    source = (root / "transforms.json").read_bytes()
    meta = json.loads(source)
    frames = meta["frames"]
    if len(frames) < 3:
        raise ValueError(
            "At least three frames are needed for disjoint train/validation/test splits"
        )
    filenames = [f["file_path"] for f in frames]
    if len(set(filenames)) != len(filenames):
        raise ValueError("Duplicate frame filenames")
    explicit = any(k in meta for k in ("train_filenames", "val_filenames", "test_filenames"))
    if explicit:
        if not all(k in meta for k in ("train_filenames", "val_filenames", "test_filenames")):
            raise ValueError(
                "Explicit splits require train_filenames, val_filenames, and test_filenames"
            )
        splits = {
            "train": meta["train_filenames"],
            "validation": meta["val_filenames"],
            "test": meta["test_filenames"],
        }
        flat = [v for group in splits.values() for v in group]
        if len(flat) != len(set(flat)) or set(flat) != set(filenames):
            raise ValueError("Splits must be disjoint and cover exactly the dataset frames")
    else:
        splits = {"train": [], "validation": [], "test": []}
        for i, name in enumerate(filenames):
            split = (
                ("test" if i == 0 else "validation" if i == 1 else "train")
                if len(frames) < 8
                else ("test" if i % 8 == 0 else "validation" if i % 8 == 4 else "train")
            )
            splits[split].append(name)
    if any(not group for group in splits.values()):
        raise ValueError("Train, validation, and test splits must each contain at least one image")
    digest = hashlib.sha256(source)
    loaded = {}
    for frame in frames:
        cfg = {**meta, **frame}
        if cfg.get("camera_model", "OPENCV") not in {"OPENCV", "PINHOLE", "SIMPLE_PINHOLE"}:
            raise ValueError("The v0.1 loader requires undistorted pinhole cameras")
        if any(abs(float(cfg.get(k, 0))) > 1e-12 for k in ("k1", "k2", "k3", "k4", "p1", "p2")):
            raise ValueError(
                "Nonzero lens distortion: undistort images and intrinsics before training"
            )
        path = (root / frame["file_path"]).resolve()
        if not path.is_relative_to(root):
            raise ValueError("Frame paths must remain inside the dataset directory")
        digest.update(path.read_bytes())
        with Image.open(path) as image:
            width, height = image.size
            if int(cfg.get("w", width)) != width or int(cfg.get("h", height)) != height:
                raise ValueError(f"Image/intrinsics resolution mismatch: {path.name}")
            w, h = max(1, width // downscale), max(1, height // downscale)
            if min(w, h) < 11:
                raise ValueError("Downscaled images must be at least 11 x 11")
            rgba = (
                np.asarray(
                    image.convert("RGBA").resize((w, h), Image.Resampling.LANCZOS), dtype=np.float32
                )
                / 255
            )
        rgb = rgba[..., :3] * rgba[..., 3:] + 1 - rgba[..., 3:]
        fx = (
            float(cfg["fl_x"])
            if "fl_x" in cfg
            else 0.5 * width / np.tan(0.5 * float(cfg["camera_angle_x"]))
        )
        fy = float(cfg.get("fl_y", fx))
        K = torch.tensor(
            [
                [fx * w / width, 0, float(cfg.get("cx", width / 2)) * w / width],
                [0, fy * h / height, float(cfg.get("cy", height / 2)) * h / height],
                [0, 0, 1],
            ],
            dtype=torch.float32,
        )
        c2w = torch.tensor(frame["transform_matrix"], dtype=torch.float32)
        if c2w.shape != (4, 4) or not torch.isfinite(c2w).all():
            raise ValueError("Camera transforms must be finite 4x4 matrices")
        c2w[:3, 1:3] *= -1  # Nerfstudio/OpenGL camera convention -> OpenCV.
        loaded[frame["file_path"]] = {
            "image": torch.from_numpy(rgb.copy()),
            "K": K,
            "viewmat": torch.linalg.inv(c2w),
            "width": w,
            "height": h,
        }
    dataset_id = digest.hexdigest()
    protocol = {
        "dataset_sha256": dataset_id,
        "splits": splits,
        "downscale": downscale,
        "background": "white",
        "camera": "opencv_pinhole_from_opengl",
        "metrics": {
            "psnr": "mean per-image RGB PSNR; MSE floor 1e-12; range [0,1]",
            "ssim": "Gaussian 11x11 sigma=1.5 valid population covariance",
        },
    }
    protocol_id = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
    return {
        "root": root,
        "meta": meta,
        "frames": loaded,
        "splits": splits,
        "protocol": protocol,
        "protocol_id": protocol_id,
        "dataset_id": dataset_id,
    }


def initial_points(dataset, count, seed, device):
    rng = np.random.default_rng(seed)
    root = dataset["root"]
    cloud = root / "points.npz"
    if cloud.exists():
        with np.load(cloud, allow_pickle=False) as data:
            points, colors = data["points"], data["colors"]
    elif dataset["meta"].get("ply_file_path"):
        from plyfile import PlyData

        vertices = PlyData.read(root / dataset["meta"]["ply_file_path"])["vertex"]
        points = np.stack([vertices[k] for k in ("x", "y", "z")], axis=-1)
        colors = np.stack([vertices[k] for k in ("red", "green", "blue")], axis=-1) / 255
    else:
        centers = np.stack(
            [np.linalg.inv(f["viewmat"].numpy())[:3, 3] for f in dataset["frames"].values()]
        )
        center = centers.mean(axis=0)
        scale = max(float(np.linalg.norm(centers - center, axis=1).mean()) * 0.3, 0.1)
        points = rng.uniform(-scale, scale, size=(count, 3)) + center
        colors = rng.uniform(0.1, 0.9, size=(count, 3))
    if len(points) < 2 or points.shape != colors.shape or points.shape[1:] != (3,):
        raise ValueError("Point initialization requires matching [N,3] positions and RGB colors")
    if (
        not np.isfinite(points).all()
        or not np.isfinite(colors).all()
        or np.any((colors < 0) | (colors > 1))
    ):
        raise ValueError("Point positions must be finite and RGB colors must be in [0,1]")
    if len(points) > count:
        ids = rng.choice(len(points), count, replace=False)
        points, colors = points[ids], colors[ids]
    return (
        torch.tensor(points.copy(), dtype=torch.float32, device=device),
        torch.tensor(colors.copy(), dtype=torch.float32, device=device),
    )


def on_device(frame, device):
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in frame.items()}
