"""Explicit OpenCV undistortion for Nerfstudio-format datasets."""

import hashlib
import json
import shutil
from pathlib import Path

import cv2
import numpy as np


def prepare_dataset(data, output, downscale=1):
    source, destination = Path(data).resolve(), Path(output).resolve()
    if downscale < 1 or destination == source or destination.is_relative_to(source):
        raise ValueError("Use a positive downscale and a separate output directory")
    if destination.exists() and any(destination.iterdir()):
        raise ValueError("Prepared-data output directory must be empty")
    raw = (source / "transforms.json").read_bytes()
    meta = json.loads(raw)
    prepared = {k: v for k, v in meta.items() if k != "frames"}
    prepared["frames"] = []
    (destination / "images").mkdir(parents=True, exist_ok=True)
    name_map = {}
    digest = hashlib.sha256(raw)
    calibration_keys = {
        "fl_x",
        "fl_y",
        "cx",
        "cy",
        "w",
        "h",
        "camera_model",
        "camera_angle_x",
        "k1",
        "k2",
        "k3",
        "k4",
        "p1",
        "p2",
    }
    for index, frame in enumerate(meta["frames"]):
        cfg = {**meta, **frame}
        if cfg.get("camera_model", "OPENCV") not in {"OPENCV", "PINHOLE", "SIMPLE_PINHOLE"}:
            raise ValueError("prepare-data supports pinhole and OpenCV radial/tangential cameras")
        if abs(float(cfg.get("k4", 0))) > 1e-12:
            raise ValueError("Nonzero k4 is unsupported; use the original camera calibration tool")
        path = (source / frame["file_path"]).resolve()
        if not path.is_relative_to(source):
            raise ValueError("Frame path escapes the dataset")
        digest.update(path.read_bytes())
        pixels = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if pixels is None:
            raise ValueError(f"Cannot read image {frame['file_path']}")
        height, width = pixels.shape[:2]
        if int(cfg.get("w", width)) != width or int(cfg.get("h", height)) != height:
            raise ValueError("Calibration resolution does not match image resolution")
        fx = (
            float(cfg["fl_x"]) if "fl_x" in cfg else width / (2 * np.tan(cfg["camera_angle_x"] / 2))
        )
        K = np.array(
            [
                [fx, 0, cfg.get("cx", width / 2)],
                [0, cfg.get("fl_y", fx), cfg.get("cy", height / 2)],
                [0, 0, 1],
            ],
            float,
        )
        distortion = np.array([cfg.get(k, 0) for k in ("k1", "k2", "p1", "p2", "k3")], float)
        # alpha=0 avoids black invalid borders. Keep the same camera pose and no crop.
        new_K, _ = cv2.getOptimalNewCameraMatrix(K, distortion, (width, height), 0)
        if np.any(distortion):
            pixels = cv2.undistort(pixels, K, distortion, None, new_K)
        else:
            new_K = K.copy()
        w, h = width // downscale, height // downscale
        if min(w, h) < 11:
            raise ValueError("Prepared images must be at least 11 x 11")
        if downscale != 1:
            pixels = cv2.resize(pixels, (w, h), interpolation=cv2.INTER_AREA)
        name = f"images/{index:06d}.png"
        if not cv2.imwrite(str(destination / name), pixels):
            raise OSError(f"Could not write {name}")
        name_map[frame["file_path"]] = name
        entry = {k: v for k, v in frame.items() if k not in calibration_keys}
        entry.update(
            file_path=name,
            camera_model="PINHOLE",
            w=w,
            h=h,
            fl_x=float(new_K[0, 0] * w / width),
            fl_y=float(new_K[1, 1] * h / height),
            cx=float(new_K[0, 2] * w / width),
            cy=float(new_K[1, 2] * h / height),
        )
        # Auxiliary masks/depth need their own interpolation rules and are not copied.
        for key in ("mask_path", "depth_file_path"):
            if key in entry:
                raise ValueError(
                    "Datasets with auxiliary masks/depth require a task-specific converter"
                )
        prepared["frames"].append(entry)
    for key in calibration_keys:
        prepared.pop(key, None)
    prepared["camera_model"] = "PINHOLE"
    for key in ("train_filenames", "val_filenames", "test_filenames"):
        if key in prepared:
            prepared[key] = [name_map[n] for n in prepared[key]]
    if (source / "points.npz").exists():
        shutil.copy2(source / "points.npz", destination / "points.npz")
    if meta.get("ply_file_path"):
        cloud = (source / meta["ply_file_path"]).resolve()
        if not cloud.is_relative_to(source):
            raise ValueError("Point cloud path escapes the dataset")
        shutil.copy2(cloud, destination / "points.ply")
        prepared["ply_file_path"] = "points.ply"
    prepared["splatstudio_preprocessing"] = {
        "source_sha256": digest.hexdigest(),
        "downscale": downscale,
        "opencv": cv2.__version__,
        "undistortion": "optimal matrix alpha=0, no crop",
        "resize": "INTER_AREA",
        "poses": "preserved in input OpenGL convention",
    }
    (destination / "transforms.json").write_text(json.dumps(prepared, indent=2) + "\n")
    return {
        "frames": len(name_map),
        "output": str(destination),
        "preprocessing": prepared["splatstudio_preprocessing"],
    }
