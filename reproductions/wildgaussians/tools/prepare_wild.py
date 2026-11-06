"""Convert WildGaussians benchmark scenes (COLMAP sparse model + image lists) to the native
SplatStudio layout (transforms.json with explicit splits, points.npz).

Matches the NerfBaselines loaders the paper used:
* phototourism: IMC-2020 ``dense/`` reconstruction, NeRF-W split file ``nerfw_split.csv``
  (rows with a non-empty id; ``split`` column train/test), full-resolution images.
* onthego: HF ``jkulhanek/nerfonthego-undistorted`` scene (pre-downscaled undistorted images,
  ``train_list.txt`` / ``test_list.txt``).

COLMAP world coordinates are kept unnormalised (NerfBaselines/3DGS). Images are hard-linked
(the runtime rejects paths that resolve outside the scene directory); the validation split is a
hard-linked copy of the test split under ``images/val/`` (reported, never used for selection).

    python prepare_wild.py --kind phototourism --raw RAW_SCENE --out NATIVE_SCENE
"""

import argparse
import csv
import json
import os
import struct
from pathlib import Path

import numpy as np

CAMERA_PARAMS = {0: ("SIMPLE_PINHOLE", 3), 1: ("PINHOLE", 4), 2: ("SIMPLE_RADIAL", 4),
                 3: ("RADIAL", 5), 4: ("OPENCV", 8)}


def _read(f, fmt):
    size = struct.calcsize("<" + fmt)
    return struct.unpack("<" + fmt, f.read(size))


def read_cameras(path):
    cams = {}
    with open(path, "rb") as f:
        (n,) = _read(f, "Q")
        for _ in range(n):
            cid, model, w, h = _read(f, "iiQQ")
            name, k = CAMERA_PARAMS[model]
            params = _read(f, "d" * k)
            cams[cid] = {"model": name, "w": int(w), "h": int(h), "params": params}
    return cams


def read_images(path):
    images = {}
    with open(path, "rb") as f:
        (n,) = _read(f, "Q")
        for _ in range(n):
            iid, qw, qx, qy, qz, tx, ty, tz, cid = _read(f, "idddddddi")
            name = b""
            while True:
                c = f.read(1)
                if c == b"\x00":
                    break
                name += c
            (n2d,) = _read(f, "Q")
            f.seek(24 * n2d, os.SEEK_CUR)
            images[iid] = {"q": (qw, qx, qy, qz), "t": (tx, ty, tz), "camera_id": cid,
                           "name": name.decode()}
    return images


def read_points(path):
    xyz, rgb = [], []
    with open(path, "rb") as f:
        (n,) = _read(f, "Q")
        for _ in range(n):
            _pid, x, y, z, r, g, b, _err = _read(f, "QdddBBBd")
            (tl,) = _read(f, "Q")
            f.seek(8 * tl, os.SEEK_CUR)
            xyz.append((x, y, z))
            rgb.append((r, g, b))
    return np.asarray(xyz, dtype=np.float64), np.asarray(rgb, dtype=np.uint8)


def qvec2rotmat(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * w * z, 2 * z * x + 2 * w * y],
        [2 * x * y + 2 * w * z, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * w * x],
        [2 * z * x - 2 * w * y, 2 * y * z + 2 * w * x, 1 - 2 * x * x - 2 * y * y]])


def intrinsics(cam):
    p = cam["params"]
    if cam["model"] == "SIMPLE_PINHOLE":
        return p[0], p[0], p[1], p[2]
    if cam["model"] == "PINHOLE":
        return p[0], p[1], p[2], p[3]
    raise ValueError(f"{cam['model']}: undistorted pinhole cameras required")


def splits_for(kind, raw):
    if kind == "phototourism":
        with open(raw / "nerfw_split.csv") as f:
            rows = list(csv.reader(f, delimiter="\t"))[1:]
        train = [r[0] for r in rows if r[1] and r[2] == "train"]
        test = [r[0] for r in rows if r[1] and r[2] == "test"]
    else:
        train = (raw / "train_list.txt").read_text().split()
        test = (raw / "test_list.txt").read_text().split()
    if set(train) & set(test):
        raise ValueError("train/test lists overlap")
    return train, test


def link(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    os.link(src, dst)


def convert(kind, raw, out):
    raw, out = Path(raw), Path(out)
    cams = read_cameras(raw / "sparse" / "cameras.bin")
    imgs = {v["name"]: v for v in read_images(raw / "sparse" / "images.bin").values()}
    xyz, rgb = read_points(raw / "sparse" / "points3D.bin")
    train, test = splits_for(kind, raw)
    missing = [n for n in train + test if n not in imgs]
    if missing:
        raise ValueError(f"{len(missing)} listed images are not in the COLMAP model: {missing[:3]}")
    from PIL import Image

    frames = []

    def frame(name, rel):
        im = imgs[name]
        cam = cams[im["camera_id"]]
        fx, fy, cx, cy = intrinsics(cam)
        with Image.open(raw / "images" / name) as image:
            w, h = image.size
        if (w, h) != (cam["w"], cam["h"]):
            if kind != "onthego":
                raise ValueError(f"{name}: image {(w, h)} != camera {(cam['w'], cam['h'])}")
            # Pre-downscaled undistorted On-the-go images: NerfBaselines
            # _dataset_rescale_intrinsics multiplies (fx, cx) by w/W and (fy, cy) by h/H.
            mx, my = w / cam["w"], h / cam["h"]
            fx, fy, cx, cy = fx * mx, fy * my, cx * mx, cy * my
        w2c = np.eye(4)
        w2c[:3, :3] = qvec2rotmat(im["q"])
        w2c[:3, 3] = im["t"]
        c2w = np.linalg.inv(w2c)
        c2w[:3, 1:3] *= -1  # OpenCV -> OpenGL/Nerfstudio
        return {"file_path": rel, "transform_matrix": c2w.tolist(), "w": w, "h": h,
                "fl_x": fx, "fl_y": fy, "cx": cx, "cy": cy, "camera_model": "PINHOLE"}

    for name in train + test:
        link(raw / "images" / name, out / "images" / name)
        frames.append(frame(name, f"images/{name}"))
    for name in test:
        link(raw / "images" / name, out / "images" / "val" / name)
        frames.append(frame(name, f"images/val/{name}"))
    meta = {
        "train_filenames": [f"images/{n}" for n in train],
        "val_filenames": [f"images/val/{n}" for n in test],
        "test_filenames": [f"images/{n}" for n in test],
        "wildgaussians": {"kind": kind, "source": str(raw.resolve()),
                          "coordinates": "COLMAP world, unnormalised",
                          "evaluation_protocol": "nerfw" if kind == "phototourism" else "default"},
        "frames": frames,
    }
    (out / "transforms.json").write_text(json.dumps(meta, indent=1))
    np.savez(out / "points.npz", points=xyz.astype(np.float32),
             colors=(rgb.astype(np.float32) / 255.0))
    return {"train": len(train), "test": len(test), "points": len(xyz)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", choices=["phototourism", "onthego"], required=True)
    ap.add_argument("--raw", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    print(json.dumps(convert(a.kind, a.raw, a.out)))


if __name__ == "__main__":
    main()
