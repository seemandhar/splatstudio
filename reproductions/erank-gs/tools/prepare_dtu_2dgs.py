"""Convert the 2DGS DTU release (COLMAP, RGBA images with object mask in alpha) to a native
SplatStudio scene that reproduces the official 3DGS/erank-gs loader with `-r 2 --eval`:

  * cameras: sparse/0/{cameras,images}.txt (PINHOLE 1554x1162, cx = W/2, cy = H/2)
  * images: sorted by name; llffhold8 (every 8th sorted image is test) -> split_protocol
  * resolution: -r 2 -> round(1554/2) x round(1162/2) = 777 x 581, PIL resize of the RGBA
    image (PILtoTorch: Image.resize default filter = BICUBIC), then rgb *= alpha
    (Camera: original_image *= gt_alpha_mask) -> stored as RGB on black; train/eval
    background must be black (white_background = False).
  * init: sparse/0/points3D.txt (COLMAP points + colors).
Source data: 2DGS Google Drive folder 1SJFgt8qhQomHX55Q4xSvYE2C6-8tFll9 / dtu.tar.gz.
"""
import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image


def qvec2rotmat(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * w * z, 2 * z * x + 2 * w * y],
        [2 * x * y + 2 * w * z, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * w * x],
        [2 * z * x - 2 * w * y, 2 * y * z + 2 * w * x, 1 - 2 * x * x - 2 * y * y]])


def read_cameras(path):
    cams = {}
    for line in path.read_text().splitlines():
        if line.startswith("#") or not line.strip():
            continue
        p = line.split()
        assert p[1] == "PINHOLE", p[1]
        cams[int(p[0])] = (int(p[2]), int(p[3]), *map(float, p[4:8]))
    return cams


def read_images(path):
    lines = [ln for ln in path.read_text().splitlines() if not ln.startswith("#")]
    out = []
    for i in range(0, len(lines), 2):
        p = lines[i].split()
        if not p:
            continue
        out.append({"q": np.array(list(map(float, p[1:5]))), "t": np.array(list(map(float, p[5:8]))),
                    "cam": int(p[8]), "name": p[9]})
    return out


def read_points(path):
    xyz, rgb = [], []
    for line in path.read_text().splitlines():
        if line.startswith("#") or not line.strip():
            continue
        p = line.split()
        xyz.append(list(map(float, p[1:4])))
        rgb.append(list(map(float, p[4:7])))
    return np.array(xyz, np.float32), np.array(rgb, np.float32) / 255.0


def convert(src, dst, r=2):
    cams = read_cameras(src / "sparse/0/cameras.txt")
    imgs = sorted(read_images(src / "sparse/0/images.txt"), key=lambda e: Path(e["name"]).stem)
    if dst.exists():
        shutil.rmtree(dst)
    (dst / "images").mkdir(parents=True)
    frames = []
    for e in imgs:
        W, H, fx, fy, cx, cy = cams[e["cam"]]
        w, h = round(W / r), round(H / r)
        with Image.open(src / "images" / e["name"]) as im:
            assert im.size == (W, H) and im.mode == "RGBA", (im.size, im.mode)
            a = np.asarray(im.resize((w, h)), dtype=np.float32) / 255.0  # PILtoTorch
        rgb = a[..., :3] * a[..., 3:4]
        fname = f"images/{Path(e['name']).stem}.png"
        Image.fromarray(np.clip(np.round(rgb * 255), 0, 255).astype(np.uint8), "RGB").save(dst / fname)
        R = qvec2rotmat(e["q"])
        w2c = np.eye(4)
        w2c[:3, :3], w2c[:3, 3] = R, e["t"]
        c2w = np.linalg.inv(w2c)
        c2w[:3, 1:3] *= -1  # OpenCV -> OpenGL (loader flips back)
        # official 3DGS uses FoV only (principal point assumed centred): focal * w / W
        frames.append({"file_path": fname, "transform_matrix": c2w.tolist(),
                       "fl_x": fx * w / W, "fl_y": fy * h / H, "cx": w / 2, "cy": h / 2,
                       "w": w, "h": h})
    meta = {"camera_model": "PINHOLE", "split_protocol": "llffhold8", "frames": frames,
            "dtu_scan": int(src.name.replace("scan", "")),
            "_doc": "2DGS DTU release, -r 2 (777x581), RGB*alpha on black, llffhold8 "
                    "(official 3DGS --eval); built by reproductions/erank-gs/tools/prepare_dtu_2dgs.py"}
    (dst / "transforms.json").write_text(json.dumps(meta, indent=1))
    xyz, rgb = read_points(src / "sparse/0/points3D.txt")
    np.savez(dst / "points.npz", points=xyz, colors=np.clip(rgb, 0, 1))
    names = [f["file_path"] for f in frames]
    return {"scan": src.name, "images": len(frames), "test": sorted(names)[::8],
            "size": [frames[0]["w"], frames[0]["h"]], "points": len(xyz)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--scans", nargs="+", default=["scan24", "scan37", "scan122"])
    a = ap.parse_args()
    for s in a.scans:
        print(json.dumps(convert(a.raw / s, a.out / s)), flush=True)
