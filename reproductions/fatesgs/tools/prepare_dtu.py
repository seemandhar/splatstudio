"""Convert the FatesGS authors' DTU release into SplatStudio native scenes.

Input  (authors' Google Drive, see paper.md): <raw>/DTU/set_23_24_33/scanX (train views 23,24,33),
       <raw>/DTU/set_22_25_28/scanX (views 22,25,28, used only as held-out PSNR views),
       <raw>/set_23_24_33/scanX/depth_npy/000k_pred.npy (Marigold, 581x777).
Output <out>/scanX/:
  images/train_000k.png   RGB 777x581 (alpha dropped: official use_mask=False trains on raw RGB),
                          per-channel PIL bicubic resize exactly like the official PILtoTorch(-r 2)
  images/view22.png, images/view25.png, images/view28.png  held-out views (same processing)
  images_full/train_000k.png  RGB 1554x1162 (input of the Vis-MVSNet level-1 features)
  transforms.json         explicit splits: train=3 views, val=view22, test=view25+view28
  points.npz              dense/fused.ply of set_23_24_33 (official init, eval=False)
  fatesgs.npz             sidecar: train viewmats (OpenCV w2c), mono depth, source pairs,
                          alpha masks (resized like the official gt_alpha_mask), names
  dtu_eval/               copy of DTU_EVAL/set_23_24_33/scanX (cameras.npz + 1600x1200 masks)
The COLMAP frame of the release is the scale_mat-normalised frame used by the evaluator.

usage: python prepare_dtu.py --raw ~/datasets/dtu_fatesgs/extract --out ~/datasets/dtu_fatesgs/native
"""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image
from plyfile import PlyData

SCANS = [24, 37, 40, 55, 63, 65, 69, 83, 97, 105, 106, 110, 114, 118, 122]


def qvec2rotmat(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * w * z, 2 * z * x + 2 * w * y],
        [2 * x * y + 2 * w * z, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * w * x],
        [2 * z * x - 2 * w * y, 2 * y * z + 2 * w * x, 1 - 2 * x * x - 2 * y * y]])


def read_colmap(sparse):
    cam = [ln.split() for ln in (sparse / "cameras.txt").read_text().splitlines()
           if ln and not ln.startswith("#")]
    assert len(cam) == 1 and cam[0][1] == "PINHOLE", cam
    w, h, fx, fy, cx, cy = int(cam[0][2]), int(cam[0][3]), *map(float, cam[0][4:8])
    lines = [ln for ln in (sparse / "images.txt").read_text().splitlines()
             if not ln.startswith("#")]
    poses = {}
    for ln in lines[0::2]:
        if not ln.strip():
            continue
        p = ln.split()
        R = qvec2rotmat(np.array(list(map(float, p[1:5]))))
        t = np.array(list(map(float, p[5:8])))
        w2c = np.eye(4)
        w2c[:3, :3], w2c[:3, 3] = R, t
        poses[p[9]] = w2c
    return (w, h, fx, fy, cx, cy), poses


def rgb_resized(path, size):
    """Official PILtoTorch per channel (PIL default bicubic for 'L'), uint8 like PIL."""
    im = Image.open(path)
    chans = im.split()
    rgb = [c.resize(size, Image.Resampling.BICUBIC) for c in chans[:3]]
    alpha = chans[3].resize(size, Image.Resampling.BICUBIC) if len(chans) > 3 else None
    return Image.merge("RGB", rgb), alpha


def read_pairs(path):
    lines = path.read_text().split("\n")
    n = int(lines[0])
    pairs = {}
    for i in range(1, 1 + 2 * n, 2):
        ref = int(lines[i].strip())
        toks = lines[i + 1].split()
        pairs[ref] = [int(toks[j]) for j in range(1, 1 + 2 * int(toks[0]), 2)]
    return pairs


def convert(raw, out, scan):
    a = raw / "DTU" / "set_23_24_33" / f"scan{scan}"
    b = raw / "DTU" / "set_22_25_28" / f"scan{scan}"
    dst = out / f"scan{scan}"
    if dst.exists():
        shutil.rmtree(dst)
    (dst / "images").mkdir(parents=True)
    (dst / "images_full").mkdir()
    (w, h, fx, fy, cx, cy), poses_a = read_colmap(a / "sparse" / "0")
    (wb, hb, fxb, fyb, cxb, cyb), poses_b = read_colmap(b / "sparse" / "0")
    assert (w, h) == (wb, hb) and abs(fx - fxb) < 1e-2 and abs(fy - fyb) < 1e-2
    # Check the shared frame: cameras.npz (normalised) centres == COLMAP centres.
    import cv2
    for root, poses in ((a, poses_a), (b, poses_b)):
        cams = np.load(root / "cameras.npz")
        for k in range(3):
            P = (cams[f"world_mat_{k}"] @ cams[f"scale_mat_{k}"])[:3, :4]
            c = cv2.decomposeProjectionMatrix(P)[2]
            c = (c[:3] / c[3])[:, 0]
            c2 = np.linalg.inv(poses[f"{k:04d}.png"])[:3, 3]
            assert np.abs(c - c2).max() < 1e-3, (root, k, c, c2)
    rw, rh = round(w / 2), round(h / 2)  # official -r 2: round(orig/2)
    size = (rw, rh)
    frames, train, val, test = [], [], [], []
    viewmats, monos, alphas, names = [], [], [], []
    pairs = read_pairs(a / "pair.txt")
    for k in range(3):
        name = f"images/train_{k:04d}.png"
        rgb, alpha = rgb_resized(a / "images" / f"{k:04d}.png", size)
        rgb.save(dst / name)
        Image.open(a / "images" / f"{k:04d}.png").convert("RGB").save(
            dst / "images_full" / f"train_{k:04d}.png")
        w2c = poses_a[f"{k:04d}.png"]
        frames.append((name, w2c))
        train.append(name)
        viewmats.append(w2c)
        mono = np.load(raw / "set_23_24_33" / f"scan{scan}" / "depth_npy" / f"{k:04d}_pred.npy")
        assert mono.shape == (rh, rw), mono.shape
        monos.append(mono.astype(np.float32))
        alphas.append(np.asarray(alpha, dtype=np.uint8))
        names.append(name)
    for k, view in enumerate((22, 25, 28)):
        name = f"images/view{view}.png"
        rgb_resized(b / "images" / f"{k:04d}.png", size)[0].save(dst / name)
        frames.append((name, poses_b[f"{k:04d}.png"]))
        (val if view == 22 else test).append(name)
    meta = {
        "camera_model": "PINHOLE", "w": rw, "h": rh,
        # official Camera: fx = W / (2 tan(FoVx/2)) with FoVx from the full-res focal; cx = W/2
        "fl_x": fx * rw / w, "fl_y": fy * rh / h, "cx": rw / 2.0, "cy": rh / 2.0,
        "train_filenames": train, "val_filenames": val, "test_filenames": test,
        "frames": [],
        "_doc": "FatesGS DTU large-overlap (views 23,24,33); val/test = views 22 / 25,28 from "
                "the authors' set_22_25_28 (information only). Built by "
                "reproductions/fatesgs/tools/prepare_dtu.py",
    }
    for name, w2c in frames:
        c2w = np.linalg.inv(w2c)
        c2w[:3, 1:3] *= -1  # OpenCV -> OpenGL (the loader flips back)
        meta["frames"].append({"file_path": name, "transform_matrix": c2w.tolist()})
    (dst / "transforms.json").write_text(json.dumps(meta, indent=1))
    ply = PlyData.read(a / "dense" / "fused.ply")["vertex"]
    pts = np.stack([ply[k] for k in "xyz"], -1).astype(np.float32)
    col = (np.stack([ply[k] for k in ("red", "green", "blue")], -1) / 255.0).astype(np.float32)
    np.savez(dst / "points.npz", points=pts, colors=col)
    np.savez(dst / "fatesgs.npz", viewmats=np.stack(viewmats).astype(np.float32),
             mono_depth=np.stack(monos), alpha=np.stack(alphas),
             pairs=np.array([pairs[k][:2] for k in range(3)], dtype=np.int64),
             names=np.array(names), full_size=np.array([w, h]))
    shutil.copytree(raw / "DTU_EVAL" / "set_23_24_33" / f"scan{scan}", dst / "dtu_eval")
    print(f"scan{scan}: {len(pts)} init points, train {size}, pairs {pairs}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--scans", type=int, nargs="*", default=SCANS)
    args = ap.parse_args()
    for scan in args.scans:
        convert(args.raw.expanduser(), args.out.expanduser(), scan)


if __name__ == "__main__":
    main()
