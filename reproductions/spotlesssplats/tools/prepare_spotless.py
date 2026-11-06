"""Convert an authors-preprocessed SpotLessSplats scene (Borealis doi:10.5683/SP3/WOFXFT:
images_<f>/, sparse/0/, SD/) into a native SplatStudio scene, reproducing the official
``examples/datasets/colmap.py`` Parser + ClutterDataset (commit 0caae3c):

* images from images_<factor> (official default factor 8), K = K_colmap / factor rescaled to the
  actual image size; OPENCV distortion removed with cv2.getOptimalNewCameraMatrix(alpha=0) +
  initUndistortRectifyMap + remap(INTER_LINEAR) + ROI crop (K_undist kept, as official);
  undistorted frames written as PNG (lossless);
* world normalised with similarity_from_cameras + align_principle_axes over ALL COLMAP cameras
  and all SfM points (official ``normalize=True``);
* splits: train = names containing ``clutter``, test = names containing ``extra``;
  validation = hard-linked copies of the test frames (reported only, never used for selection);
* points.npz = all normalised SfM points with their colours; ``SD`` -> symlink to the authors'
  Stable Diffusion features of the training images.

Run with the splatstudio env:  tools/ssenv python -I prepare_spotless.py --raw RAW/android
--output NATIVE/android [--factor 8]
"""

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, "/home/sejain/splatify/splatstudio/scripts")
import colmap_to_transforms as cm  # noqa: E402  (COLMAP binary readers)


# --------------------------------------------- official examples/datasets/normalize.py (verbatim)
def similarity_from_cameras(c2w, strict_scaling=False, center_method="focus"):
    t = c2w[:, :3, 3]
    R = c2w[:, :3, :3]
    ups = np.sum(R * np.array([0, -1.0, 0]), axis=-1)
    world_up = np.mean(ups, axis=0)
    world_up /= np.linalg.norm(world_up)
    up_camspace = np.array([0.0, -1.0, 0.0])
    c = (up_camspace * world_up).sum()
    cross = np.cross(world_up, up_camspace)
    skew = np.array([[0.0, -cross[2], cross[1]], [cross[2], 0.0, -cross[0]],
                     [-cross[1], cross[0], 0.0]])
    if np.sign(c) > -1:
        R_align = np.eye(3) + skew + (skew @ skew) * 1 / (1 + c)
    else:
        R_align = np.array([[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    R = R_align @ R
    fwds = np.sum(R * np.array([0, 0.0, 1.0]), axis=-1)
    t = (R_align @ t[..., None])[..., 0]
    if center_method == "focus":
        nearest = t + (fwds * -t).sum(-1)[:, None] * fwds
        translate = -np.median(nearest, axis=0)
    else:
        translate = -np.median(t, axis=0)
    transform = np.eye(4)
    transform[:3, 3] = translate
    transform[:3, :3] = R_align
    scale_fn = np.max if strict_scaling else np.median
    scale = 1.0 / scale_fn(np.linalg.norm(t + translate, axis=-1))
    transform[:3, :] *= scale
    return transform


def align_principle_axes(point_cloud):
    centroid = np.median(point_cloud, axis=0)
    translated = point_cloud - centroid
    cov = np.cov(translated, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    eigenvectors = eigenvectors[:, eigenvalues.argsort()[::-1]]
    if np.linalg.det(eigenvectors) < 0:
        eigenvectors[:, 0] *= -1
    rot = eigenvectors.T
    transform = np.eye(4)
    transform[:3, :3] = rot
    transform[:3, 3] = -rot @ centroid
    return transform


def transform_points(matrix, points):
    return points @ matrix[:3, :3].T + matrix[:3, 3]


def transform_cameras(matrix, camtoworlds):
    camtoworlds = np.einsum("nij, ki -> nkj", camtoworlds, matrix)
    scaling = np.linalg.norm(camtoworlds[:, 0, :3], axis=1)
    camtoworlds[:, :3, :3] = camtoworlds[:, :3, :3] / scaling[:, None, None]
    return camtoworlds


# ------------------------------------------------------------------------------------- convert
def link(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    os.link(src, dst)


def convert(raw, output, factor, train_kw="clutter", test_kw="extra"):
    raw, output = Path(raw), Path(output)
    sparse = raw / "sparse" / "0"
    cams = cm.read_cameras(sparse / "cameras.bin")
    images = cm.read_images(sparse / "images.bin")
    xyz, rgb = cm.read_points(sparse / "points3D.bin")
    xyz = xyz.astype(np.float64)
    images = sorted(images, key=lambda im: im["name"])
    w2c = []
    for im in images:
        m = np.eye(4)
        m[:3, :3] = cm.rotation(im["q"])
        m[:3, 3] = im["t"]
        w2c.append(m)
    c2w = np.linalg.inv(np.stack(w2c))
    T1 = similarity_from_cameras(c2w)
    c2w = transform_cameras(T1, c2w)
    points = transform_points(T1, xyz)
    T2 = align_principle_axes(points)
    c2w = transform_cameras(T2, c2w)
    points = transform_points(T2, points)

    image_dir = raw / (f"images_{factor}" if factor > 1 else "images")
    files = {Path(f).stem: f for f in os.listdir(image_dir)}
    out_img = output / "images"
    out_img.mkdir(parents=True, exist_ok=True)
    frames, train, test, val = [], [], [], []
    undist_cache = {}
    for im, pose in zip(images, c2w):
        name = im["name"]
        stem = Path(name).stem
        if train_kw not in name and test_kw not in name:
            continue
        cam = cams[im["camera"]]
        model, p = cam["model"], cam["params"]
        if model == "OPENCV":
            fx, fy, cx, cy = p[:4]
            dist = np.array(p[4:8], dtype=np.float64)
        elif model == "PINHOLE":
            fx, fy, cx, cy = p
            dist = None
        else:
            raise ValueError(f"unsupported camera model {model}")
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
        K[:2, :] /= factor
        rgb_img = np.asarray(Image.open(image_dir / files[stem]).convert("RGB"))
        h, w = rgb_img.shape[:2]
        cw, ch = cam["width"] // factor, cam["height"] // factor
        K[0, :] *= w / cw
        K[1, :] *= h / ch
        if dist is not None and np.any(dist != 0):
            key = (im["camera"], w, h)
            if key not in undist_cache:
                K_u, roi = cv2.getOptimalNewCameraMatrix(K, dist, (w, h), 0)
                mapx, mapy = cv2.initUndistortRectifyMap(K, dist, None, K_u, (w, h),
                                                         cv2.CV_32FC1)
                undist_cache[key] = (K_u, roi, mapx, mapy)
            K_u, roi, mapx, mapy = undist_cache[key]
            rgb_img = cv2.remap(rgb_img, mapx, mapy, cv2.INTER_LINEAR)
            x, y, rw, rh = roi
            rgb_img = rgb_img[y:y + rh, x:x + rw]
            K = K_u
        out = out_img / f"{stem}.png"
        Image.fromarray(np.ascontiguousarray(rgb_img)).save(out)
        gl = pose.copy()
        gl[:3, 1:3] *= -1  # OpenCV c2w -> Nerfstudio/OpenGL
        frame = {"file_path": f"images/{stem}.png", "transform_matrix": gl.tolist(),
                 "w": int(rgb_img.shape[1]), "h": int(rgb_img.shape[0]),
                 "fl_x": float(K[0, 0]), "fl_y": float(K[1, 1]),
                 "cx": float(K[0, 2]), "cy": float(K[1, 2]), "camera_model": "PINHOLE"}
        frames.append(frame)
        if train_kw in name:
            train.append(frame["file_path"])
        else:
            test.append(frame["file_path"])
            vpath = f"images/val/{stem}.png"
            link(out, output / vpath)
            frames.append({**frame, "file_path": vpath})
            val.append(vpath)
    meta = {"frames": frames, "train_filenames": train, "val_filenames": val,
            "test_filenames": test,
            "spotless": {"factor": factor, "normalize": "similarity_from_cameras+align_principle_axes",
                         "transform": (T2 @ T1).tolist(), "train_keyword": train_kw,
                         "test_keyword": test_kw,
                         "rois": {str(k): [int(v) for v in c[1]] for k, c in undist_cache.items()}}}
    (output / "transforms.json").write_text(json.dumps(meta, indent=1))
    np.savez(output / "points.npz", points=points.astype(np.float32),
             colors=np.asarray(rgb, dtype=np.float32))  # reader returns [0,1]
    sd = output / "SD"
    if sd.is_symlink() or sd.exists():
        sd.unlink()
    sd.symlink_to((raw / "SD").resolve())
    missing = [t for t in train if not (raw / "SD" / f"{Path(t).stem}.npy").exists()]
    return {"train": len(train), "test": len(test), "points": int(len(points)),
            "size": [frames[0]["w"], frames[0]["h"]], "missing_features": missing,
            "rois": meta["spotless"]["rois"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--factor", type=int, default=8)
    a = ap.parse_args()
    print(json.dumps(convert(a.raw, a.output, a.factor)))


if __name__ == "__main__":
    main()
