"""Build the FSGS/CoR-GS/AD-GS few-view LLFF protocol as native SplatStudio scenes.

Mirrors the official AD-GS code exactly where possible:
  * split  (scene/dataset_readers.py:349-360): sort cameras by image name, test = idx % 8 == 0,
    train = remaining images at round(np.linspace(0, len-1, n_views)) (banker's rounding);
  * images (utils/camera_utils.py loadCam with -r 4): full-res images/ resized by PIL to
    round(W/4) x round(H/4) (bicubic, Pillow default), saved losslessly as PNG;
  * cameras: the official reader treats SIMPLE_RADIAL as a pinhole (k1 ignored) and its
    projection matrix assumes a centred principal point -> PINHOLE, cx = W/2, cy = H/2;
  * init points (tools/colmap_llff.py): COLMAP SIFT on the n full-res training images,
    exhaustive guided matching, point_triangulator with the known poses, image_undistorter,
    patch_match_stereo, stereo_fusion -> dense fused cloud from the training views only.

Output: <out>/<scene>/{images/*.png, transforms.json, points.npz, colmap/, prep.json}.
Unused (non-train, non-test) images go to val_filenames; validation is never used for selection.

    tools/ssenv python reproductions/ad-gs/tools/prepare_llff_fewview.py --scenes horns \
        --raw ~/datasets/llff/nerf_llff_data --out ~/datasets/llff/fewview3 --n-views 3
"""

import argparse
import json
import os
import shutil
import sqlite3
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image

COLMAP = os.environ.get("COLMAP_BIN", str(Path.home() / "miniconda3/envs/colmap/bin/colmap"))
MODELS = {0: ("SIMPLE_PINHOLE", 3), 1: ("PINHOLE", 4), 2: ("SIMPLE_RADIAL", 4), 3: ("RADIAL", 5)}


def read(f, fmt):
    return struct.unpack("<" + fmt, f.read(struct.calcsize("<" + fmt)))


def read_cameras(path):
    cams = {}
    with open(path, "rb") as f:
        for _ in range(read(f, "Q")[0]):
            cid, mid, w, h = read(f, "iiQQ")
            name, n = MODELS[mid]
            cams[cid] = {"model": name, "width": w, "height": h, "params": read(f, "d" * n)}
    return cams


def read_images(path):
    images = []
    with open(path, "rb") as f:
        for _ in range(read(f, "Q")[0]):
            iid, qw, qx, qy, qz, tx, ty, tz, cid = read(f, "idddddddi")
            name = b""
            while (c := f.read(1)) != b"\x00":
                name += c
            f.seek(read(f, "Q")[0] * 24, os.SEEK_CUR)
            images.append({"id": iid, "name": name.decode(), "q": (qw, qx, qy, qz),
                           "t": (tx, ty, tz), "camera": cid})
    return images


def rotation(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def split(names, n_views, hold=8):
    ordered = sorted(names)
    test = [n for i, n in enumerate(ordered) if i % hold == 0]
    cand = [n for i, n in enumerate(ordered) if i % hold != 0]
    idx = [round(i) for i in np.linspace(0, len(cand) - 1, n_views)]
    train = [n for i, n in enumerate(cand) if i in idx]
    assert len(train) == n_views
    return train, test


def run(cmd, log):
    log.write("$ " + " ".join(cmd) + "\n")
    log.flush()
    subprocess.run(cmd, check=True, stdout=log, stderr=subprocess.STDOUT)


def read_ply(path):
    from plyfile import PlyData

    v = PlyData.read(path)["vertex"]
    xyz = np.stack([v["x"], v["y"], v["z"]], -1).astype(np.float32)
    rgb = np.stack([v["red"], v["green"], v["blue"]], -1).astype(np.float32) / 255
    return xyz, rgb


def colmap_points(raw, work, cams, images, train):
    """tools/colmap_llff.py pipeline (same options) into work/; returns the fused cloud."""
    if work.exists():
        shutil.rmtree(work)
    for d in ("created", "triangulated", "images"):
        (work / d).mkdir(parents=True)
    for n in train:
        shutil.copy2(raw / "images" / n, work / "images" / n)
    with open(work / "created" / "cameras.txt", "w") as f:
        for cid, c in cams.items():
            f.write(f"{cid} {c['model']} {c['width']} {c['height']} "
                    + " ".join(repr(p) for p in c["params"]) + "\n")
    (work / "created" / "points3D.txt").write_text("")
    db = str(work / "database.db")
    with open(work / "colmap.log", "w") as log:
        run([COLMAP, "feature_extractor", "--database_path", db, "--image_path",
             str(work / "images"), "--SiftExtraction.max_image_size", "4032",
             "--SiftExtraction.max_num_features", "32768",
             "--SiftExtraction.estimate_affine_shape", "1",
             "--SiftExtraction.domain_size_pooling", "1"], log)
        run([COLMAP, "exhaustive_matcher", "--database_path", db,
             "--SiftMatching.guided_matching", "1", "--SiftMatching.max_num_matches", "32768"], log)
        con = sqlite3.connect(db)
        rank = [r[1] for r in con.execute("SELECT * FROM images")]
        con.close()
        by_name = {i["name"]: i for i in images}
        with open(work / "created" / "images.txt", "w") as f:
            for k, name in enumerate(rank):
                i = by_name[os.path.basename(name)]
                f.write(f"{k + 1} " + " ".join(repr(v) for v in (*i["q"], *i["t"]))
                        + f" {i['camera']} {i['name']}\n\n")
        run([COLMAP, "point_triangulator", "--database_path", db, "--image_path",
             str(work / "images"), "--input_path", str(work / "created"), "--output_path",
             str(work / "triangulated"), "--Mapper.ba_local_max_num_iterations", "40",
             "--Mapper.ba_local_max_refinements", "3", "--Mapper.ba_global_max_num_iterations",
             "100"], log)
        run([COLMAP, "model_converter", "--input_path", str(work / "triangulated"),
             "--output_path", str(work / "triangulated"), "--output_type", "TXT"], log)
        run([COLMAP, "image_undistorter", "--image_path", str(work / "images"), "--input_path",
             str(work / "triangulated"), "--output_path", str(work / "dense")], log)
        run([COLMAP, "patch_match_stereo", "--workspace_path", str(work / "dense")], log)
        run([COLMAP, "stereo_fusion", "--workspace_path", str(work / "dense"), "--output_path",
             str(work / "dense" / "fused.ply")], log)
    return read_ply(work / "dense" / "fused.ply")


def convert(raw, out, n_views, downscale, skip_colmap):
    raw, out = Path(raw), Path(out)
    cams = read_cameras(raw / "sparse/0/cameras.bin")
    images = read_images(raw / "sparse/0/images.bin")
    names = sorted(i["name"] for i in images)
    train, test = split(names, n_views)
    val = [n for n in names if n not in train and n not in test]
    (out / "images").mkdir(parents=True, exist_ok=True)
    frames, png = [], {}
    for im in sorted(images, key=lambda i: i["name"]):
        cam = cams[im["camera"]]
        with Image.open(raw / "images" / im["name"]) as src:
            W, H = src.size
            w, h = round(W / downscale), round(H / downscale)
            target = out / "images" / (Path(im["name"]).stem + ".png")
            src.convert("RGB").resize((w, h), Image.Resampling.BICUBIC).save(target)
        assert (W, H) == (cam["width"], cam["height"])
        f = cam["params"][0] * w / W  # SIMPLE_RADIAL focal; k1 ignored as in the official reader
        w2c = np.eye(4)
        w2c[:3, :3] = rotation(im["q"])
        w2c[:3, 3] = im["t"]
        c2w = np.linalg.inv(w2c)
        c2w[:3, 1:3] *= -1
        rel = f"images/{target.name}"
        png[im["name"]] = rel
        frames.append({"file_path": rel, "transform_matrix": c2w.tolist(), "w": w, "h": h,
                       "fl_x": f, "fl_y": f, "cx": w / 2, "cy": h / 2,
                       "camera_model": "PINHOLE"})
    if not skip_colmap:
        xyz, rgb = colmap_points(raw, out / "colmap", cams, images, train)
        np.savez(out / "points.npz", points=xyz, colors=rgb)
    else:
        xyz = np.load(out / "points.npz")["points"]
    meta = {
        "frames": frames,
        "train_filenames": [png[n] for n in train],
        "val_filenames": [png[n] for n in val],
        "test_filenames": [png[n] for n in test],
        "source": {"scene": raw.name, "images": "images (full res, PIL bicubic /%d)" % downscale,
                   "protocol": f"FSGS/CoR-GS/AD-GS few-view: test=every 8th sorted, train={n_views} "
                               "uniform (np.linspace+round) of the rest; val=unused images, "
                               "never used for selection",
                   "init": "COLMAP point_triangulator + MVS fused.ply from train views only "
                           "(AD-GS tools/colmap_llff.py)"},
    }
    (out / "transforms.json").write_text(json.dumps(meta, indent=2) + "\n")
    bounds = np.load(raw / "poses_bounds.npy")[:, -2:]
    order = sorted(range(len(images)), key=lambda k: images[k]["id"])
    # official: bounds row index = enumeration index over sorted image ids
    id_bounds = {images[k]["name"]: bounds[r].tolist() for r, k in enumerate(order)}
    info = {"scene": raw.name, "train": train, "test": test, "n_val": len(val),
            "points": int(len(xyz)), "image_size": [frames[0]["w"], frames[0]["h"]],
            "train_bounds_poses_bounds_npy": {n: id_bounds[n] for n in train}}
    (out / "prep.json").write_text(json.dumps(info, indent=2) + "\n")
    return info


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--raw", default=str(Path.home() / "datasets/llff/nerf_llff_data"))
    p.add_argument("--out", default=str(Path.home() / "datasets/llff/fewview3"))
    p.add_argument("--scenes", nargs="+", required=True)
    p.add_argument("--n-views", type=int, default=3)
    p.add_argument("--downscale", type=int, default=4)
    p.add_argument("--skip-colmap", action="store_true")
    a = p.parse_args()
    for s in a.scenes:
        print(json.dumps(convert(Path(a.raw) / s, Path(a.out) / s, a.n_views, a.downscale,
                                 a.skip_colmap)), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
