"""Build the SparseGS 12-view Mip-NeRF 360 protocol as native SplatStudio scenes.

Protocol (see reproductions/sparsegs/paper.md §6):
  * train  = the 12 images of the authors' released `<scene>_12.zip` (Google Drive folder linked
             from the official README, MipNeRF-360 Dataset/12-view). Their names are listed in
             TRAIN below; the zips themselves are kept in ~/datasets/sparsegs/raw12/.
  * test   = 3DGS llffhold8 (sorted names, idx % 8 == 0), minus any released training image
             (paper §4.1 "every eighth image as testing view").
  * val    = every other non-training image. test + val = the official metrics.py set
             (all images of the full scene excluding the training images).
  * images = full-res 360_v2/<scene>/images PIL-bicubic-resized to round(W/4) x round(H/4)
             (paper: "all input images are downscaled to 1/4"; official loadCam -r 4 path),
             saved as PNG. Cameras: the full-scene COLMAP PINHOLE model (the released sparse/0
             keeps these poses: prep_data.bat = model subselection + point_triangulator).
  * points = COLMAP 3.9.1 MVS on the 12 training views only: SIFT, exhaustive matching,
             point_triangulator with the known poses, image_undistorter, patch_match_stereo
             (max_image_size 2000, geom_consistency 1, the released run-colmap-geometric.sh /
             patch-match.cfg), stereo_fusion -> fused.ply (paper §4.1: "initial point cloud is
             output from the multi-view stereo (MVS) step").
  * depth  = BoostingMonocularDepth (LeReS res101 + pix2pix merge, Final, depthNet=2) on the
             full-res training images (official prepare_gt_depth.py), 16-bit PNG, then the
             official loader: cv2.resize INTER_CUBIC to the training size, min-max to [0, 1].
             Stored in sgs_depth.npz keyed by the native viewmat (larger = farther).

    tools/ssenv python reproductions/sparsegs/tools/prepare_m360_sgs12.py --scenes garden
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "ad-gs" / "tools"))
from prepare_llff_fewview import (  # noqa: E402
    COLMAP, read_cameras, read_images, read_ply, rotation, run)

TRAIN = {
    "bicycle": "_DSC8679 _DSC8683 _DSC8687 _DSC8691 _DSC8695 _DSC8699 _DSC8703 _DSC8707 "
               "_DSC8711 _DSC8715 _DSC8719 _DSC8723",
    "bonsai": "DSCF5567 DSCF5583 DSCF5607 DSCF5630 DSCF5653 DSCF5679 DSCF5691 DSCF5718 "
              "DSCF5737 DSCF5761 DSCF5786 DSCF5810",
    "counter": "DSCF5861 DSCF5872 DSCF5891 DSCF5912 DSCF5927 DSCF5937 DSCF5964 DSCF5979 "
               "DSCF6003 DSCF6030 DSCF6055 DSCF6091",
    "garden": "DSC07956 DSC07959 DSC07962 DSC07965 DSC07968 DSC07971 DSC07974 DSC07977 "
              "DSC07980 DSC07983 DSC07986 DSC07989",
    "kitchen": "DSCF0656 DSCF0666 DSCF0674 DSCF0745 DSCF0784 DSCF0799 DSCF0829 DSCF0848 "
               "DSCF0856 DSCF0898 DSCF0902 DSCF0927",
    "stump": "_DSC9213 _DSC9215 _DSC9217 _DSC9219 _DSC9221 _DSC9223 _DSC9225 _DSC9227 "
             "_DSC9229 _DSC9231 _DSC9253 _DSC9261",
}
BMD_REPO = Path.home() / "models/bmd/repo"
BMD_PYLIB = Path.home() / "models/bmd/pylib"


def colmap_mvs(raw, work, cams, images, train):
    """Training-view-only MVS cloud (AD-GS pipeline + the authors' patch-match settings)."""
    import sqlite3

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
             str(work / "images"), "--ImageReader.single_camera", "1",
             "--SiftExtraction.max_image_size", "4032",
             "--SiftExtraction.max_num_features", "32768"], log)
        run([COLMAP, "exhaustive_matcher", "--database_path", db,
             "--SiftMatching.guided_matching", "1"], log)
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
             str(work / "triangulated")], log)
        run([COLMAP, "image_undistorter", "--image_path", str(work / "images"), "--input_path",
             str(work / "triangulated"), "--output_path", str(work / "dense")], log)
        run([COLMAP, "patch_match_stereo", "--workspace_path", str(work / "dense"),
             "--PatchMatchStereo.max_image_size", "2000",
             "--PatchMatchStereo.geom_consistency", "1"], log)
        run([COLMAP, "stereo_fusion", "--workspace_path", str(work / "dense"), "--input_type",
             "geometric", "--output_path", str(work / "dense" / "fused.ply")], log)
    return read_ply(work / "dense" / "fused.ply")


def bmd_depths(raw, work, train):
    """Official prepare_gt_depth.py (BMD Final, LeReS) on the full-res training images."""
    src = work / "in"
    if work.exists():
        shutil.rmtree(work)
    src.mkdir(parents=True)
    for n in train:  # upstream ImageDataset only accepts lower-case extensions
        os.symlink(raw / "images" / n, src / (Path(n).stem + ".jpg"))
    env = {**os.environ, "PYTHONPATH": str(BMD_PYLIB)}
    with open(work / "bmd.log", "w") as log:
        subprocess.run([sys.executable, str(Path(__file__).with_name("run_bmd.py")), str(src),
                        str(work / "out")], cwd=BMD_REPO, env=env, check=True, stdout=log,
                       stderr=subprocess.STDOUT)
    return {n: np.array(Image.open(work / "out" / (Path(n).stem + ".png"))).astype(np.float32)
            for n in train}


def convert(scene, raw, out, downscale, skip_colmap, skip_depth):
    import cv2

    raw, out = Path(raw), Path(out)
    cams = read_cameras(raw / "sparse/0/cameras.bin")
    images = read_images(raw / "sparse/0/images.bin")
    names = sorted(i["name"] for i in images)
    train = [n + ".JPG" for n in TRAIN[scene].split()]
    assert all(n in names for n in train), "released training image missing from the scene"
    test = [n for i, n in enumerate(names) if i % 8 == 0 and n not in train]
    val = [n for n in names if n not in train and n not in test]
    (out / "images").mkdir(parents=True, exist_ok=True)
    frames, png, viewmats = [], {}, {}
    for im in sorted(images, key=lambda i: i["name"]):
        cam = cams[im["camera"]]
        assert cam["model"] == "PINHOLE"
        target = out / "images" / (Path(im["name"]).stem + ".png")
        with Image.open(raw / "images" / im["name"]) as src:
            W, H = src.size
            w, h = round(W / downscale), round(H / downscale)
            if not target.exists():
                src.convert("RGB").resize((w, h), Image.Resampling.BICUBIC).save(target)
        assert (W, H) == (cam["width"], cam["height"])
        fx, fy, cx, cy = cam["params"]
        w2c = np.eye(4)
        w2c[:3, :3] = rotation(im["q"])
        w2c[:3, 3] = im["t"]
        viewmats[im["name"]] = w2c
        c2w = np.linalg.inv(w2c)
        c2w[:3, 1:3] *= -1
        rel = f"images/{target.name}"
        png[im["name"]] = rel
        # Inria loader: FoV from the full-res focal, principal point at the image centre.
        frames.append({"file_path": rel, "transform_matrix": c2w.tolist(), "w": w, "h": h,
                       "fl_x": fx * w / W, "fl_y": fy * h / H, "cx": w / 2, "cy": h / 2,
                       "camera_model": "PINHOLE"})
    if not skip_colmap:
        xyz, rgb = colmap_mvs(raw, out / "colmap", cams, images, train)
        np.savez(out / "points.npz", points=xyz, colors=rgb)
    else:
        xyz = np.load(out / "points.npz")["points"]
    if not skip_depth:
        raw_depth = bmd_depths(raw, out / "bmd", train)
        size = (frames[0]["w"], frames[0]["h"])
        depth = []
        for n in train:
            d = cv2.resize(raw_depth[n], size, interpolation=cv2.INTER_CUBIC)  # NP_resize
            depth.append((d - d.min()) / (d.max() - d.min()))  # camera_utils.loadCam
        np.savez(out / "sgs_depth.npz", names=np.array(train),
                 viewmats=np.stack([viewmats[n] for n in train]).astype(np.float32),
                 depth=np.stack(depth).astype(np.float32))
    meta = {
        "frames": frames,
        "train_filenames": [png[n] for n in train],
        "val_filenames": [png[n] for n in val],
        "test_filenames": [png[n] for n in test],
        "source": {"scene": scene, "images": f"images (full res, PIL bicubic /{downscale})",
                   "protocol": "SparseGS 12-view: train = authors' released <scene>_12 images; "
                               "test = llffhold8 minus train; val = all other non-train images "
                               "(test+val = official metrics.py set)",
                   "init": "COLMAP MVS fused.ply from the 12 train views only",
                   "depth": "sgs_depth.npz: BoostingMonocularDepth (LeReS) train views"},
    }
    (out / "transforms.json").write_text(json.dumps(meta, indent=2) + "\n")
    info = {"scene": scene, "train": train, "n_test": len(test), "n_val": len(val),
            "points": int(len(xyz)), "image_size": [frames[0]["w"], frames[0]["h"]]}
    (out / "prep.json").write_text(json.dumps(info, indent=2) + "\n")
    return info


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--raw", default=str(Path.home() / "datasets/mipnerf360/360_v2"))
    p.add_argument("--out", default=str(Path.home() / "datasets/mipnerf360/sparsegs12"))
    p.add_argument("--scenes", nargs="+", required=True)
    p.add_argument("--downscale", type=int, default=4)
    p.add_argument("--skip-colmap", action="store_true")
    p.add_argument("--skip-depth", action="store_true")
    a = p.parse_args()
    for s in a.scenes:
        print(json.dumps(convert(s, Path(a.raw) / s, Path(a.out) / s, a.downscale,
                                 a.skip_colmap, a.skip_depth)), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
