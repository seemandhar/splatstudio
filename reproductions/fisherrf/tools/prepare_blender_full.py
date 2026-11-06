"""Build the FisherRF / 3DGS NeRF-Synthetic (Blender) protocol as native SplatStudio scenes.

Mirrors the official FisherRF code (github.com/JiangWenPL/FisherRF @ b747328,
scene/dataset_readers.py readNerfSyntheticInfo / readCamerasFromTransforms, --eval
--white_background):
  * train = the 100 transforms_train.json views in file order (the active-selection candidate
    pool; train index i == official scene.train_cameras[i]); test = all 200
    transforms_test.json views (never candidates); validation = the first 8 transforms_val.json
    views (never used for selection; the native runtime needs a non-empty validation split);
  * images: RGBA composited on white, `norm*a + (1-a)`, truncated to 8 bit like the official
    `np.array(arr*255.0, dtype=np.byte)`; saved as RGB PNG; 800x800 (resolution -1 keeps it);
  * cameras: Blender c2w as-is (OpenGL), focal = fov2focal(camera_angle_x, W), centred
    principal point;
  * init (no points3d.ply): np.random.seed(0) (official safe_state seed 0 runs before Scene),
    100k points U[-1.3, 1.3]^3, shs = U[0,1)/255; colours SH2RGB(shs) stored to the official
    points3d.ply as uint8 (storePly) and read back /255 -> written to points.npz exactly so.

Run in the splatstudio env:
    tools/ssenv python -I reproductions/fisherrf/tools/prepare_blender_full.py \
        --raw ~/datasets/nerf_synthetic/raw --out ~/datasets/nerf_synthetic/fisherrf
"""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

C0 = 0.28209479177387814
SCENES = ("chair", "drums", "ficus", "hotdog", "lego", "materials", "mic", "ship")


def composite_8bit(path):
    im = np.array(Image.open(path).convert("RGBA"))
    norm = im / 255.0
    arr = norm[:, :, :3] * norm[:, :, 3:4] + np.array([1, 1, 1]) * (1 - norm[:, :, 3:4])
    return np.trunc(arr * 255.0).astype(np.uint8)  # == np.byte cast for values in [0, 255]


def official_points():
    np.random.seed(0)
    xyz = np.random.random((100_000, 3)) * 2.6 - 1.3
    shs = np.random.random((100_000, 3)) / 255.0
    rgb = shs * C0 + 0.5  # SH2RGB
    stored = (rgb * 255).astype(np.uint8)  # storePly dtype u1
    return xyz.astype(np.float32), (stored / 255.0).astype(np.float32)


def build(scene, raw, out, n_val):
    src = raw / scene
    target = out / scene
    (target / "images").mkdir(parents=True, exist_ok=True)
    frames, splits = [], {"train": [], "val": [], "test": []}
    for split in ("train", "val", "test"):
        meta = json.loads((src / f"transforms_{split}.json").read_text())
        angle = meta["camera_angle_x"]
        chosen = meta["frames"][:n_val] if split == "val" else meta["frames"]
        for frame in chosen:
            rgb = composite_8bit(src / (frame["file_path"] + ".png"))
            H, W = rgb.shape[:2]
            name = f"images/{split}_{Path(frame['file_path']).name}.png"
            Image.fromarray(rgb, "RGB").save(target / name)
            focal = W / (2 * np.tan(angle / 2))
            frames.append({"file_path": name, "transform_matrix": frame["transform_matrix"],
                           "w": W, "h": H, "fl_x": float(focal), "fl_y": float(focal),
                           "cx": W / 2, "cy": H / 2, "camera_model": "PINHOLE"})
            splits[split].append(name)
    meta = {"frames": frames, "train_filenames": splits["train"],
            "val_filenames": splits["val"], "test_filenames": splits["test"]}
    (target / "transforms.json").write_text(json.dumps(meta, indent=1))
    points, colors = official_points()
    np.savez(target / "points.npz", points=points, colors=colors)
    prep = {"scene": scene, "n_train": len(splits["train"]), "n_val": len(splits["val"]),
            "n_test": len(splits["test"]), "init_points": len(points), "image_size": [W, H]}
    (target / "prep.json").write_text(json.dumps(prep, indent=1))
    return prep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--scenes", nargs="+", default=list(SCENES))
    ap.add_argument("--n-val", type=int, default=8)
    a = ap.parse_args()
    for scene in a.scenes:
        print(json.dumps(build(scene, a.raw.expanduser(), a.out.expanduser(), a.n_val)),
              flush=True)


if __name__ == "__main__":
    main()
