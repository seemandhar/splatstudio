"""Convert an authors-preprocessed RobustSplat scene (hf.co/datasets/fcy99/RobustSplat-data:
COLMAP re-run + image_undistorter -> PINHOLE ``sparse/0``, mediapy-downsampled ``images_<f>`` and
``images_<4f>``) into a native SplatStudio scene, reproducing the official 3DGS-Inria loader of
github.com/fcyycf/RobustSplat @ a130281 (``scene/dataset_readers.py``, ``utils/camera_utils.py``):

* training images = ``images_<factor>`` files used as-is (``loadCam`` reads the pre-downsampled
  file and does not resize it); hard-linked, no re-encoding;
* intrinsics: Inria keeps only the FoV (focal2fov with the COLMAP width/height) and a centred
  principal point, so fl_x = fx * w / W_colmap, fl_y = fy * h / H_colmap, cx = w / 2, cy = h / 2;
* coarse view for mask bootstrapping (``scene.getTrainCameras(4)``) = ``images_<4 factor>``,
  hard-linked to ``coarse/``; its size and intrinsics are stored per frame
  (``coarse_file_path``, ``coarse_w``, ``coarse_h``, ``coarse_fl_x``, ``coarse_fl_y``);
* world frame = COLMAP (no normalisation); points.npz = all SfM points, colours / 255;
* split: train = names containing ``clutter``, test = names containing ``extra`` (``clean`` frames
  unused, official ``readColmapSceneInfo``); validation = hard-linked copies of the test frames
  (reported only, never used for selection).

Run:  tools/ssenv python -I prepare_robustsplat.py --raw RAW/<scene> --output NATIVE/<scene>
      --factor 8   (4 for On-the-go patio / arcdetriomphe)
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, "/home/sejain/splatify/splatstudio/scripts")
import colmap_to_transforms as cm  # noqa: E402  (COLMAP binary readers)


def link(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    os.link(src, dst)


def convert(raw, output, factor, train_kw="clutter", test_kw="extra"):
    raw, output = Path(raw), Path(output)
    sparse = raw / "sparse" / "0"
    cams = cm.read_cameras(sparse / "cameras.bin")
    images = sorted(cm.read_images(sparse / "images.bin"), key=lambda im: im["name"])
    xyz, rgb = cm.read_points(sparse / "points3D.bin")
    fine_dir, coarse_dir = raw / f"images_{factor}", raw / f"images_{4 * factor}"
    frames, train, test, val = [], [], [], []
    for im in images:
        name = im["name"]
        if train_kw not in name and test_kw not in name:
            continue
        cam = cams[im["camera"]]
        if cam["model"] != "PINHOLE":
            raise ValueError(f"{name}: expected undistorted PINHOLE camera, got {cam['model']}")
        fx, fy = cam["params"][:2]
        W, H = cam["width"], cam["height"]
        src = fine_dir / name
        with Image.open(src) as img:
            w, h = img.size
        rel = f"images/{name}"
        link(src, output / rel)
        csrc = coarse_dir / name
        with Image.open(csrc) as img:
            cw, ch = img.size
        crel = f"coarse/{name}"
        link(csrc, output / crel)
        w2c = np.eye(4)
        w2c[:3, :3] = cm.rotation(im["q"])
        w2c[:3, 3] = im["t"]
        c2w = np.linalg.inv(w2c)
        c2w[:3, 1:3] *= -1  # OpenCV -> Nerfstudio/OpenGL
        frame = {"file_path": rel, "transform_matrix": c2w.tolist(), "w": w, "h": h,
                 "fl_x": fx * w / W, "fl_y": fy * h / H, "cx": w / 2.0, "cy": h / 2.0,
                 "camera_model": "PINHOLE", "coarse_file_path": crel, "coarse_w": cw,
                 "coarse_h": ch, "coarse_fl_x": fx * cw / W, "coarse_fl_y": fy * ch / H}
        frames.append(frame)
        if train_kw in name:
            train.append(rel)
        else:
            test.append(rel)
            vrel = f"images/val/{name}"
            link(src, output / vrel)
            frames.append({**frame, "file_path": vrel})
            val.append(vrel)
    meta = {"frames": frames, "train_filenames": train, "val_filenames": val,
            "test_filenames": test,
            "robustsplat": {"factor": factor, "coarse_factor": 4 * factor, "source": str(raw),
                            "principal_point": "centred (Inria FoV camera)",
                            "train_keyword": train_kw, "test_keyword": test_kw}}
    output.mkdir(parents=True, exist_ok=True)
    (output / "transforms.json").write_text(json.dumps(meta, indent=1))
    np.savez(output / "points.npz", points=xyz.astype(np.float32),
             colors=np.asarray(rgb, dtype=np.float32))
    return {"train": len(train), "test": len(test), "points": int(len(xyz)),
            "size": [frames[0]["w"], frames[0]["h"]],
            "coarse_size": [frames[0]["coarse_w"], frames[0]["coarse_h"]]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--factor", type=int, default=8)
    args = ap.parse_args()
    print(json.dumps(convert(args.raw, args.output, args.factor)))


if __name__ == "__main__":
    main()
