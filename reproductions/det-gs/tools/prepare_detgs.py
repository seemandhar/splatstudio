"""Build the DET-GS dataset copy: hard-linked native scene + offline priors of the training views.

For every training view (llffhold8 split of the native loader) this stores
  * ``disparity``: Depth Anything V2-Large (vitl) relative disparity at the native image size
    (official ``infer_image(raw_bgr, 518)``), float16;
  * ``canny``: Canny edge map of the ground-truth RGB (grayscale, thresholds 20 / 200), uint8 {0,1};
  * ``viewmats``: the native loader's world-to-camera matrix, used as the lookup key.
into ``<out>/<scene>/detgs_priors.npz``. Images and points.npz are hard-linked and transforms.json
is copied byte-for-byte, so ``splatstudio.data.load_dataset`` yields the same protocol_id.

Run (official DA-V2 code from a separate clone, interpreter isolated):
  tools/ssenv python -I reproductions/det-gs/tools/prepare_detgs.py \
      --src ~/datasets/mipnerf360/native --out ~/datasets/detgs-depth/mipnerf360 garden counter
"""

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

DA_CODE = Path(os.environ.get("DAV2_CODE", "~/models/depth-anything-v2/code")).expanduser()
DA_WEIGHTS = Path(
    os.environ.get("DAV2_WEIGHTS", "~/models/depth-anything-v2/depth_anything_v2_vitl.pth")
).expanduser()
CANNY_LOW, CANNY_HIGH = 20, 200  # paper §III-C


def train_names(meta):
    names = sorted(f["file_path"] for f in meta["frames"])
    held = set(names[::8])
    return [n for n in names if n not in held]


def viewmat(frame):
    c2w = torch.tensor(frame["transform_matrix"], dtype=torch.float32)
    c2w[:3, 1:3] *= -1  # same as splatstudio.data
    return torch.linalg.inv(c2w).numpy()


def load_model(device):
    sys.path.insert(0, str(DA_CODE))
    from depth_anything_v2.dpt import DepthAnythingV2

    model = DepthAnythingV2(encoder="vitl", features=256, out_channels=[256, 512, 1024, 1024])
    state = torch.load(DA_WEIGHTS, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    return model.to(device).eval()


def link_scene(src, dst):
    dst.mkdir(parents=True, exist_ok=True)
    (dst / "images").mkdir(exist_ok=True)
    for p in sorted((src / "images").iterdir()):
        q = dst / "images" / p.name
        if not q.exists():
            os.link(p, q)
    if not (dst / "points.npz").exists():
        os.link(src / "points.npz", dst / "points.npz")
    shutil.copyfile(src / "transforms.json", dst / "transforms.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("scenes", nargs="+")
    args = ap.parse_args()
    model = load_model(args.device)
    for scene in args.scenes:
        src, dst = Path(args.src).expanduser() / scene, Path(args.out).expanduser() / scene
        link_scene(src, dst)
        meta = json.loads((dst / "transforms.json").read_text())
        frames = {f["file_path"]: f for f in meta["frames"]}
        names = train_names(meta)
        disp, edges, vms = [], [], []
        for k, name in enumerate(names):
            path = dst / name
            raw_bgr = cv2.imread(str(path))  # official run.py input
            with torch.no_grad():
                d = model.infer_image(raw_bgr, 518)
            with Image.open(path) as im:  # same decoder as the training loader
                rgb = np.asarray(im.convert("RGB"))
            gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
            e = cv2.Canny(gray, CANNY_LOW, CANNY_HIGH) > 0
            if d.shape != rgb.shape[:2]:
                raise RuntimeError(f"{name}: depth {d.shape} vs image {rgb.shape}")
            disp.append(d.astype(np.float16))
            edges.append(e.astype(np.uint8))
            vms.append(viewmat(frames[name]))
            if k % 20 == 0:
                print(scene, k, len(names), float(d.min()), float(d.max()), float(e.mean()),
                      flush=True)
        np.savez(
            dst / "detgs_priors.npz",
            names=np.array(names),
            viewmats=np.stack(vms).astype(np.float32),
            disparity=np.stack(disp),
            canny=np.stack(edges),
        )
        info = {
            "scene": scene, "views": len(names), "estimator": "Depth-Anything-V2-Large (vitl)",
            "weights": str(DA_WEIGHTS), "input_size": 518, "canny": [CANNY_LOW, CANNY_HIGH],
            "split": "llffhold8 train views only",
        }
        (dst / "detgs_priors.json").write_text(json.dumps(info, indent=1))
        print("wrote", dst / "detgs_priors.npz", flush=True)


if __name__ == "__main__":
    main()
