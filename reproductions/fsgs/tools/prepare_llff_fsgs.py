"""Build ~/datasets/llff/fewview3-r8-fsgs: llff-fewview3-r8 with the official FSGS images.

FSGS loads `--images images_8` (arguments/__init__.py: _images = "images_8", resolution -1),
i.e. the LLFF-provided 504x378 PNGs, matched to COLMAP cameras by sorted order
(scene/dataset_readers.py: rgb_mapping = sorted(glob(images_8/*))). llff-fewview3-r8 instead
PIL-resizes the full-res JPGs (~39 dB apart). Everything else (split, cameras, MVS init from the
3 training views, prep.json bounds) is hard-linked / copied from llff-fewview3-r8 unchanged.
Each images_8 PNG is hard-linked under the full-res stem name (images/IMG_xxxx.png), and the
mapping is verified per image (PSNR vs. the PIL-resized copy must exceed 30 dB).
"""

import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image

RAW = Path.home() / "datasets/llff/nerf_llff_data"
SRC = Path.home() / "datasets/llff/fewview3-r8"
DST = Path.home() / "datasets/llff/fewview3-r8-fsgs"
SCENES = ["fern", "flower", "fortress", "horns", "leaves", "orchids", "room", "trex"]


def psnr(a, b):
    a = np.asarray(Image.open(a).convert("RGB"), dtype=np.float64) / 255
    b = np.asarray(Image.open(b).convert("RGB"), dtype=np.float64) / 255
    return float(10 * np.log10(1 / ((a - b) ** 2).mean()))


def build(scene):
    full = sorted(p for p in os.listdir(RAW / scene / "images") if p.lower().endswith((".jpg", ".png")))
    small = sorted(p for p in os.listdir(RAW / scene / "images_8") if p.lower().endswith(".png"))
    if len(full) != len(small):
        raise SystemExit(f"{scene}: {len(full)} full-res vs {len(small)} images_8")
    out = DST / scene
    (out / "images").mkdir(parents=True, exist_ok=False)
    for name in ("points.npz", "transforms.json", "prep.json"):
        os.link(SRC / scene / name, out / name)
    worst = 99.0
    for f, s in zip(full, small):
        stem = Path(f).stem + ".png"
        os.link(RAW / scene / "images_8" / s, out / "images" / stem)
        worst = min(worst, psnr(out / "images" / stem, SRC / scene / "images" / stem))
    if worst < 30:
        raise SystemExit(f"{scene}: images_8 mapping suspicious (min PSNR {worst:.1f})")
    meta = {"scene": scene, "n_images": len(full), "min_psnr_vs_fewview3_r8": worst,
            "images": "nerf_llff_data/<scene>/images_8 (official FSGS --images images_8), "
                      "hard links renamed to the sorted full-res stems"}
    (out / "fsgs_prep.json").write_text(json.dumps(meta, indent=1) + "\n")
    print(json.dumps(meta))


if __name__ == "__main__":
    for sc in sys.argv[1:] or SCENES:
        build(sc)
