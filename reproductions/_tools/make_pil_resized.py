#!/usr/bin/env python3
"""Rebuild a native scene with images downscaled from full resolution by PIL `Image.resize`
(the 3DGS / Mip-Splatting `-r N` loader: size = (round(W/N), round(H/N)), Pillow default
BICUBIC), instead of the dataset's pre-made images_N. Poses, intrinsics, splits and SfM points
are copied from the existing native scene, which must already have exactly that image size.

    tools/ssenv python -I tools/make_pil_resized.py --native ~/datasets/mipnerf360/native/garden \
        --full ~/datasets/mipnerf360/360_v2/garden/images --factor 4 \
        --output ~/datasets/mipnerf360/pilr/garden
"""

import argparse
import json
import shutil
from pathlib import Path

from PIL import Image


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--native", required=True)
    p.add_argument("--full", required=True, help="full-resolution image folder")
    p.add_argument("--factor", type=int, help="integer -r N downscale")
    p.add_argument("--width", type=int, help="3DGS auto mode: scale = W/width, size = (int(W/scale), int(H/scale))")
    p.add_argument("--max-side", type=int, help="SuGaR load_gs_cameras: scale = max(W,H)/max_side, size = (round(W/scale), round(H/scale))")
    p.add_argument("--output", required=True)
    a = p.parse_args()
    native, full, out = Path(a.native), Path(a.full), Path(a.output)
    if out.exists() and any(out.iterdir()):
        raise SystemExit("output must be empty")
    (out / "images").mkdir(parents=True, exist_ok=True)
    meta = json.loads((native / "transforms.json").read_text())
    for frame in meta["frames"]:
        name = Path(frame["file_path"]).name
        with Image.open(full / name) as img:
            w, h = img.size
            if a.max_side:
                scale = max(w, h) / a.max_side
                size = (round(w / scale), round(h / scale))
            elif a.width:
                scale = w / a.width
                size = (int(w / scale), int(h / scale))
            else:
                size = (round(w / a.factor), round(h / a.factor))
            resized = img.convert("RGB").resize(size)  # Pillow default resample (BICUBIC)
        if size != (frame["w"], frame["h"]):
            # e.g. bicycle: round(4946/4) = 1236 (banker's rounding) vs images_4 width 1237.
            # Same full-res calibration, so rescale intrinsics by the size ratio.
            sx, sy = size[0] / frame["w"], size[1] / frame["h"]
            frame.update(fl_x=frame["fl_x"] * sx, cx=frame["cx"] * sx,
                         fl_y=frame["fl_y"] * sy, cy=frame["cy"] * sy, w=size[0], h=size[1])
        target = out / "images" / (Path(name).stem + ".png")  # lossless, unlike re-encoding JPEG
        resized.save(target)
        frame["file_path"] = f"images/{target.name}"
    meta["source"] = {**meta.get("source", {}), "images": f"PIL resize of full-res to max side {a.max_side} (round)" if a.max_side else f"PIL resize of full-res to width {a.width}" if a.width else f"PIL resize of full-res / {a.factor}",
                      "pillow": Image.__version__ if hasattr(Image, "__version__") else "unknown"}
    (out / "transforms.json").write_text(json.dumps(meta, indent=2) + "\n")
    shutil.copy2(native / "points.npz", out / "points.npz")
    print(json.dumps({"output": str(out), "frames": len(meta["frames"])}))


if __name__ == "__main__":
    main()
