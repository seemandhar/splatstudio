#!/usr/bin/env python3
"""Build a multi-scale Mip-NeRF 360 scene following MS-GS's official protocol
(github.com/JokerYan/MS-GS @ 50e5040; reproductions/ms-gs/source/code):

  * train.py / train_all.py: Scene(resolution_scales=[2**i for i in 0..max_scale]); Mip-NeRF 360
    scenes are read from `images` (full res) with the default `-r -1`, so
    utils/camera_utils.py loadCam uses global_down = W/1600 (width > 1600) and
    size = (int(W / (global_down * s)), int(H / (global_down * s))), image =
    PIL `Image.resize(size)` (Pillow default BICUBIC) of the full-res JPEG, FoV camera
    (focal scales with w/W, h/H; centred principal point).
  * --eval: llffhold 8 on name-sorted cameras.

Layout (explicit splits):
  train      = the llffhold train frames at 1x only (images/s1/<stem>.png, hard links of the
               width-1600 native scene, which was built with exactly the loadCam formula). The
               MS-GS plugin renders the 2x..128x training views itself (resampling the 1x image),
               which reproduces the official 75 % / 25 % per-epoch resolution sampling without
               replicating 1x frames in RAM.
  test       = held-out frames at every scale in --scales (PIL resize of the full-res JPEG).
  validation = hard-linked copies of test (images/val/s<k>/...), reported, never used.
128x (12 x 8 px for 1600 x 1036) is below the native loader's 11-px minimum and is skipped.

    tools/ssenv python -I tools/make_multiscale_msgs.py --native ~/datasets/mipnerf360/w1600/garden \
        --full ~/datasets/mipnerf360/360_v2/garden/images --output ~/datasets/mipnerf360/msgs/garden
"""

import argparse
import json
import os
import shutil
from pathlib import Path

from PIL import Image


def link(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    os.link(src, dst)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--native", required=True, help="width-1600 native scene (mipnerf360-1600)")
    p.add_argument("--full", required=True, help="full-resolution image folder")
    p.add_argument("--output", required=True)
    p.add_argument("--scales", default="1,2,4,8,16,32,64")
    p.add_argument("--base-width", type=int, default=1600)
    a = p.parse_args()
    native, full, out = Path(a.native), Path(a.full), Path(a.output)
    scales = [int(s) for s in a.scales.split(",")]
    if out.exists() and any(out.iterdir()):
        raise SystemExit("output must be empty")
    out.mkdir(parents=True, exist_ok=True)
    meta = json.loads((native / "transforms.json").read_text())
    if meta.get("split_protocol") != "llffhold8":
        raise SystemExit("expected an llffhold8 native scene")
    frames_in = sorted(meta["frames"], key=lambda f: f["file_path"])
    held = {f["file_path"] for f in frames_in[::8]}

    frames, train, val, test = [], [], [], []
    for frame in frames_in:
        stem = Path(frame["file_path"]).stem
        src_full = next(full.glob(stem + ".*"))
        w1, h1 = frame["w"], frame["h"]
        if abs(frame["cx"] - w1 / 2) > 1e-6 or abs(frame["cy"] - h1 / 2) > 1e-6:
            raise SystemExit(f"principal point not centred: {stem}")
        base = {k: v for k, v in frame.items() if k not in ("file_path", "w", "h", "fl_x", "fl_y", "cx", "cy")}
        if frame["file_path"] not in held:
            rel = f"images/s1/{stem}.png"
            link(native / frame["file_path"], out / rel)
            frames.append({**base, "file_path": rel, "w": w1, "h": h1, "fl_x": frame["fl_x"],
                           "fl_y": frame["fl_y"], "cx": w1 / 2, "cy": h1 / 2})
            train.append(rel)
            continue
        with Image.open(src_full) as img:
            img = img.convert("RGB")
            W, H = img.size
            g = W / a.base_width if W > a.base_width else 1.0
            if (int(W / g), int(H / g)) != (w1, h1):
                raise SystemExit(f"native 1x size mismatch for {stem}")
            for s in scales:
                size = (int(W / (g * s)), int(H / (g * s)))
                if min(size) < 11:
                    raise SystemExit(f"scale {s} gives {size}, below the loader minimum")
                if s == 1:
                    rel = f"images/s1/{stem}.png"
                    link(native / frame["file_path"], out / rel)
                else:
                    rel = f"images/s{s}/{stem}.png"
                    (out / rel).parent.mkdir(parents=True, exist_ok=True)
                    img.resize(size).save(out / rel)  # Pillow default BICUBIC, as PILtoTorch
                w, h = size
                fr = {**base, "file_path": rel, "w": w, "h": h, "fl_x": frame["fl_x"] * w / w1,
                      "fl_y": frame["fl_y"] * h / h1, "cx": w / 2, "cy": h / 2}
                frames.append(fr)
                test.append(rel)
                vrel = f"images/val/s{s}/{stem}.png"
                link(out / rel, out / vrel)
                frames.append({**fr, "file_path": vrel})
                val.append(vrel)

    out_meta = {k: v for k, v in meta.items() if k not in ("frames", "split_protocol")}
    out_meta["source"] = {**meta.get("source", {}),
                          "multiscale": f"MS-GS protocol: base width {a.base_width}, test scales {scales}, train 1x only (plugin renders lower training scales)"}
    out_meta.update(frames=frames, train_filenames=train, val_filenames=val, test_filenames=test)
    (out / "transforms.json").write_text(json.dumps(out_meta, indent=1) + "\n")
    shutil.copy2(native / "points.npz", out / "points.npz")
    print(json.dumps({"output": str(out), "train": len(train), "test": len(test), "val": len(val)}))


if __name__ == "__main__":
    main()
