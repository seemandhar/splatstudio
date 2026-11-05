#!/usr/bin/env python3
"""Build a multi-scale (MTMT) Mip-NeRF 360 scene following Analytic-Splatting's / Mip-Splatting's
official `train_ms.py` protocol, from an existing native scene.

Official protocol (reproductions/analytic-splatting/source/code):
  * scripts/mtmt_360.sh: `train_ms.py -i images_4 -r 1` (outdoor) / `-i images_2 -r 1` (indoor),
    `--eval` (llffhold 8 on name-sorted cameras), `--sample_more_highres`, 30k iterations.
  * train_ms.py: Scene(resolution_scales=[1,2,4,8]); train = union of the 4 scales of the
    llffhold train cameras, test = the 4 scales of the held-out cameras.
  * utils/camera_utils.py loadCam: size = (round(W/s), round(H/s)) (Python round), image =
    PIL `Image.resize(size)` (Pillow default = BICUBIC) of the images_N JPEG. The camera keeps
    the full-image FoV, so the focal scales by w/W (x) and h/H (y) and the principal point is
    the image centre (FoV projection; also required by the vendored rasterizer).
  * --sample_more_highres: after popping a camera (without-replacement stack over all 4
    scales), with prob 0.3 it is replaced by a uniformly random 1x camera, so
    P(1x) = 0.7*0.25 + 0.3 = 0.475, P(2x) = P(4x) = P(8x) = 0.175.

The runtime samples training frames uniformly with replacement, so the 1x bias is approximated
by duplicating each 1x training frame `--dup` times as distinct (hard-linked) files:
P(1x) = dup/(dup+3). dup=3 -> 0.50 (closest integer; dup=2 -> 0.40). The exact 19:7 weighting
would need 19 copies of 1x and 7 of every other scale (~40x frames in RAM), so it is not used.

Validation = hard-linked copies of the test frames at all 4 scales (runtime convention: the
loader needs disjoint splits; validation is reported, never used for selection).

    tools/ssenv python -I tools/make_multiscale.py --native ~/datasets/mipnerf360/native/garden \
        --output ~/datasets/mipnerf360/ms/garden
"""

import argparse
import json
import os
import shutil
from pathlib import Path

from PIL import Image

SCALES = (1, 2, 4, 8)


def link(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    os.link(src, dst)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--native", required=True, help="native scene (images = official images_4/images_2)")
    p.add_argument("--output", required=True)
    p.add_argument("--dup", type=int, default=3, help="copies of each 1x training frame (sample_more_highres)")
    p.add_argument("--hold", type=int, default=8)
    a = p.parse_args()
    native, out = Path(a.native), Path(a.output)
    if out.exists() and any(out.iterdir()):
        raise SystemExit("output must be empty")
    out.mkdir(parents=True, exist_ok=True)
    meta = json.loads((native / "transforms.json").read_text())
    if meta.get("split_protocol") != "llffhold8" or a.hold != 8:
        raise SystemExit("expected an llffhold8 native scene")
    native_frames = sorted(meta["frames"], key=lambda f: f["file_path"])
    held = {f["file_path"] for f in native_frames[:: a.hold]}  # == data.py llffhold8 / official

    frames, train, val, test = [], [], [], []
    for frame in native_frames:
        stem = Path(frame["file_path"]).stem
        is_test = frame["file_path"] in held
        with Image.open(native / frame["file_path"]) as img:
            img = img.convert("RGB")
            W, H = img.size
            if (W, H) != (frame["w"], frame["h"]):
                raise SystemExit(f"size mismatch {frame['file_path']}")
            if abs(frame["cx"] - W / 2) > 1e-6 or abs(frame["cy"] - H / 2) > 1e-6:
                raise SystemExit(f"native principal point not centred: {frame['file_path']}")
            for s in SCALES:
                size = (round(W / s), round(H / s))  # official loadCam rounding
                resized = img.resize(size) if s != 1 else img  # PIL default BICUBIC; 1x = images_N
                rel = f"images/s{s}/{stem}.png"
                (out / rel).parent.mkdir(parents=True, exist_ok=True)
                resized.save(out / rel)
                w, h = size
                base = {k: v for k, v in frame.items() if k not in ("file_path", "w", "h", "fl_x", "fl_y", "cx", "cy")}
                entry = {**base, "w": w, "h": h, "fl_x": frame["fl_x"] * w / W, "fl_y": frame["fl_y"] * h / H,
                         "cx": w / 2, "cy": h / 2, "scale": s, "source_frame": frame["file_path"]}
                copies = [rel]
                if is_test:
                    vrel = f"images/val/s{s}/{stem}.png"
                    link(out / rel, out / vrel)
                    frames.append({**entry, "file_path": vrel, "copy": "validation"})
                    val.append(vrel)
                    test.append(rel)
                else:
                    train.append(rel)
                    if s == 1:
                        for d in range(1, a.dup):
                            drel = f"images/s1_dup{d}/{stem}.png"
                            link(out / rel, out / drel)
                            frames.append({**entry, "file_path": drel, "copy": d})
                            train.append(drel)
                frames.append({**entry, "file_path": rel})
    p1 = a.dup / (a.dup + 3)
    out_meta = {k: v for k, v in meta.items() if k not in ("frames", "split_protocol")}
    out_meta["source"] = {**meta.get("source", {}), "native": str(native.resolve()),
                          "images": "official images_N, PIL Image.resize((round(W/s), round(H/s))) default BICUBIC, PNG",
                          "pillow": getattr(Image, "__version__", "unknown")}
    out_meta["multiscale"] = {
        "protocol": "Analytic-Splatting/Mip-Splatting train_ms.py MTMT (scales 1,2,4,8; llffhold8)",
        "scales": list(SCALES), "dup_1x_train": a.dup,
        "p_1x": p1, "p_1x_official": 0.475,
        "note": "1x train frames duplicated (hard links) to approximate --sample_more_highres; "
                "validation = hard-linked copies of test (never used for selection)",
    }
    out_meta.update(frames=frames, train_filenames=train, val_filenames=val, test_filenames=test)
    (out / "transforms.json").write_text(json.dumps(out_meta, indent=1) + "\n")
    shutil.copy2(native / "points.npz", out / "points.npz")
    print(json.dumps({"output": str(out), "frames": len(frames), "train": len(train), "val": len(val),
                      "test": len(test), "p_1x": p1}))


if __name__ == "__main__":
    main()
