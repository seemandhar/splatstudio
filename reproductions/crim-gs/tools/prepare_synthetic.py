#!/usr/bin/env python3
"""Build native SplatStudio scenes for the Deblur-NeRF *synthetic* camera-motion-blur set from the
Deblurring-3DGS release (deblur_dataset.zip; COLMAP run on the blurry images by the authors).

  raw  ~/datasets/deblurring3dgs/raw/deblur_dataset/synthetic_camera_motion_blur/blur<scene>
  out  ~/datasets/deblurring3dgs/native-synmotion/<scene>  (full res 600x400 = official -r 1)

Split = 3DGS/CoMoGaussian readColmapSceneInfo(eval, llffhold=8): cameras sorted by image name,
idx % 8 == 0 -> test (these are the sharp reference images of Deblur-NeRF synthetic; verified by
Laplacian energy), rest -> train (blurry). Validation = byte copies of test (never used for
selection). The release has no hold=N file for the synthetic set; Deblur-NeRF's synthetic
configs use llffhold = 8.

    python3 prepare_synthetic.py [scene ...]      (default: all 5)
"""
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path.home() / "datasets" / "deblurring3dgs"
RAW = ROOT / "raw" / "deblur_dataset" / "synthetic_camera_motion_blur"
SPLAT = Path("/home/sejain/splatify")
SSENV = SPLAT / "tools" / "ssenv"
HOLD = 8


def main(scenes):
    if not scenes:
        scenes = sorted(p.name[len("blur"):] for p in RAW.iterdir() if p.is_dir())
    for s in scenes:
        raw, out = RAW / f"blur{s}", ROOT / "native-synmotion" / s
        if (out / "transforms.json").exists():
            meta = json.loads((out / "transforms.json").read_text())
            if "test_filenames" in meta:
                print(s, "exists")
                continue
        else:
            subprocess.run([str(SSENV), "python", "-I",
                            str(SPLAT / "splatstudio/scripts/colmap_to_transforms.py"),
                            "--scene", str(raw), "--images", "images", "--output", str(out),
                            "--split-protocol", "none"], check=True)
        meta = json.loads((out / "transforms.json").read_text())
        frames = sorted(meta["frames"], key=lambda f: Path(f["file_path"]).stem)
        test = [f["file_path"] for i, f in enumerate(frames) if i % HOLD == 0]
        train = [f["file_path"] for i, f in enumerate(frames) if i % HOLD != 0]
        val = []
        for f in frames:
            if f["file_path"] in test:
                src = out / f["file_path"]
                dst = src.with_name("val_" + src.name)
                shutil.copy2(src, dst)
                g = dict(f, file_path=f"images/{dst.name}")
                meta["frames"].append(g)
                val.append(g["file_path"])
        meta.pop("split_protocol", None)
        meta.update(train_filenames=train, val_filenames=val, test_filenames=test,
                    crimgs_protocol={"hold": HOLD, "kind": "synthetic_camera_motion_blur",
                                     "resolution": "full (600x400, official -r 1)",
                                     "test": "idx % 8 == 0 of sorted names (sharp)",
                                     "validation": "copies of test images",
                                     "colmap": "Deblurring-3DGS release (on blurry images)"})
        (out / "transforms.json").write_text(json.dumps(meta, indent=2) + "\n")
        print(s, len(train), "train", len(test), "test", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
