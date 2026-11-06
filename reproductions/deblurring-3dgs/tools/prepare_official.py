#!/usr/bin/env python3
"""Build native SplatStudio scenes from the Deblurring-3DGS release (deblur_dataset.zip, Drive
1N24ipi52XXiPfqWWAxYeyaItNXrL7nSe): same Deblur-NeRF images, but COLMAP re-run by the authors
(denser SfM points than the original Deblur-NeRF release, e.g. cake 14465 vs 6655 points).

  raw   ~/datasets/deblurring3dgs/raw/deblur_dataset/real_{defocus,camera_motion}_blur/<dir>
  full  ~/datasets/deblurring3dgs/native-full-<kind>/<scene>   (colmap_to_transforms, full res)
  out   ~/datasets/deblurring3dgs/native-<kind>/<scene>        (PIL resize /4 = official -r 4)

Official split (scene/dataset_readers.readColmapSceneInfo): cameras sorted by image name,
idx % hold == 0 -> test (sharp reference), rest -> train (blurry); hold from the hold=N file.
Validation = byte copies of the test images (val_<name>.png), never trained on or selected on.

    python3 prepare_official.py [defocus|motion] [scene ...]
"""
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path.home() / "datasets" / "deblurring3dgs"
RAW = ROOT / "raw" / "deblur_dataset"
SPLAT = Path("/home/sejain/splatify")
SSENV = SPLAT / "tools" / "ssenv"
KINDS = {"defocus": ("real_defocus_blur", "defocus"), "motion": ("real_camera_motion_blur", "blur")}


def main(kind, scenes):
    sub, prefix = KINDS[kind]
    if not scenes:
        scenes = sorted(p.name[len(prefix):] for p in (RAW / sub).iterdir() if p.is_dir())
    for s in scenes:
        raw = RAW / sub / f"{prefix}{s}"
        holds = [p.name for p in raw.glob("hold=*")]
        assert len(holds) == 1, (s, holds)
        hold = int(holds[0].split("=")[1])
        full, out = ROOT / f"native-full-{kind}" / s, ROOT / f"native-{kind}" / s
        if not (full / "transforms.json").exists():
            subprocess.run([str(SSENV), "python", "-I",
                            str(SPLAT / "splatstudio/scripts/colmap_to_transforms.py"),
                            "--scene", str(raw), "--images", "images", "--output", str(full),
                            "--split-protocol", "none"], check=True)
        if (out / "transforms.json").exists():
            print(s, "exists")
            continue
        subprocess.run([str(SSENV), "python", "-I", str(SPLAT / "tools/make_pil_resized.py"),
                        "--native", str(full), "--full", str(raw / "images"), "--factor", "4",
                        "--output", str(out)], check=True)
        meta = json.loads((out / "transforms.json").read_text())
        frames = sorted(meta["frames"], key=lambda f: Path(f["file_path"]).stem)
        test = [f["file_path"] for i, f in enumerate(frames) if i % hold == 0]
        train = [f["file_path"] for i, f in enumerate(frames) if i % hold != 0]
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
                    deblurring3dgs_protocol={"hold": hold, "kind": kind,
                                             "test": "idx % hold == 0 of sorted names",
                                             "validation": "copies of test images",
                                             "colmap": "Deblurring-3DGS release (re-run)"})
        (out / "transforms.json").write_text(json.dumps(meta, indent=2) + "\n")
        print(s, len(train), "train", len(test), "test", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
