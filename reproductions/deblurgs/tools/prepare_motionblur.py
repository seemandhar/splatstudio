#!/usr/bin/env python3
"""Build native SplatStudio scenes of the Deblur-NeRF real camera-motion-blur set (DeblurGS protocol).

  raw   ~/datasets/deblurnerf/real_camera_motion_blur/blur<scene>/{images,sparse/0,hold=N}
  full  ~/datasets/deblurnerf/native-motion-full/<scene>  (colmap_to_transforms, full-res 2400x1600)
  out   ~/datasets/deblurnerf/native-motion/<scene>       (PIL resize /4 = official -r 4, 600x400)

Splits (official deblurgs scene/dataset_readers.readColmapSceneInfo): int(image name) % hold == 0
-> test (sharp reference), rest -> train (motion-blurred). Validation = byte copies of the test
images (val_<name>.png) with the same poses; never trained on, never used for selection.
"""
import json
import shutil
import subprocess
import sys
from pathlib import Path

HOLD = {"ball": 7, "basket": 7, "buick": 7, "coffee": 6, "decoration": 6, "girl": 7, "heron": 8,
        "parterre": 6, "puppet": 6, "stair": 6}  # hold=N files of the release
ROOT = Path.home() / "datasets" / "deblurnerf"
SPLAT = Path("/home/sejain/splatify")
SSENV = SPLAT / "tools" / "ssenv"


def main(scenes):
    for s in scenes:
        raw = ROOT / "real_camera_motion_blur" / f"blur{s}"
        hold_files = [p.name for p in raw.glob("hold=*")]
        assert hold_files == [f"hold={HOLD[s]}"], (s, hold_files)
        full, out = ROOT / "native-motion-full" / s, ROOT / "native-motion" / s
        # COLMAP (run on 600x400 images_4) names the frames NNN.png; the release's full-res
        # images/ are NNN.jpg (some scenes' COLMAP names are .jpg). The official loader falls back to the .jpg (readColmapCameras),
        # i.e. it reads images/ and resizes with -r 4. Stage .png-named links to the JPEGs.
        stage = ROOT / "motion-stage" / s
        if not stage.exists():
            (stage / "images").mkdir(parents=True)
            (stage / "sparse").symlink_to(raw / "sparse")
            for jpg in sorted((raw / "images").glob("*.jpg")):
                for ext in (".png", ".jpg"):  # COLMAP names differ per scene
                    (stage / "images" / (jpg.stem + ext)).symlink_to(jpg)
        if not (full / "transforms.json").exists():
            subprocess.run([str(SSENV), "python", "-I",
                            str(SPLAT / "splatstudio/scripts/colmap_to_transforms.py"),
                            "--scene", str(stage), "--images", "images", "--output", str(full),
                            "--split-protocol", "none"], check=True)
        if (out / "transforms.json").exists():
            print(s, "exists")
            continue
        subprocess.run([str(SSENV), "python", "-I", str(SPLAT / "tools/make_pil_resized.py"),
                        "--native", str(full), "--full", str(stage / "images"), "--factor", "4",
                        "--output", str(out)], check=True)
        meta = json.loads((out / "transforms.json").read_text())
        frames = sorted(meta["frames"], key=lambda f: Path(f["file_path"]).stem)
        test = [f["file_path"] for f in frames if int(Path(f["file_path"]).stem) % HOLD[s] == 0]
        train = [f["file_path"] for f in frames if int(Path(f["file_path"]).stem) % HOLD[s] != 0]
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
                    deblurgs_protocol={"hold": HOLD[s], "test": "int(name) % hold == 0",
                                       "validation": "copies of test images",
                                       "resize": "PIL /4 of the 2400x1600 JPGs (-r 4)"})
        (out / "transforms.json").write_text(json.dumps(meta, indent=2) + "\n")
        print(s, len(train), "train", len(test), "test")


if __name__ == "__main__":
    main(sys.argv[1:] or list(HOLD))
