#!/usr/bin/env python3
"""Build native SplatStudio scenes of the Deblur-NeRF real defocus set with the DOF-GS protocol.

  raw   ~/datasets/deblurnerf/real_defocus_blur/defocus<scene>/{images,sparse/0,hold=N}
  full  ~/datasets/deblurnerf/native-full/<scene>   (colmap_to_transforms, full-res links)
  out   ~/datasets/deblurnerf/native/<scene>        (PIL resize /4 = official -r 4, 600x400)

Splits (official readColmapSceneInfo): names sorted, idx % hold == 0 -> test (sharp reference),
rest -> train (defocused). SplatStudio's explicit-split loader needs a disjoint, non-empty
validation split, so validation = byte-identical copies of the test images (val_<name>.png)
with the same poses; they are never trained on and never used for selection.
"""
import json
import shutil
import subprocess
import sys
from pathlib import Path

HOLD = {"cake": 9, "caps": 5, "cisco": 7, "cupcake": 6, "cups": 8, "daisy": 8, "sausage": 6,
        "seal": 6, "tools": 7}  # config_files/scene_info.yaml (matches hold=N files)
ROOT = Path.home() / "datasets" / "deblurnerf"
SPLAT = Path("/home/sejain/splatify")
SSENV = SPLAT / "tools" / "ssenv"


def main(scenes):
    for s in scenes:
        raw = ROOT / "real_defocus_blur" / f"defocus{s}"
        hold_files = [p.name for p in raw.glob("hold=*")]
        assert hold_files == [f"hold={HOLD[s]}"], (s, hold_files)
        full, out = ROOT / "native-full" / s, ROOT / "native" / s
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
        test = [f["file_path"] for i, f in enumerate(frames) if i % HOLD[s] == 0]
        train = [f["file_path"] for i, f in enumerate(frames) if i % HOLD[s] != 0]
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
                    dofgs_protocol={"hold": HOLD[s], "test": "idx % hold == 0 of sorted names",
                                    "validation": "copies of test images"})
        (out / "transforms.json").write_text(json.dumps(meta, indent=2) + "\n")
        print(s, len(train), "train", len(test), "test")


if __name__ == "__main__":
    main(sys.argv[1:] or list(HOLD))
