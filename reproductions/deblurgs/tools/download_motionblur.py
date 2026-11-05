#!/usr/bin/env python3
"""Download the Deblur-NeRF real_camera_motion_blur set (images/, sparse/0, hold=N, poses_bounds.npy)
from the authors' Google Drive listing (~/datasets/deblurnerf/listing_root.txt) with gdown
(~/models/gdown-pylib). Resumable; retries rate-limited files for hours.

    python3 -I download_motionblur.py [scene ...]
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path.home() / "models" / "gdown-pylib"))
import gdown  # noqa: E402

ROOT = Path.home() / "datasets" / "deblurnerf"
SCENES = ["ball", "basket", "buick", "coffee", "decoration", "girl", "heron", "parterre",
          "puppet", "stair"]


def have(p):
    f = ROOT / p
    return f.exists() and ("hold=" in p or f.stat().st_size > 0)


def main(order):
    rows = [ln.rstrip("\n").split("\t") for ln in (ROOT / "listing_root.txt").read_text().splitlines()]
    rows = [(p, i) for p, i in rows if p.startswith("real_camera_motion_blur/")]
    want = []
    for s in order:
        mine = [(p, i) for p, i in rows if p.split("/")[1] == f"blur{s}" and "/images_4/" not in p]
        want += sorted(mine, key=lambda r: ("/images/" in r[0], r[0]))
    for _ in range(60):
        todo = [(p, i) for p, i in want if not have(p)]
        if not todo:
            break
        for p, i in todo:
            out = ROOT / p
            out.parent.mkdir(parents=True, exist_ok=True)
            try:
                gdown.download(id=i, output=str(out), quiet=True)
            except Exception as e:  # Drive rate limit
                print("retry", p, repr(e)[:100], flush=True)
                if out.exists() and out.stat().st_size == 0 and "hold=" not in p:
                    out.unlink()
                time.sleep(20)
        time.sleep(600)
    todo = [p for p, i in want if not have(p)]
    print("FAILED" if todo else "ALL DONE", todo, flush=True)


if __name__ == "__main__":
    main(sys.argv[1:] or SCENES)
