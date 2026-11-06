"""Build the Splatfacto-W *code-protocol* Photo Tourism layout from the native NeRF-W layout.

Official code (splatfactow_datamanager.next_train): the training split is every image of the
NeRF-W tsv (train + test); for test images the right half is masked (mask[:, :W//2] = 1), and the
test view is evaluated on its right half with its own jointly trained embedding.

SplatStudio splits must be disjoint, so each test image gets a second frame
``images/lefthalf/<stem>.png``: a lossless crop of its left W//2 columns with the same pose and
pixel intrinsics (cx stays at ~W/2, i.e. ~at the crop's right edge). The plugin recognises these
frames by cx > 0.75 * width and zero-pads them back to full width in the loss, which is exactly the
code's ``gt * mask, pred * mask``. Everything else is hard-linked (no extra disk).

    python prepare_sfw.py --src ~/datasets/phototourism/native/brandenburg-gate \
                          --out ~/datasets/phototourism/sfw/brandenburg-gate
"""

import argparse
import json
import os
from pathlib import Path

from PIL import Image

MASKED_CX_RATIO = 0.75  # plugin threshold: cx / width above this = left-half (masked) view


def link(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    os.link(src, dst)


def convert(src, out):
    src, out = Path(src).resolve(), Path(out)
    meta = json.loads((src / "transforms.json").read_text())
    frames = {f["file_path"]: f for f in meta["frames"]}
    for f in meta["frames"]:
        r = float(f["cx"]) / float(f["w"])
        if abs(r - 0.5) > 0.1:
            raise ValueError(f"{f['file_path']}: cx/W = {r:.3f}; the masked-view heuristic needs ~0.5")
    out.mkdir(parents=True, exist_ok=True)
    for rel in frames:
        link(src / rel, out / rel)
    link(src / "points.npz", out / "points.npz")
    new_frames = [dict(f) for f in meta["frames"]]
    crops = []
    for rel in meta["test_filenames"]:
        f = frames[rel]
        name = Path(rel).stem + ".png"
        crop_rel = f"images/lefthalf/{name}"
        with Image.open(src / rel) as im:
            w, h = im.size
            if (w, h) != (int(f["w"]), int(f["h"])):
                raise ValueError(f"{rel}: size mismatch")
            half = im.convert("RGB").crop((0, 0, w // 2, h))
            (out / "images" / "lefthalf").mkdir(parents=True, exist_ok=True)
            half.save(out / crop_rel)
        g = dict(f)
        g.update(file_path=crop_rel, w=w // 2, h=h)
        if not float(g["cx"]) / g["w"] > MASKED_CX_RATIO:
            raise ValueError(f"{crop_rel}: crop not recognisable as a masked view")
        new_frames.append(g)
        crops.append(crop_rel)
    new = {
        "train_filenames": list(meta["train_filenames"]) + crops,
        "val_filenames": list(meta["val_filenames"]),
        "test_filenames": list(meta["test_filenames"]),
        "splatfacto_w": {
            "source": str(src),
            "protocol": "official code: train on all images, test images with right half masked "
                        "(left W//2 columns as PNG crops in images/lefthalf), evaluate right half",
            "masked_views": crops,
            "masked_cx_ratio": MASKED_CX_RATIO,
        },
        "frames": new_frames,
    }
    for k, v in meta.items():
        if k not in new and k != "frames":
            new[k] = v
    (out / "transforms.json").write_text(json.dumps(new, indent=1))
    return {"train": len(new["train_filenames"]), "masked": len(crops),
            "test": len(new["test_filenames"])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    print(json.dumps(convert(a.src, a.out)))


if __name__ == "__main__":
    main()
