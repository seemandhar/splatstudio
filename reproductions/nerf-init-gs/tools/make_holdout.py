"""Train-views-only diagnostic split: drop the llffhold8 test views entirely, then hold out every
8th remaining (sorted) training view as validation/test. Hard-links images and points.npz."""
import json
import os
import sys
from pathlib import Path

src, dst = Path(sys.argv[1]).expanduser(), Path(sys.argv[2]).expanduser()
meta = json.loads((src / "transforms.json").read_text())
names = sorted(f["file_path"] for f in meta["frames"])
test = set(names[::8])
train = [n for n in names if n not in test]
held = train[::8]
inner = [n for n in train if n not in set(held)]
meta["frames"] = [f for f in meta["frames"] if f["file_path"] in set(train)]
meta.pop("split_protocol", None)
meta.update(train_filenames=inner, val_filenames=held[: len(held) // 2],
            test_filenames=held[len(held) // 2 :], diagnostic="train-views-only holdout")
dst.mkdir(parents=True, exist_ok=True)
(dst / "transforms.json").write_text(json.dumps(meta, indent=1))
for n in train:
    (dst / n).parent.mkdir(parents=True, exist_ok=True)
    if not (dst / n).exists():
        os.link((src / n).resolve(), dst / n)
if not (dst / "points.npz").exists():
    os.link(src / "points.npz", dst / "points.npz")
print(len(inner), "train", len(held), "held")
