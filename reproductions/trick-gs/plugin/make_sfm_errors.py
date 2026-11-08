"""Write the per-point SfM reprojection errors Trick-GS needs for its 20 % lowest-error init.

The native datasets store only xyz/rgb (points.npz, same order as COLMAP points3D.bin). This
reads `<raw>/sparse/0/points3D.bin` (COLMAP binary: id, xyz, rgb, error, track), checks that its
float32 xyz equals the native points.npz exactly, and saves
`<out>/<sha256(float32 xyz bytes)>.npy` (float64 errors) — the key the density looks up from the
initial Gaussian means. Usage:
  tools/ssenv python make_sfm_errors.py --raw ~/datasets/tandt_db/tandt/truck \
      --native ~/datasets/tandt_db/native/truck --out ../sfm_errors
"""

import argparse
import hashlib
import json
import os
import struct
from pathlib import Path

import numpy as np


def read(f, fmt):
    return struct.unpack("<" + fmt, f.read(struct.calcsize("<" + fmt)))


def read_points(path):
    xyz, err = [], []
    with open(path, "rb") as f:
        for _ in range(read(f, "Q")[0]):
            v = read(f, "QdddBBBd")
            xyz.append(v[1:4])
            err.append(v[7])
            f.seek(read(f, "Q")[0] * 8, os.SEEK_CUR)
    return np.asarray(xyz, np.float32), np.asarray(err, np.float64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True, type=Path)
    ap.add_argument("--native", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()
    xyz, err = read_points(a.raw / "sparse" / "0" / "points3D.bin")
    with np.load(a.native / "points.npz", allow_pickle=False) as d:
        native = d["points"].astype(np.float32)
    if native.shape != xyz.shape or not np.array_equal(native, xyz):
        raise SystemExit("points3D.bin xyz differs from the native points.npz")
    key = hashlib.sha256(np.ascontiguousarray(xyz).tobytes()).hexdigest()
    a.out.mkdir(parents=True, exist_ok=True)
    np.save(a.out / f"{key}.npy", err)
    index = a.out / "index.json"
    table = json.loads(index.read_text()) if index.exists() else {}
    table[str(a.native.resolve())] = {"sha256": key, "points": len(err),
                                      "error_p20": float(np.quantile(err, 0.2))}
    index.write_text(json.dumps(table, indent=1, sort_keys=True) + "\n")
    print(json.dumps({"scene": a.native.name, "sha256": key, "points": len(err)}))


if __name__ == "__main__":
    main()
