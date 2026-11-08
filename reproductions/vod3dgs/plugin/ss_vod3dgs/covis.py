"""SfM covisibility for VoD-3DGS view sampling (paper §4.4 / §5.1).

The paper samples the second view of L_vc with probability proportional to LightGlue keypoint
matches and notes that "all the necessary information could be collected during the SfM
computation". We use that SfM information: matches(i, j) = number of triangulated COLMAP points
observed by both images i and j. Files: ``<scene>.npz`` with ``names`` (image names), ``centers``
(world camera centers, COLMAP frame = native frame) and ``matches`` [C, C] (zero diagonal).

CLI: ``python -m ss_vod3dgs.covis --sparse <scene>/sparse/0 --name <scene> [--out DIR]``.
"""

import argparse
import struct
from pathlib import Path

import numpy as np
import torch

DATA_DIR = Path(__file__).with_name("covis_data")


def _read(f, fmt):
    size = struct.calcsize("<" + fmt)
    return struct.unpack("<" + fmt, f.read(size))


def read_images_bin(path):
    """COLMAP images.bin -> list of (image_id, name, center[3], point3D_ids array)."""
    out = []
    with open(path, "rb") as f:
        (n,) = _read(f, "Q")
        for _ in range(n):
            image_id, qw, qx, qy, qz, tx, ty, tz, _cam = _read(f, "idddddddi")
            name = b""
            while (c := f.read(1)) != b"\x00":
                name += c
            (n2d,) = _read(f, "Q")
            data = np.frombuffer(f.read(24 * n2d), dtype=np.dtype([("x", "<f8"), ("y", "<f8"), ("p", "<i8")]))
            q = np.array([qw, qx, qy, qz])
            q /= np.linalg.norm(q)
            w, x, y, z = q
            R = np.array([
                [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
            ])
            center = -R.T @ np.array([tx, ty, tz])
            out.append((image_id, name.decode(), center, data["p"].copy()))
    return out


def covisibility(images):
    """matches[i, j] = number of distinct 3D points seen by both images (diagonal zeroed)."""
    rows, cols = [], []
    for i, (_, _, _, pids) in enumerate(images):
        pids = np.unique(pids[pids >= 0])
        rows.append(np.full(len(pids), i))
        cols.append(pids)
    rows, cols = np.concatenate(rows), np.concatenate(cols)
    _, cols = np.unique(cols, return_inverse=True)
    inc = torch.sparse_coo_tensor(
        torch.from_numpy(np.stack([rows, cols])), torch.ones(len(rows)), (len(images), int(cols.max()) + 1)
    ).coalesce()
    m = torch.sparse.mm(inc, inc.t().coalesce()).to_dense().numpy()
    np.fill_diagonal(m, 0)
    return m.astype(np.int64)


def build(sparse_dir, name, out_dir=DATA_DIR):
    images = sorted(read_images_bin(Path(sparse_dir) / "images.bin"), key=lambda r: r[1])
    out = Path(out_dir) / f"{name}.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out,
        names=np.array([r[1] for r in images]),
        centers=np.stack([r[2] for r in images]).astype(np.float64),
        matches=covisibility(images),
    )
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sparse", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--out", default=str(DATA_DIR))
    a = ap.parse_args()
    print(build(a.sparse, a.name, a.out))


if __name__ == "__main__":
    main()
