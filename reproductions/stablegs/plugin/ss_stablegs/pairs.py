"""Image-pair set P for the depth consistency loss (paper Sec. 3.4) and the official pair stack.

Paper criteria: (1) more than 30 co-visible COLMAP feature points, (2) relative rotation angle in
[16°, 60°], (3) fraction of co-visible feature points consistent with a homography below 0.8
(degenerate near-pure-rotation pairs). The homography test uses cv2 RANSAC with a 4 px threshold,
COLMAP's default two-view `max_error` (0.8 is COLMAP's `max_H_inlier_ratio`). The official code
selects pairs differently (mvs.py: BoQ global-descriptor retrieval + rotation/translation filter);
that retrieval model is not available here (recorded fidelity gap).

The pair stack reproduces train_pair.py: stack = P + len(P) random pairs whose cameras are drawn
with probability ∝ 1 / (number of pairs in P containing the camera); one entry is popped at a
uniformly random position per iteration; refilled when empty.
"""

import math
import struct
from pathlib import Path

import numpy as np


def read_images_bin(path):
    """COLMAP images.bin -> {name: {"qvec", "tvec", "xys" [M,2], "point3D_ids" [M]}}
    (format of colmap/src/colmap/scene/reconstruction_io.cc, as read by 3DGS colmap_loader)."""
    images = {}
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        for _ in range(n):
            props = struct.unpack("<idddddddi", f.read(64))
            qvec, tvec = np.array(props[1:5]), np.array(props[5:8])
            name = b""
            while True:
                ch = f.read(1)
                if ch == b"\x00":
                    break
                name += ch
            (m,) = struct.unpack("<Q", f.read(8))
            data = np.frombuffer(f.read(24 * m), dtype=np.dtype([("x", "<f8"), ("y", "<f8"),
                                                                  ("id", "<i8")]))
            images[name.decode()] = {
                "qvec": qvec, "tvec": tvec,
                "xys": np.stack([data["x"], data["y"]], axis=1),
                "point3D_ids": data["id"].copy(),
            }
    return images


def find_colmap(data_root, colmap="0"):
    """Locate the COLMAP model of a native scene (~/datasets/<d>/native/<scene>)."""
    root = Path(data_root).resolve()
    scene = root.name
    for sub in ("tandt", "db", "360_v2"):
        cand = root.parent.parent / sub / scene / "sparse" / str(colmap)
        if (cand / "images.bin").exists():
            return cand
    raise FileNotFoundError(f"No COLMAP sparse/{colmap} found for native scene {root}")


def rotation_angle_deg(Ri, Rj):
    """Angle of the relative rotation Ri Rj^T (world-to-camera rotations), degrees."""
    c = (np.trace(Ri @ Rj.T) - 1.0) / 2.0
    return math.degrees(math.acos(float(np.clip(c, -1.0, 1.0))))


def homography_inlier_ratio(xy_i, xy_j, threshold=4.0):
    import cv2

    _, mask = cv2.findHomography(xy_i.astype(np.float64), xy_j.astype(np.float64), cv2.RANSAC,
                                 threshold)
    if mask is None:
        return 0.0
    return float(mask.mean())


def covisible(obs_i, obs_j):
    """Matched 2D observations of the 3D points seen by both images -> (xy_i, xy_j)."""
    ids_i, ids_j = obs_i["point3D_ids"], obs_j["point3D_ids"]
    vi, vj = ids_i >= 0, ids_j >= 0
    common, ii, jj = np.intersect1d(ids_i[vi], ids_j[vj], return_indices=True)
    return obs_i["xys"][vi][ii], obs_j["xys"][vj][jj]


def select_pairs(names, rotations, observations, min_covisible=30, min_rot=16.0, max_rot=60.0,
                 max_h_ratio=0.8, h_threshold=4.0):
    """names: training image names (index order); rotations: [N,3,3] world-to-camera;
    observations: {name: images.bin entry}. Returns sorted list of (i, j), i < j."""
    pairs = []
    n = len(names)
    for i in range(n):
        for j in range(i + 1, n):
            ang = rotation_angle_deg(rotations[i], rotations[j])
            if not (min_rot <= ang <= max_rot):
                continue
            xi, xj = covisible(observations[names[i]], observations[names[j]])
            if len(xi) <= min_covisible:
                continue
            if homography_inlier_ratio(xi, xj, h_threshold) >= max_h_ratio:
                continue
            pairs.append((i, j))
    return pairs


def inverse_frequency(pairs, n):
    """train_pair.py: p = 1 / np.unique(pairs, return_counts=True)[1]; p /= p.sum().
    Cameras absent from P get count 1 (the official code assumes every camera appears)."""
    counts = np.ones(n)
    if pairs:
        idx, cnt = np.unique(np.asarray(pairs), return_counts=True)
        counts[idx] = cnt
    p = 1.0 / counts
    return p / p.sum()


def refill_stack(pairs, n, rng):
    """Selected pairs + len(pairs) random pairs (size 2*len(pairs) draws reshaped to (-1, 2))."""
    stack = [list(p) for p in pairs]
    m = max(len(pairs), 1)
    draws = rng.choice(np.arange(n), size=2 * m, p=inverse_frequency(pairs, n)).reshape(-1, 2)
    stack += draws.tolist()
    return stack
