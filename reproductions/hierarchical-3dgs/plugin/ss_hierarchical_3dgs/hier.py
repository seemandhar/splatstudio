"""Hierarchy construction, storage in the Gaussian state, and cut selection.

Build (paper Sec. 3.1 + 4.2; official `GaussianHierarchyCreator <ply> <colmap> <out>` as called
by scripts/full_train.py after train_single.py): the leaf Gaussians are written as a 3DGS .ply
(`GaussianModel.save_ply` layout), the training cameras as a COLMAP `sparse/0` (only the camera
centres are read by `AppearanceFilter::init`), and the vendored creator produces
`hierarchy.hier` + `anchors.bin`, loaded with the vendored `load_hierarchy`
(`GaussianModel.create_from_hier`).

Storage: the runtime only persists per-Gaussian fields (`checkpoint.pt` -> ss-eval), so the tree
is packed into the per-Gaussian field `h3d` [N, 16] (float32, exact for integers < 2^24):
  row i, cols 0-6 : node i (depth, parent, start, count_leafs, count_merged, start_children,
                    count_children) for i < M (M = #nodes <= N = #Gaussians, checked)
  row i, cols 7-14: box i (minn xyzw, maxx xyzw; minn.w = node size used by computeSizeGPU)
  row i, col 15   : flags, 1 = hierarchy built (all rows), +2 = row is a node, +4 = Gaussian i
                    is an anchor (frozen during hierarchy optimization).
Before the hierarchy is built every row is zero (leaf training).
"""

import math
import struct
import tempfile
from pathlib import Path

import numpy as np
import torch

from .ext import creator, hierarchy

FIELD = "h3d"
WIDTH = 16
BUILT, NODE, ANCHOR = 1, 2, 4
MAX_EXACT = 1 << 24


# ----------------------------------------------------------------------------- export
def write_ply(path, params):
    """GaussianModel.save_ply: x y z nx ny nz f_dc_0..2 f_rest_0..44 opacity scale_0..2 rot_0..3
    (raw logit opacity / log scale; f_rest channel-major = features_rest.transpose(1, 2))."""
    with torch.no_grad():
        n = len(params["means"])
        cols = [params["means"].float(), torch.zeros(n, 3, device=params["means"].device),
                params["sh0"].transpose(1, 2).flatten(1).float(),
                params["shN"].transpose(1, 2).flatten(1).float(),
                params["opacities"].float()[:, None], params["scales"].float(),
                params["quats"].float()]
        data = torch.cat(cols, dim=1).cpu().numpy().astype("<f4")
    names = (["x", "y", "z", "nx", "ny", "nz"] + [f"f_dc_{i}" for i in range(3)]
             + [f"f_rest_{i}" for i in range(45)] + ["opacity"]
             + [f"scale_{i}" for i in range(3)] + [f"rot_{i}" for i in range(4)])
    assert data.shape[1] == len(names) == 62
    header = "ply\nformat binary_little_endian 1.0\nelement vertex %d\n" % n
    header += "".join(f"property float {k}\n" for k in names) + "end_header\n"
    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        f.write(np.ascontiguousarray(data).tobytes())


def rotmat_to_qvec(R):
    """COLMAP Rotation matrix -> (qw, qx, qy, qz)."""
    R = np.asarray(R, dtype=np.float64)
    K = np.array([
        [R[0, 0] - R[1, 1] - R[2, 2], 0, 0, 0],
        [R[1, 0] + R[0, 1], R[1, 1] - R[0, 0] - R[2, 2], 0, 0],
        [R[2, 0] + R[0, 2], R[2, 1] + R[1, 2], R[2, 2] - R[0, 0] - R[1, 1], 0],
        [R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1], R[0, 0] + R[1, 1] + R[2, 2]],
    ]) / 3.0
    w, V = np.linalg.eigh(K)
    q = V[[3, 0, 1, 2], np.argmax(w)]
    return -q if q[0] < 0 else q


def write_colmap(root, cameras):
    """Minimal COLMAP binary model (cameras.bin PINHOLE, images.bin without 2D points)."""
    d = Path(root) / "sparse" / "0"
    d.mkdir(parents=True, exist_ok=True)
    with open(d / "cameras.bin", "wb") as f:
        f.write(struct.pack("<Q", len(cameras)))
        for i, c in enumerate(cameras):
            K = c["K"].detach().cpu().double().numpy()
            f.write(struct.pack("<iiQQ", i + 1, 1, int(c["width"]), int(c["height"])))
            f.write(struct.pack("<4d", K[0, 0], K[1, 1], K[0, 2], K[1, 2]))
    with open(d / "images.bin", "wb") as f:
        f.write(struct.pack("<Q", len(cameras)))
        for i, c in enumerate(cameras):
            V = c["viewmat"].detach().cpu().double().numpy()
            q = rotmat_to_qvec(V[:3, :3])
            f.write(struct.pack("<I", i + 1))
            f.write(struct.pack("<7d", *q, *V[:3, 3]))
            f.write(struct.pack("<I", i + 1))
            f.write(f"view{i:05d}".encode() + b"\0")
            f.write(struct.pack("<Q", 0))


def read_anchors(path):
    """GaussianModel.create_from_hier: int32 count, then int32 Gaussian indices."""
    raw = Path(path).read_bytes()
    return torch.from_numpy(np.frombuffer(raw[4:], dtype="<i4").astype(np.int64))


def build(params, cameras, workdir=None):
    """Run the official creator on the current leaves; returns the loaded hierarchy."""
    with tempfile.TemporaryDirectory(dir=workdir, prefix="h3dgs-") as tmp:
        tmp = Path(tmp)
        write_ply(tmp / "point_cloud.ply", params)
        write_colmap(tmp / "colmap", cameras)
        (tmp / "out").mkdir()
        valid = creator().create(str(tmp / "point_cloud.ply"), str(tmp / "colmap"),
                                 str(tmp / "out"))
        xyz, shs, alpha, scales, rots, nodes, boxes = hierarchy().load_hierarchy(
            str(tmp / "out" / "hierarchy.hier"))
        anchors = read_anchors(tmp / "out" / "anchors.bin")
    return {"means": xyz, "shs": shs, "alpha": alpha, "scales": scales, "rots": rots,
            "nodes": nodes, "boxes": boxes, "anchors": anchors, "valid_leaves": int(valid)}


# ----------------------------------------------------------------------------- storage
def pack(nodes, boxes, anchors, n):
    m = len(nodes)
    if m > n:
        raise ValueError(f"hierarchy has more nodes ({m}) than Gaussians ({n})")
    if n >= MAX_EXACT:
        raise ValueError("more than 2^24 Gaussians cannot be indexed exactly in float32")
    blob = torch.zeros(n, WIDTH)
    blob[:m, :7] = nodes.float()
    blob[:m, 7:15] = boxes.reshape(m, 8).float()
    flags = torch.full((n,), float(BUILT))
    flags[:m] += NODE
    if len(anchors):
        flags[anchors] += ANCHOR
    blob[:, 15] = flags
    return blob


def is_built(params):
    blob = params[FIELD]
    return len(blob) > 0 and float(blob[0, 15]) >= BUILT


def unpack(params):
    """-> nodes int32 [M,7], boxes float [M,2,4] (both on the state's device), anchor mask."""
    blob = params[FIELD].detach()
    flags = blob[:, 15].round().long()
    node_rows = (flags & NODE) > 0
    m = int(node_rows.sum())
    if not bool(node_rows[:m].all()):
        raise ValueError("h3d node rows are not a prefix (state was reordered)")
    nodes = blob[:m, :7].round().int().contiguous()
    boxes = blob[:m, 7:15].reshape(m, 2, 4).float().contiguous()
    return nodes, boxes, (flags & ANCHOR) > 0


# ----------------------------------------------------------------------------- cut selection
def pixel_threshold(tau, camera):
    """render_hierarchy.py:57  threshold = (2 (tau + 0.5)) tan(fovx/2) / (0.5 W)."""
    W = int(camera["width"])
    tanfovx = W / (2 * float(camera["K"][0, 0]))
    return (2 * (tau + 0.5)) * tanfovx / (0.5 * W)


def sample_limit(lim_min, lim_max, u):
    """train_post.py:89  limit = 2^(u (log2 max - log2 min) + log2 min) (paper Sec. 4.1)."""
    return math.pow(2, u * (math.log2(lim_max) - math.log2(lim_min)) + math.log2(lim_min))


class Cut:
    """Buffers + calls of `expand_to_size` / `get_interpolation_weights` (render_hierarchy.py)."""

    def __init__(self):
        self.key = None

    def _buffers(self, n, device):
        if self.key != (n, device):
            self.render = torch.zeros(n, dtype=torch.int32, device=device)
            self.parent = torch.zeros(n, dtype=torch.int32, device=device)
            self.node_for = torch.zeros(n, dtype=torch.int32, device=device)
            self.weights = torch.zeros(n, dtype=torch.float32, device=device)
            self.kids = torch.zeros(n, dtype=torch.int32, device=device)
            self.key = (n, device)

    def select(self, nodes, boxes, n_gaussians, threshold, campos):
        dev = nodes.device
        self._buffers(n_gaussians, dev)
        zdir = torch.zeros(3)
        count = hierarchy().expand_to_size(nodes, boxes, float(threshold),
                                           campos.float().contiguous(), zdir, self.render,
                                           self.parent, self.node_for)
        render = self.render[:count].long()
        node_ids = self.node_for[:count].contiguous()
        hierarchy().get_interpolation_weights(node_ids, float(threshold), nodes, boxes,
                                              campos.detach().float().cpu().contiguous(), zdir,
                                              self.weights, self.kids)
        return (render, self.parent[:count].long(), self.weights[:count].clone(),
                self.kids[:count].clone())
