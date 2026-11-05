"""Run-wide shared state of the SparseGS components and the official camera geometry.

The official train.py loop draws, every iteration and in this order,
``pick_diff_cam = randint(1, 100) <= SDS_freq*100 and lambda_diffusion and it > iters*2/3`` and
``pick_warp_cam = randint(1, 10) <= 4 and lambda_reg > 0 and it > warp_reg_start_itr``
(both ``randint`` calls always happen). ``Context.draws(step)`` reproduces that per step so the
Pearson loss (skipped on warp iterations), the warp loss, the SDS loss and the density (which
takes its densification statistics from the warp render on warp iterations) agree.

Also here: the BMD depth sidecar (``sgs_depth.npz``, keyed by the native viewmat), the training
images (read from the data root, the runtime only exposes cameras), the warp cameras
(scene/__init__.py + scene/generate_new_cam.py), the forward depth warp
(scene/cameras.py generate_warp_gt) and the elliptic-cylinder sampler (utils/pose_utils.py
setup_ellipse_sampling + scene/__init__.py getRandEllipsePose).
"""

import json
import math
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image


# ------------------------------------------------------------------ data root / sidecars
def data_root():
    env = os.environ.get("SGS_DATA")
    if env:
        return Path(env)
    argv = sys.argv
    for i, a in enumerate(argv):
        if a == "--data" and i + 1 < len(argv):
            return Path(argv[i + 1])
        if a.startswith("--data="):
            return Path(a.split("=", 1)[1])
    raise RuntimeError("SparseGS: set SGS_DATA or run through `ss-train --data`")


def native_viewmat(transform_matrix):
    """Same conversion as splatstudio.data.load_dataset (OpenGL c2w -> OpenCV w2c)."""
    c2w = torch.tensor(transform_matrix, dtype=torch.float32)
    c2w[:3, 1:3] *= -1
    return torch.linalg.inv(c2w)


class FrameTable:
    """Per training view: monocular depth (BMD, [0,1], larger = farther) and RGB image."""

    def __init__(self, viewmats, depth, images, tol=1e-4):
        self.viewmats = torch.as_tensor(np.asarray(viewmats), dtype=torch.float32)
        self.depth = torch.as_tensor(np.asarray(depth), dtype=torch.float32)
        self.images = images  # list of [H,W,3] float tensors or None
        self.tol = float(tol)
        self.cache = {}

    @classmethod
    def load(cls, root):
        root = Path(root)
        path = root / "sgs_depth.npz"
        if not path.exists():
            raise FileNotFoundError(f"SparseGS depth sidecar {path} not found (build it with "
                                    "reproductions/sparsegs/tools/prepare_m360_sgs12.py)")
        meta = json.loads((root / "transforms.json").read_text())
        by_vm = []
        for f in meta["frames"]:
            if f["file_path"] in meta["train_filenames"]:
                by_vm.append((native_viewmat(f["transform_matrix"]), f["file_path"]))
        with np.load(path, allow_pickle=False) as d:
            depth = d["depth"]
            vms = torch.as_tensor(d["viewmats"], dtype=torch.float32)
        images = []
        for vm in vms:
            err = [float((vm - v).abs().max()) for v, _ in by_vm]
            k = int(np.argmin(err))
            if err[k] > 1e-3:
                raise KeyError("SparseGS sidecar viewmat does not match any training frame")
            with Image.open(root / by_vm[k][1]) as im:
                images.append(torch.from_numpy(
                    np.asarray(im.convert("RGB"), dtype=np.float32) / 255).contiguous())
            vms[len(images) - 1] = by_vm[k][0]  # exact native viewmat for lookups
        return cls(vms, depth, images)

    def index(self, viewmat):
        err = (self.viewmats - viewmat.detach().float().cpu()[None]).abs().amax(dim=(1, 2))
        i = int(err.argmin())
        if float(err[i]) > self.tol:
            raise KeyError(f"SparseGS: no depth prior for this camera (closest {float(err[i])})")
        return i

    def lookup(self, viewmat, device):
        i = self.index(viewmat)
        key = ("d", i, str(device))
        if key not in self.cache:
            self.cache[key] = self.depth[i].to(device)
        return self.cache[key]

    def image(self, i, device):
        key = ("i", i, str(device))
        if key not in self.cache:
            self.cache[key] = self.images[i].to(device)
        return self.cache[key]


# ------------------------------------------------------------------ warp geometry
def construct_rotation_matrix(axis, theta):
    """scene/generate_new_cam.py construct_rotation_matrix (verbatim numerics)."""
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / np.linalg.norm(axis)
    a = np.cos(theta / 2.0)
    b, c, d = -axis * np.sin(theta / 2.0)
    return np.array([[a * a + b * b - c * c - d * d, 2 * (b * c - a * d), 2 * (b * d + a * c)],
                     [2 * (b * c + a * d), a * a + c * c - b * b - d * d, 2 * (c * d - a * b)],
                     [2 * (b * d - a * c), 2 * (c * d + a * b), a * a + d * d - b * b - c * c]])


def average_up(viewmats):
    """calculate_average_up_vector: mean of the c2w rotations' 2nd column (Inria cam.R)."""
    R = np.stack([np.asarray(v[:3, :3].cpu(), dtype=np.float64).T for v in viewmats])
    up = R[:, :, 1].mean(axis=0)
    return up / np.linalg.norm(up)


def warp_viewmats(viewmats, angles_deg):
    """scene/__init__.py: for d in angles: R_c2w' = M(up, d) R_c2w, w2c translation kept.
    Returns list of (viewmat [4,4] float32, source index), angle-major like the official list."""
    up = average_up(viewmats)
    out = []
    for deg in angles_deg:
        M = construct_rotation_matrix(up, deg * np.pi / 180)
        for i, v in enumerate(viewmats):
            v = np.asarray(v.cpu(), dtype=np.float64)
            R_c2w = v[:3, :3].T
            new = np.eye(4)
            new[:3, :3] = (M @ R_c2w).T
            new[:3, 3] = v[:3, 3]
            out.append((torch.tensor(new, dtype=torch.float32), i))
    return out


def forward_warp(img, depth01, K, w2c_src, w2c_trg, dmin, dmax):
    """scene/cameras.py depth_warping, vectorised. img [H,W,3], depth01 [H,W] (any range; the
    official code min-max normalises it again), K: native pixel-centre intrinsics.
    Returns warped [H,W,3] (zeros where unwritten) and mask [H,W] float."""
    H, W = depth01.shape
    dev = img.device
    d = depth01.double()
    Z = (d - d.min()) / (d.max() - d.min()) * (dmax - dmin) + dmin
    Kd = K.double().clone()
    Kd[0, 2] -= 0.5  # official K: cx = (W-1)/2 on integer pixel coordinates
    Kd[1, 2] -= 0.5
    y, x = torch.meshgrid(torch.arange(H, device=dev, dtype=torch.float64),
                          torch.arange(W, device=dev, dtype=torch.float64), indexing="ij")
    pix = torch.stack([x, y, torch.ones_like(x)], dim=-1).reshape(-1, 3)
    P_A = (pix @ torch.linalg.inv(Kd).T) * Z.reshape(-1, 1)
    RA, TA = w2c_src[:3, :3].double(), w2c_src[:3, 3].double()
    RB, TB = w2c_trg[:3, :3].double(), w2c_trg[:3, 3].double()
    R_AB = RB @ torch.linalg.inv(RA)
    T_AB = TB - R_AB @ TA
    P_B = P_A @ R_AB.T + T_AB
    h = P_B @ Kd.T
    xb = torch.round(h[:, 0] / h[:, 2])  # round half to even, like Python round()
    yb = torch.round(h[:, 1] / h[:, 2])
    ok = torch.isfinite(xb) & torch.isfinite(yb) & (xb >= 0) & (xb < W) & (yb >= 0) & (yb < H)
    src = torch.nonzero(ok).squeeze(1)
    tgt = (yb[ok] * W + xb[ok]).long()
    last = torch.full((H * W,), -1, dtype=torch.long, device=dev)
    last.scatter_reduce_(0, tgt, src, reduce="amax")  # raster-order loop: last write wins
    written = last >= 0
    warped = torch.zeros(H * W, 3, dtype=img.dtype, device=dev)
    warped[written] = img.reshape(-1, 3)[last[written]]
    return warped.reshape(H, W, 3), written.reshape(H, W).to(img.dtype)


# ------------------------------------------------------------------ ellipse sampling (SDS)
def _normalize(x):
    return x / np.linalg.norm(x)


def _viewmatrix(lookdir, up, position):
    vec2 = _normalize(lookdir)
    vec0 = _normalize(np.cross(up, vec2))
    vec1 = _normalize(np.cross(vec2, vec0))
    return np.stack([vec0, vec1, vec2, position], axis=1)


def _pad(p):
    bottom = np.broadcast_to([0, 0, 0, 1.0], p[..., :1, :4].shape)
    return np.concatenate([p[..., :3, :4], bottom], axis=-2)


def transform_poses_pca(poses):
    """utils/pose_utils.py transform_poses_pca (verbatim)."""
    t = poses[:, :3, 3]
    t_mean = t.mean(axis=0)
    t = t - t_mean
    eigval, eigvec = np.linalg.eig(t.T @ t)
    inds = np.argsort(eigval)[::-1]
    eigvec = eigvec[:, inds]
    rot = eigvec.T
    if np.linalg.det(rot) < 0:
        rot = np.diag(np.array([1, 1, -1])) @ rot
    transform = np.concatenate([rot, rot @ -t_mean[:, None]], -1)
    poses_recentered = (transform @ _pad(poses))[..., :3, :4]
    transform = np.concatenate([transform, np.eye(4)[3:]], axis=0)
    if poses_recentered.mean(axis=0)[2, 1] < 0:
        poses_recentered = np.diag(np.array([1, -1, -1])) @ poses_recentered
        transform = np.diag(np.array([1, -1, -1, 1])) @ transform
    scale_factor = 1.0 / np.max(np.abs(poses_recentered[:, :3, 3]))
    poses_recentered[:, :3, 3] *= scale_factor
    transform = np.diag(np.array([scale_factor] * 3 + [1])) @ transform
    return poses_recentered, transform


def focus_point_fn(poses):
    directions, origins = poses[:, :3, 2:3], poses[:, :3, 3:4]
    m = np.eye(3) - directions * np.transpose(directions, [0, 2, 1])
    mt_m = np.transpose(m, [0, 2, 1]) @ m
    return np.linalg.inv(mt_m.mean(0)) @ (mt_m @ origins).mean(0)[:, 0]


class Ellipse:
    """setup_ellipse_sampling on the training cameras + getRandEllipsePose(idx, std=0, z=0)."""

    def __init__(self, viewmats):
        poses = []
        for v in viewmats:
            c2w = np.linalg.inv(np.asarray(v.cpu(), dtype=np.float64))  # OpenCV c2w
            c2w[:, 1:3] *= -1  # -> OpenGL (official tmp_view[:, 1:3] *= -1)
            poses.append(c2w)
        poses, self.transform = transform_poses_pca(np.stack(poses))
        self.center = focus_point_fn(poses)
        offset = np.array([self.center[0], self.center[1], 0])
        sc = np.percentile(np.abs(poses[:, :3, 3] - offset), 90, axis=0)
        self.low, self.high = -sc + offset, sc + offset
        self.z_low = np.percentile(poses[:, :3, 3], 10, axis=0)
        self.z_high = np.percentile(poses[:, :3, 3], 90, axis=0)
        avg_up = poses[:, :3, 1].mean(0)
        avg_up = avg_up / np.linalg.norm(avg_up)
        ind_up = np.argmax(np.abs(avg_up))
        self.up = np.eye(3)[ind_up] * np.sign(avg_up[ind_up])

    def official_w2c(self, rng):
        z = rng.uniform(self.z_low[2], self.z_high[2])
        theta = rng.uniform(0, 2 * np.pi)
        x = self.low[0] + (self.high - self.low)[0] * (np.cos(theta) * 0.5 + 0.5)
        y = self.low[1] + (self.high - self.low)[1] * (np.sin(theta) * 0.5 + 0.5)
        position = np.stack([x, y, z], -1)
        pose = np.eye(4)
        pose[:3] = _viewmatrix(position - self.center, self.up, position)
        pose = np.linalg.inv(self.transform) @ pose
        pose[:3, 1:3] *= -1
        return np.linalg.inv(pose), position

    def sample(self, rng):
        """Official pose with its rotation orthonormalised (the PCA scale only rescales camera
        coordinates uniformly; projection and SH directions are unchanged)."""
        w2c, _ = self.official_w2c(rng)
        R, t = w2c[:3, :3], w2c[:3, 3]
        center = -np.linalg.solve(R, t)
        s = np.cbrt(np.linalg.det(R))
        Ro = R / s
        out = np.eye(4)
        out[:3, :3] = Ro
        out[:3, 3] = -Ro @ center
        return torch.tensor(out, dtype=torch.float32)


# ------------------------------------------------------------------ shared context
class Context:
    def __init__(self):
        self.reset()

    def reset(self):
        self.cameras = None
        self.table = None
        self.draw_cfg = {"sds_freq": 0.1, "sds_enabled": False, "sds_start": math.inf,
                         "warp_prob": 0.4, "warp_enabled": False, "warp_start": math.inf}
        self.rng = random.Random(0)
        self.np_rng = np.random.default_rng(0)
        self._draw = (None, (False, False))
        self.warp_cams = None  # list of dicts: camera, target, mask, src
        self.warp_stack = []
        self.warp_info = None  # (step, info) of the warp render, for the density
        self.ellipse = None

    def seed(self, seed):
        self.rng = random.Random(int(seed))
        self.np_rng = np.random.default_rng(int(seed))

    def frames(self):
        if self.table is None:
            self.table = FrameTable.load(data_root())
        return self.table

    def draws(self, step):
        """(pick_sds, pick_warp) of this step, drawn once in the official order."""
        if self._draw[0] != step:
            c = self.draw_cfg
            a = self.rng.randint(1, 100) <= c["sds_freq"] * 100
            b = self.rng.randint(1, 10) <= c["warp_prob"] * 10
            self._draw = (step, (bool(a and c["sds_enabled"] and step > c["sds_start"]),
                                 bool(b and c["warp_enabled"] and step > c["warp_start"])))
        return self._draw[1]

    def next_warp(self):
        if not self.warp_stack:
            self.warp_stack = list(range(len(self.warp_cams)))
        return self.warp_cams[self.warp_stack.pop(self.rng.randint(0, len(self.warp_stack) - 1))]


CTX = Context()
