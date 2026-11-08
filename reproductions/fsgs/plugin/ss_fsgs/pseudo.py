"""Pseudo-view poses for forward-facing scenes (FSGS C2, paper Eq. 5).

Port of `generate_random_poses_llff` from the official FSGS code (utils/pose_utils.py:207-254,
itself from RegNeRF); copied from the AD-GS plugin (reproductions/ad-gs, same function in the
CoR-GS/AD-GS forks of FSGS). Inputs/outputs are OpenCV world-to-camera matrices (SplatStudio
`viewmat`). The LLFF near/far bounds the official code reads from `poses_bounds.npy`
(`view.bounds`) are passed in explicitly.
"""

import numpy as np


def normalize(x):
    return x / np.linalg.norm(x)


def viewmatrix(lookdir, up, position):
    vec2 = normalize(lookdir)
    vec0 = normalize(np.cross(up, vec2))
    vec1 = normalize(np.cross(vec2, vec0))
    return np.stack([vec0, vec1, vec2, position], axis=1)


def poses_avg(poses):
    position = poses[:, :3, 3].mean(0)
    z_axis = poses[:, :3, 2].mean(0)
    up = poses[:, :3, 1].mean(0)
    return viewmatrix(z_axis, up, position)


def pad_poses(p):
    bottom = np.broadcast_to([0, 0, 0, 1.0], p[..., :1, :4].shape)
    return np.concatenate([p[..., :3, :4], bottom], axis=-2)


def recenter_poses(poses):
    cam2world = poses_avg(poses)
    transform = np.linalg.inv(pad_poses(cam2world))
    poses = transform @ pad_poses(poses)
    return poses[..., :3, :4], transform


def depth_bounds(points, viewmats, Ks, sizes, low=0.1, high=99.9):
    """LLFF imgs2poses bounds: per view, percentiles of camera depth of the visible points."""
    bounds = []
    for w2c, K, (w, h) in zip(viewmats, Ks, sizes):
        x = points @ w2c[:3, :3].T + w2c[:3, 3]
        z = x[:, 2]
        front = z > 1e-8
        u = x[front, 0] / z[front] * K[0, 0] + K[0, 2]
        v = x[front, 1] / z[front] * K[1, 1] + K[1, 2]
        inside = (u >= 0) & (u < w) & (v >= 0) & (v < h)
        zs = z[front][inside]
        if len(zs) < 2:
            raise ValueError("depth_bounds: fewer than two initial points visible in a training view")
        bounds.append([np.percentile(zs, low), np.percentile(zs, high)])
    return np.asarray(bounds)


def random_poses_llff(viewmats, bounds, n_poses, rng):
    """Returns [n_poses, 4, 4] OpenCV world-to-camera matrices (official pose_utils.py:320-366)."""
    poses = []
    for w2c in viewmats:
        c2w = np.linalg.inv(np.asarray(w2c, dtype=np.float64))
        c2w[:, 1:3] *= -1  # OpenCV -> OpenGL (tmp_view[:, 1:3] *= -1)
        poses.append(c2w)
    poses = np.stack(poses, 0)
    bounds = np.asarray(bounds, dtype=np.float64).copy()
    scale = 1.0 / (bounds.min() * 0.75)
    poses[:, :3, 3] *= scale
    bounds *= scale
    poses, transform = recenter_poses(poses)
    close_depth, inf_depth = bounds.min() * 0.9, bounds.max() * 5.0
    dt = 0.75
    focal = 1 / ((1 - dt) / close_depth + dt / inf_depth)
    positions = poses[:, :3, 3]
    radii = np.percentile(np.abs(positions), 100, 0)
    radii = np.concatenate([radii, [1.0]])
    cam2world = poses_avg(poses)
    up = poses[:, :3, 1].mean(0)
    out = []
    for _ in range(n_poses):
        t = radii * np.concatenate([2 * rng.random(3) - 1.0, [1]])
        position = cam2world @ t
        lookat = cam2world @ [0, 0, -focal, 1.0]
        z_axis = position - lookat
        pose = np.eye(4)
        pose[:3] = viewmatrix(z_axis, up, position)
        pose = np.linalg.inv(transform) @ pose
        pose[:3, 1:3] *= -1
        pose[:3, 3] /= scale
        out.append(np.linalg.inv(pose))
    return np.stack(out, 0), {"focus_distance": float(focal / scale), "scale": float(scale)}
