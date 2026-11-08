"""Pseudo-view poses for forward-facing scenes (CoR-GS Eq. 4, "following FSGS").

CoR-GS `utils/pose_utils.py:320 generate_random_poses_llff` is byte-identical to FSGS's; this
module is copied from the FSGS plugin (reproductions/fsgs/plugin/ss_fsgs/pseudo.py + its
official LLFF bounds table).


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


# LLFF poses_bounds.npy near/far of the 3 FSGS training views (min near / max far over the views,
# the only quantities generate_random_poses_llff uses), from ~/datasets/llff/fewview3-r8/<s>/
# prep.json; matched to the scene by the training-camera centres (c2w translation, world units).
SCENE_BOUNDS = {
    "fern": {"near": 17.542520113884287, "far": 78.71925302875819, "train_centers": [
        [-1.658869, -1.609145, -0.242867], [-4.766545, 0.720178, -0.17121],
        [-3.881079, 1.736659, 0.7926]]},
    "flower": {"near": 22.656267550734345, "far": 162.94281555616587, "train_centers": [
        [-4.323173, -2.258927, 1.372147], [-4.035961, 0.720233, -0.023411],
        [-4.668914, 3.036076, -1.819622]]},
    "fortress": {"near": 13.472437825125102, "far": 128.38342286860328, "train_centers": [
        [-2.343452, -2.985312, 1.402754], [-0.596759, 0.169303, -0.406764],
        [4.019429, 2.461339, 0.372861]]},
    "horns": {"near": 12.473701480698228, "far": 168.81094102559487, "train_centers": [
        [-3.77684, -3.147624, -1.43279], [3.351668, 0.198719, 0.574782],
        [3.00166, 2.723918, 0.929056]]},
    "leaves": {"near": 35.262850166998774, "far": 449.9130266665557, "train_centers": [
        [-3.326046, -2.317046, -1.206822], [-1.263424, 0.250338, 0.229431],
        [6.435536, 2.427958, 0.479507]]},
    "orchids": {"near": 15.062411908163723, "far": 96.9429995809968, "train_centers": [
        [-1.975985, -3.74603, -0.495709], [0.042977, 0.292588, -0.050057],
        [0.474495, 3.573621, 0.663743]]},
    "room": {"near": 10.729799505777695, "far": 48.929631822500625, "train_centers": [
        [-3.619974, -3.791045, 0.452161], [0.16049, 2.171683, -0.405748],
        [-0.337692, 1.516422, 0.225718]]},
    "trex": {"near": 26.506352452490084, "far": 112.93942182214286, "train_centers": [
        [-1.90164, -2.832524, 0.605863], [-0.239929, 1.547036, -0.79153],
        [3.713924, 1.138788, -0.674476]]},
}


def official_bounds(table, viewmats, tol=1e-3):
    """Return (scene, [[near, far]]) whose recorded training-camera centres match these views."""
    centers = np.stack([np.linalg.inv(v)[:3, 3] for v in viewmats])
    for scene, entry in table.items():
        ref = np.asarray(entry["train_centers"], dtype=np.float64)
        if ref.shape == centers.shape and np.abs(ref - centers).max() < tol:
            return scene, np.array([[float(entry["near"]), float(entry["far"])]])
    raise ValueError("pseudo_bounds='official': no scene_bounds entry matches the training "
                     "cameras (add the scene's poses_bounds near/far or set pseudo_bounds='points')")
