"""AGS-Mesh Depth-Normal Consistency (paper Sec. 4.1, Eq. 1-2) for one frame, replicating the
official MuSHRoom data path (dn-splatter @ 97588b4 scripts/depth_to_normal.py:DepthToNormal.main,
called by data/mushroom_dataparser.py:180-192 with --load-depth-confidence-masks True).
Used offline by reproductions/ags-mesh/tools/prepare_dnc.py and by the dnc-filter probe.

open3d is not part of the splatstudio env; ``import_open3d`` falls back to the reproduction
pylib (``AGSMESH_PYLIB``, default ~/models/dtueval-pylib) and raises if it is unavailable.
"""

import os
import sys
from pathlib import Path

import numpy as np

OPENGL_TO_OPENCV = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]], float)


def import_open3d():
    try:
        import open3d as o3d
    except ImportError:
        lib = os.environ.get("AGSMESH_PYLIB", str(Path.home() / "models/dtueval-pylib"))
        if lib not in sys.path:
            sys.path.append(lib)
        import open3d as o3d
    return o3d


def knn_normals(points, knn=200):
    """Open3D estimate_normals(KDTreeSearchParamKNN(knn)) (unoriented covariance eigenvector)."""
    o3d = import_open3d()
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.asarray(points, dtype=np.float64))
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=knn))
    return np.asarray(pcd.normals)


def backproject(depth, fx, fy, cx, cy, c2w):
    """depth_to_normal.py:backproject (x/y with +0.5 pixel offset, rows of R^-1 on the right)."""
    H, W = depth.shape
    u, v = np.meshgrid(np.arange(W), np.arange(H), indexing="xy")
    u = (u.reshape(-1) + 0.5).astype(np.float32)
    v = (v.reshape(-1) + 0.5).astype(np.float32)
    d = depth.reshape(-1)
    pts = np.zeros((H * W, 3), dtype=np.float32)
    pts[:, 0] = (u - cx) * d / fx
    pts[:, 1] = (v - cy) * d / fy
    pts[:, 2] = d
    return pts @ np.linalg.inv(c2w[:3, :3]) + c2w[:3, 3]


def angle_deg(a, b):
    """depth_to_normal.py:compute_angle_between_normals."""
    a = a / np.linalg.norm(a, axis=2, keepdims=True)
    b = b / np.linalg.norm(b, axis=2, keepdims=True)
    return np.degrees(np.arccos(np.clip(np.sum(a * b, axis=2), -1.0, 1.0)))


def dnc_frame(depth_m, normal_u8, c2w_gl, intr, knn=200, tau=10.0, angle_space="encoded",
              jpeg=True, return_debug=False):
    """One frame of DepthToNormal.main. Returns the decoded uint8 mask (255 = rejected)."""
    import cv2

    fx, fy, cx, cy = intr
    H, W = depth_m.shape
    c2w = np.asarray(c2w_gl, dtype=np.float64) @ OPENGL_TO_OPENCV
    pts = backproject(depth_m.astype(np.float32), fx, fy, cx, cy, c2w)
    nd = knn_normals(pts, knn)
    ray = pts - c2w[:3, 3].reshape(1, 3)
    wrong = (ray * nd).sum(-1) > 0
    nd[wrong] = -nd[wrong]
    nd = nd.reshape(H, W, 3)
    npr = normal_u8.astype(np.float64) / 255.0
    R = np.linalg.inv(c2w)[:3, :3].T
    npw = (R @ ((npr.reshape(-1, 3).T - 0.5) * 2)).T
    npw = (npw / np.linalg.norm(npw, axis=1, keepdims=True)).reshape(H, W, 3)
    if angle_space == "encoded":
        theta = angle_deg((nd + 1) / 2, npw * 0.5 + 0.5)
    elif angle_space == "signed":
        theta = angle_deg(nd, npw)
    else:
        raise ValueError(angle_space)
    mask = (theta > tau).astype(np.uint8) * 255
    if jpeg:
        ok, buf = cv2.imencode(".jpg", mask.astype(np.float64))
        assert ok
        mask = cv2.imdecode(buf, cv2.IMREAD_ANYDEPTH)
    if return_debug:
        return mask, {"theta": theta, "normal_depth": nd, "normal_prior_world": npw}
    return mask
