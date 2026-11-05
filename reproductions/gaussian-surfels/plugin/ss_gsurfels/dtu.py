"""Mesh extraction (volumetric cutting + screened Poisson, mesh.py) and DTU Chamfer distance
(official eval.py eval_dtu = DTUeval-python, MIT) for a finished ss-train run.

Needs pymeshlab (~/models/gsurfels-pylib) and open3d / scikit-learn (~/models/dtueval-pylib);
both directories are appended to sys.path when the imports fail, or pass them on PYTHONPATH.

usage:
  gsurfels-dtu-eval --run RUN --data NATIVE_SCENE --dtu ~/datasets/dtu_official/gt [--no-cut]
writes RUN/mesh/poisson_mesh_10_{plain,pruned}.ply and RUN/dtu_cd.json
"""

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import numpy as np

EVAL_PYLIB = Path(os.environ.get("DTUEVAL_PYLIB", "~/models/dtueval-pylib")).expanduser()


def _ensure_eval_libs():
    try:
        import open3d  # noqa: F401
        import sklearn  # noqa: F401
    except ImportError:
        if str(EVAL_PYLIB) not in sys.path:
            sys.path.append(str(EVAL_PYLIB))


def sample_single_tri(input_):
    n1, n2, v1, v2, tri_vert = input_
    c = np.mgrid[:n1 + 1, :n2 + 1]
    c += 0.5
    c[0] /= max(n1, 1e-7)
    c[1] /= max(n2, 1e-7)
    c = np.transpose(c, (1, 2, 0))
    k = c[c.sum(axis=-1) < 1]
    return v1 * k[:, :1] + v2 * k[:, 1:] + tri_vert


def eval_mesh(in_file, scene, dataset_dir, S, T):
    """official eval.py eval(): DTUeval-python with ObsMask / Plane, vertices / S + T."""
    _ensure_eval_libs()
    import open3d as o3d
    import sklearn.neighbors as skln
    from scipy.io import loadmat

    data_mesh = o3d.io.read_triangle_mesh(str(in_file))
    data_mesh.remove_unreferenced_vertices()
    max_dist, patch, thresh = 20, 60, 0.2
    vertices = np.asarray(data_mesh.vertices) / S + T
    triangles = np.asarray(data_mesh.triangles)
    tri_vert = vertices[triangles]
    v1 = tri_vert[:, 1] - tri_vert[:, 0]
    v2 = tri_vert[:, 2] - tri_vert[:, 0]
    l1 = np.linalg.norm(v1, axis=-1, keepdims=True)
    l2 = np.linalg.norm(v2, axis=-1, keepdims=True)
    area2 = np.linalg.norm(np.cross(v1, v2), axis=-1, keepdims=True)
    nz = (area2 > 0)[:, 0]
    l1, l2, area2, v1, v2, tri_vert = [a[nz] for a in [l1, l2, area2, v1, v2, tri_vert]]
    thr = thresh * np.sqrt(l1 * l2 / area2)
    n1 = np.floor(l1 / thr)
    n2 = np.floor(l2 / thr)
    with mp.get_context("fork").Pool(min(32, os.cpu_count())) as pool:
        new_pts = pool.map(sample_single_tri,
                           ((n1[i, 0], n2[i, 0], v1[i:i + 1], v2[i:i + 1], tri_vert[i:i + 1, 0])
                            for i in range(len(n1))), chunksize=1024)
    data_pcd = np.concatenate([vertices, np.concatenate(new_pts, axis=0)], axis=0)
    np.random.default_rng(0).shuffle(data_pcd, axis=0)
    nn_engine = skln.NearestNeighbors(n_neighbors=1, radius=thresh, algorithm="kd_tree",
                                      n_jobs=16)
    nn_engine.fit(data_pcd)
    rnn_idxs = nn_engine.radius_neighbors(data_pcd, radius=thresh, return_distance=False)
    mask = np.ones(data_pcd.shape[0], dtype=np.bool_)
    for curr, idxs in enumerate(rnn_idxs):
        if mask[curr]:
            mask[idxs] = 0
            mask[curr] = 1
    data_down = data_pcd[mask]
    obs = loadmat(f"{dataset_dir}/ObsMask/ObsMask{scene}_10.mat")
    ObsMask, BB, Res = [obs[a] for a in ["ObsMask", "BB", "Res"]]
    BB = BB.astype(np.float32)
    inbound = ((data_down >= BB[:1] - patch) & (data_down < BB[1:] + patch * 2)).sum(-1) == 3
    data_in = data_down[inbound]
    data_grid = np.around((data_in - BB[:1]) / Res).astype(np.int32)
    grid_in = ((data_grid >= 0) & (data_grid < np.expand_dims(ObsMask.shape, 0))).sum(-1) == 3
    dgi = data_grid[grid_in]
    in_obs = ObsMask[dgi[:, 0], dgi[:, 1], dgi[:, 2]].astype(np.bool_)
    data_in_obs = data_in[grid_in][in_obs]
    stl = np.asarray(o3d.io.read_point_cloud(
        f"{dataset_dir}/Points/stl/stl{scene:03}_total.ply").points)
    nn_engine.fit(stl)
    dist_d2s, _ = nn_engine.kneighbors(data_in_obs, n_neighbors=1, return_distance=True)
    mean_d2s = dist_d2s[dist_d2s < max_dist].mean()
    plane = loadmat(f"{dataset_dir}/ObsMask/Plane{scene}.mat")["P"]
    stl_hom = np.concatenate([stl, np.ones_like(stl[:, :1])], -1)
    stl_above = stl[(plane.reshape((1, 4)) * stl_hom).sum(-1) > 0]
    nn_engine.fit(data_in)
    dist_s2d, _ = nn_engine.kneighbors(stl_above, n_neighbors=1, return_distance=True)
    mean_s2d = dist_s2d[dist_s2d < max_dist].mean()
    return {"accuracy": float(mean_d2s), "completeness": float(mean_s2d),
            "chamfer": float((mean_d2s + mean_s2d) / 2)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--dtu", type=Path, default=Path("~/datasets/dtu_official/gt").expanduser())
    ap.add_argument("--scan", type=int, default=None)
    ap.add_argument("--poisson-depth", type=int, default=10)
    ap.add_argument("--no-cut", action="store_true", help="ablation: skip volumetric cutting")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    from .mesh import extract

    meta = json.loads((a.data / "transforms.json").read_text())
    scan = a.scan if a.scan is not None else int(meta["dtu_scan"])
    t0 = time.time()
    mesh_path, stats = extract(a.run, a.data, a.device, a.poisson_depth, cut=not a.no_cut)
    t1 = time.time()
    scale_mat = np.load(a.data / "cameras.npz")["scale_mat_0"]
    S = np.linalg.inv(scale_mat[:3, :3])[0][0]
    T = scale_mat[:3, 3:].T
    cd = eval_mesh(mesh_path, scan, a.dtu, S, T)
    res = {"scan": scan, "mesh": str(mesh_path), "chamfer_mm": cd["chamfer"], **cd,
           "mesh_stats": stats, "mesh_seconds": t1 - t0, "eval_seconds": time.time() - t1}
    name = "dtu_cd_nocut.json" if a.no_cut else "dtu_cd.json"
    (a.run / name).write_text(json.dumps(res, indent=2))
    print(json.dumps(res))


if __name__ == "__main__":
    main()
