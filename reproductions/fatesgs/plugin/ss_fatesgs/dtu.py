"""Mesh extraction (official render.py / utils/mesh_utils.py, bounded TSDF) and DTU Chamfer
evaluation (official eval/eval_dtu: mask culling + DTUeval-python, MIT) for a finished ss-train
run.

Needs open3d, scikit-learn, trimesh, scikit-image (not in the splatstudio env): put them on
PYTHONPATH, e.g. ``PYTHONPATH=~/models/dtueval-pylib``.

usage:
  fatesgs-dtu-eval --run RUN --data NATIVE_SCENE --dtu ~/datasets/dtu_official/gt [--scan 24]
writes RUN/mesh/fuse.ply, fuse_post.ply, culled_mesh.ply and RUN/dtu_cd.json
"""

import argparse
import json
import multiprocessing as mp
import re
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

TRAIN_VIEWS_LARGE = [23, 24, 33]


# ------------------------------------------------------------------ mesh (render.py)
def focus_point_fn(poses):
    directions, origins = poses[:, :3, 2:3], poses[:, :3, 3:4]
    m = np.eye(3) - directions * np.transpose(directions, [0, 2, 1])
    mt_m = np.transpose(m, [0, 2, 1]) @ m
    return np.linalg.inv(mt_m.mean(0)) @ (mt_m @ origins).mean(0)[:, 0]


def post_process_mesh(mesh, cluster_to_keep=50):
    import copy

    import open3d as o3d
    mesh_0 = copy.deepcopy(mesh)
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Error):
        tri_clusters, cluster_n, _ = mesh_0.cluster_connected_triangles()
    tri_clusters, cluster_n = np.asarray(tri_clusters), np.asarray(cluster_n)
    n_cluster = np.sort(cluster_n.copy())[-cluster_to_keep] if len(cluster_n) >= cluster_to_keep \
        else np.sort(cluster_n.copy())[0]
    n_cluster = max(n_cluster, 50)
    mesh_0.remove_triangles_by_mask(cluster_n[tri_clusters] < n_cluster)
    mesh_0.remove_unreferenced_vertices()
    mesh_0.remove_degenerate_triangles()
    return mesh_0


@torch.no_grad()
def extract_mesh(run, data, device="cuda", mesh_res=1024, num_cluster=50):
    import open3d as o3d
    from splatstudio.data import load_dataset, on_device

    from . import sidecar
    from .renderer import Renderer

    ckpt = torch.load(Path(run) / "checkpoint.pt", map_location=device, weights_only=False)
    params = {k: v.to(device) for k, v in ckpt["params"].items()}
    ds = load_dataset(data, 1)
    tab = sidecar.table(data)
    renderer = Renderer(type("S", (), {"config": {}})())
    frames = [on_device(ds["frames"][n], device) for n in ds["splits"]["train"]]
    rgbs, depths = [], []
    for fr in frames:
        out = renderer.render(params, fr, sh_degree=0)  # official: active_sh_degree = 0
        rgbs.append(out["rgb"].clamp(0, 1).cpu())
        d = out["depth"][..., 0].clone()
        d[tab.alpha[tab.index(fr["viewmat"])].to(device) < 0.5] = 0  # mask_backgrond
        depths.append(d.cpu())
    c2ws = np.stack([np.linalg.inv(fr["viewmat"].cpu().numpy()) for fr in frames])
    poses = c2ws[:, :3, :] @ np.diag([1, -1, -1, 1])
    center = focus_point_fn(poses)
    radius = float(np.linalg.norm(c2ws[:, :3, 3] - center, axis=-1).min())
    depth_trunc = radius * 2.0
    voxel = depth_trunc / mesh_res
    sdf_trunc = 5.0 * voxel
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=voxel, sdf_trunc=sdf_trunc,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)
    for fr, rgb, d in zip(frames, rgbs, depths):
        K = fr["K"].cpu().numpy()
        intr = o3d.camera.PinholeCameraIntrinsic(width=fr["width"], height=fr["height"],
                                                 fx=K[0, 0], fy=K[1, 1], cx=K[0, 2], cy=K[1, 2])
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(np.ascontiguousarray((rgb.numpy() * 255).astype(np.uint8))),
            o3d.geometry.Image(np.ascontiguousarray(d.numpy()[..., None].astype(np.float32))),
            depth_trunc=depth_trunc, convert_rgb_to_intensity=False, depth_scale=1.0)
        volume.integrate(rgbd, intrinsic=intr,
                         extrinsic=fr["viewmat"].cpu().numpy().astype(np.float64))
    mesh = volume.extract_triangle_mesh()
    out_dir = Path(run) / "mesh"
    out_dir.mkdir(exist_ok=True)
    o3d.io.write_triangle_mesh(str(out_dir / "fuse.ply"), mesh)
    post = post_process_mesh(mesh, num_cluster)
    o3d.io.write_triangle_mesh(str(out_dir / "fuse_post.ply"), post)
    return out_dir / "fuse_post.ply", {"radius": radius, "voxel": voxel, "sdf_trunc": sdf_trunc,
                                       "depth_trunc": depth_trunc,
                                       "vertices": len(post.vertices)}


# ------------------------------------------------------------------ cull (dtu_eval.py)
def load_K_Rt_from_P(P):
    import cv2
    K, R, t = cv2.decomposeProjectionMatrix(P)[:3]
    K = K / K[2, 2]
    intr = np.eye(4)
    intr[:3, :3] = K
    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = R.transpose()
    pose[:3, 3] = (t[:3] / t[3])[:, 0]
    return intr, pose


def cull_scan(mesh_path, result_file, n_images, instance_dir, device="cuda"):
    import cv2
    import trimesh
    from skimage.morphology import binary_dilation, disk
    cams = np.load(Path(instance_dir) / "cameras.npz")
    scale_mats = [cams[f"scale_mat_{i}"].astype(np.float32) for i in range(n_images)]
    world_mats = [cams[f"world_mat_{i}"].astype(np.float32) for i in range(n_images)]
    intrinsics, poses = [], []
    for s, w in zip(scale_mats, world_mats):
        k, p = load_K_Rt_from_P((w @ s)[:3, :4])
        intrinsics.append(torch.from_numpy(k).float())
        poses.append(torch.from_numpy(p).float())
    masks = [cv2.imread(str(p)) for p in sorted((Path(instance_dir) / "mask").glob("*.png"))]
    W, H = 1600, 1200
    mesh = trimesh.load(mesh_path)
    verts = torch.from_numpy(mesh.vertices).to(device)
    verts = torch.cat((verts, torch.ones_like(verts[:, :1])), dim=-1).permute(1, 0).float()
    sampled = []
    for i in range(n_images):
        w2c = torch.inverse(poses[i]).to(device)
        cam_points = intrinsics[i].to(device) @ w2c @ verts
        pix = (cam_points[:2, :] / (cam_points[2, :].unsqueeze(0) + 1e-6)).permute(1, 0)
        pix[..., 0] /= W - 1
        pix[..., 1] /= H - 1
        pix = (pix - 0.5) * 2
        valid = ((pix > -1.0) & (pix < 1.0)).all(dim=-1).float()
        maski = masks[i][:, :, 0].astype(np.float32) / 256.0
        maski = torch.from_numpy(binary_dilation(maski, disk(24))).float()[None, None].to(device)
        sm = F.grid_sample(maski, pix[None, None], mode="nearest", padding_mode="zeros",
                           align_corners=True)[0, -1, 0]
        sampled.append(sm + (1.0 - valid))
    sampled = torch.stack(sampled, -1)
    mask = (sampled > 0.0).all(dim=-1).cpu().numpy()
    face_mask = mask[mesh.faces].all(axis=1)
    mesh.update_vertices(mask)
    mesh.update_faces(face_mask)
    s = scale_mats[0]
    mesh.vertices = mesh.vertices * s[0, 0] + s[:3, 3][None]
    mesh.export(result_file)


# ------------------------------------------------------------------ DTUeval-python (eval.py)
def _sample_single_tri(input_):
    n1, n2, v1, v2, tri_vert = input_
    c = np.mgrid[: n1 + 1, : n2 + 1]
    c += 0.5
    c[0] /= max(n1, 1e-7)
    c[1] /= max(n2, 1e-7)
    c = np.transpose(c, (1, 2, 0))
    k = c[c.sum(axis=-1) < 1]
    return v1 * k[:, :1] + v2 * k[:, 1:] + tri_vert


def eval_scene(mesh_file, scan, dataset_dir, downsample_density=0.2, patch_size=60,
               max_dist=20):
    import open3d as o3d
    import sklearn.neighbors as skln
    from scipy.io import loadmat
    thresh = downsample_density
    data_mesh = o3d.io.read_triangle_mesh(str(mesh_file))
    vertices = np.asarray(data_mesh.vertices)
    tri_vert = vertices[np.asarray(data_mesh.triangles)]
    v1 = tri_vert[:, 1] - tri_vert[:, 0]
    v2 = tri_vert[:, 2] - tri_vert[:, 0]
    l1 = np.linalg.norm(v1, axis=-1, keepdims=True)
    l2 = np.linalg.norm(v2, axis=-1, keepdims=True)
    area2 = np.linalg.norm(np.cross(v1, v2), axis=-1, keepdims=True)
    nz = (area2 > 0)[:, 0]
    l1, l2, area2, v1, v2, tri_vert = [a[nz] for a in (l1, l2, area2, v1, v2, tri_vert)]
    thr = thresh * np.sqrt(l1 * l2 / area2)
    n1 = np.floor(l1 / thr)
    n2 = np.floor(l2 / thr)
    with mp.get_context("fork").Pool() as pool:
        new_pts = pool.map(_sample_single_tri, ((n1[i, 0], n2[i, 0], v1[i:i + 1], v2[i:i + 1],
                                                 tri_vert[i:i + 1, 0]) for i in range(len(n1))),
                           chunksize=1024)
    data_pcd = np.concatenate([vertices] + new_pts, axis=0)
    np.random.default_rng().shuffle(data_pcd, axis=0)
    nn_engine = skln.NearestNeighbors(n_neighbors=1, radius=thresh, algorithm="kd_tree", n_jobs=-1)
    nn_engine.fit(data_pcd)
    rnn_idxs = nn_engine.radius_neighbors(data_pcd, radius=thresh, return_distance=False)
    mask = np.ones(data_pcd.shape[0], dtype=np.bool_)
    for curr, idxs in enumerate(rnn_idxs):
        if mask[curr]:
            mask[idxs] = 0
            mask[curr] = 1
    data_down = data_pcd[mask]
    obs = loadmat(f"{dataset_dir}/ObsMask/ObsMask{scan}_10.mat")
    ObsMask, BB, Res = [obs[a] for a in ("ObsMask", "BB", "Res")]
    BB = BB.astype(np.float32)
    inbound = ((data_down >= BB[:1] - patch_size) & (data_down < BB[1:] + patch_size * 2)).sum(
        axis=-1) == 3
    data_in = data_down[inbound]
    grid = np.around((data_in - BB[:1]) / Res).astype(np.int32)
    gin = ((grid >= 0) & (grid < np.expand_dims(ObsMask.shape, 0))).sum(axis=-1) == 3
    gi = grid[gin]
    in_obs = ObsMask[gi[:, 0], gi[:, 1], gi[:, 2]].astype(np.bool_)
    data_in_obs = data_in[gin][in_obs]
    stl = np.asarray(o3d.io.read_point_cloud(f"{dataset_dir}/Points/stl/stl{scan:03}_total.ply")
                     .points)
    nn_engine.fit(stl)
    d2s, _ = nn_engine.kneighbors(data_in_obs, n_neighbors=1, return_distance=True)
    mean_d2s = d2s[d2s < max_dist].mean()
    plane = loadmat(f"{dataset_dir}/ObsMask/Plane{scan}.mat")["P"]
    stl_hom = np.concatenate([stl, np.ones_like(stl[:, :1])], -1)
    stl_above = stl[(plane.reshape((1, 4)) * stl_hom).sum(-1) > 0]
    nn_engine.fit(data_in)
    s2d, _ = nn_engine.kneighbors(stl_above, n_neighbors=1, return_distance=True)
    mean_s2d = s2d[s2d < max_dist].mean()
    return float(mean_d2s), float(mean_s2d), float((mean_d2s + mean_s2d) / 2)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--run", required=True, type=Path)
    ap.add_argument("--data", required=True, type=Path)
    ap.add_argument("--dtu", required=True, type=Path, help="dir with ObsMask/ and Points/stl/")
    ap.add_argument("--scan", type=int, default=None, help="default: parsed from --data name")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    scan = args.scan
    if scan is None:
        m = re.search(r"scan(\d+)", args.data.resolve().name)
        scan = int(m.group(1))
    mesh_post, info = extract_mesh(args.run, args.data, args.device)
    culled = Path(args.run) / "mesh" / "culled_mesh.ply"
    cull_scan(str(mesh_post), str(culled), 3, args.data / "dtu_eval", args.device)
    acc, comp, cd = eval_scene(culled, scan, str(args.dtu.expanduser()))
    result = {"scan": scan, "accuracy_mm": acc, "completeness_mm": comp, "chamfer_mm": cd,
              "mesh": info, "protocol": "FatesGS render.py TSDF + eval_dtu (DTUeval-python, "
              "downsample 0.2, patch 60, max_dist 20), mask culling with views 23,24,33"}
    (Path(args.run) / "dtu_cd.json").write_text(json.dumps(result, indent=1))
    print(json.dumps(result))


if __name__ == "__main__":
    main()
