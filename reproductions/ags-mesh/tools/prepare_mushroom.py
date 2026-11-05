"""Build a native SplatStudio scene + DN-Splatter sidecar from a MuSHRoom iPhone room.

Replicates the official DN-Splatter data path (github.com/maturk/dn-splatter @ 97588b4):

* splits (data/mushroom_dataparser.py): train = long_capture frames not in test.txt;
  test  = long_capture frames listed in test.txt ("within"/"left" protocol of paper Table 3);
  validation = all short_capture frames ("with"/"right" protocol: different trajectory).
  The official default eval_mode "all" is the union of both; per-view rows let us report it.
* poses: transformations_colmap.json (globally optimised COLMAP), orientation/centering none,
  no auto scale (metric); OpenGL c2w in transforms.json (native loader converts to OpenCV).
* init (mushroom_utils/pointcloud_utils.py:generate_iPhone_pointcloud_within_sequence):
  incremental Open3D ScalableTSDF (voxel 0.04, trunc 0.2, depth_trunc 4.0) over the training
  frames in json order, after every frame sample samples_per_frame points of the current
  cloud; 1M points exactly.  Normals: Open3D estimate_normals(radius 0.1, max_nn 30).
  Scales/quats (dn_model.py populate_modules): log(mean 3-NN distance) on x/y, log(mean/10)
  on z; quaternion rotating +z onto the point normal (Rodrigues, dn_model.py:1520).
* priors: sensor depth png (mm, uint16, already image resolution) and Omnidata v2 DPT-hybrid
  normals (scripts/normals_from_pretrain.py:run_monocular_normals, "low"): ToTensor image,
  bilinear resize to 384x384 (no antialias), clamp(0,1), resize back, png = uint8(255 x).

Run (open3d/sklearn from ~/models/dtueval-pylib, timm 0.6.12 from ~/models/dpt/pylib,
omnidata_tools from ~/models/omnidata/pylib):

  PYTHONPATH=$HOME/models/dtueval-pylib:$HOME/models/dpt/pylib:$HOME/models/omnidata/pylib \
  CUDA_VISIBLE_DEVICES=<gpu> tools/ssenv python reproductions/dn-splatter/tools/prepare_mushroom.py --room vr_room
"""

import argparse
import hashlib
import json
import os
import random
from pathlib import Path

import numpy as np

RAW = Path.home() / "datasets/mushroom/raw"
NATIVE = Path.home() / "datasets/mushroom/native"
OMNI_CKPT = Path.home() / "models/omnidata/omnidata_dpt_normal_v2.ckpt"
OPENGL_TO_OPENCV = np.array(
    [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]], dtype=np.float64
)


def stem(frame):
    return Path(frame["file_path"]).stem


def load_meta(d):
    return json.loads((d / "transformations_colmap.json").read_text())


def tsdf_pointcloud(long_dir, meta, test_names, num_points, seed):
    """pointcloud_utils.py:86-196 (json frame order, cumulative TSDF, per-frame sampling)."""
    import cv2
    import open3d as o3d

    random.seed(seed)
    np.random.seed(seed)
    frames = meta["frames"]
    names = [stem(f) for f in frames]
    i_eval = [names.index(n) for n in test_names if n in names]
    i_train = np.setdiff1d(np.arange(len(frames)), i_eval)
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=0.04, sdf_trunc=0.2,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
    )
    fx, fy, cx, cy = (float(meta[k]) for k in ("fl_x", "fl_y", "cx", "cy"))
    H, W = int(meta["h"]), int(meta["w"])
    intr = o3d.camera.PinholeCameraIntrinsic(W, H, fx, fy, cx, cy)
    spf = (num_points + len(i_train)) // len(i_train)
    pts, cols = [], []
    for k, item in enumerate(i_train):
        fr = frames[item]
        color = cv2.cvtColor(cv2.imread(str(long_dir / fr["file_path"])), cv2.COLOR_BGR2RGB)
        pose = np.array(fr["transform_matrix"]) @ OPENGL_TO_OPENCV
        depth = cv2.imread(str(long_dir / fr["depth_file_path"]), cv2.IMREAD_ANYDEPTH)
        depth = cv2.resize(depth, (W, H))
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(color), o3d.geometry.Image(depth), depth_trunc=4.0,
            convert_rgb_to_intensity=False)
        volume.integrate(rgbd, intr, np.linalg.inv(pose))
        pcd = volume.extract_point_cloud()
        spf = min(spf, len(pcd.points))  # official: the reduction persists (variable reused)
        mask = np.asarray(random.sample(range(len(pcd.points)), spf))
        cols.append(np.asarray(pcd.colors)[mask])
        pts.append(np.asarray(pcd.points)[mask])
        if k % 50 == 0:
            print(f"tsdf {k}/{len(i_train)} cloud={len(pcd.points)}", flush=True)
    points, colors = np.vstack(pts), np.vstack(cols)
    if points.shape[0] > num_points:
        idx = np.random.choice(points.shape[0], size=num_points, replace=False)
        points, colors = points[idx], colors[idx]
    # official round trip through iphone_pointcloud.ply (open3d stores float64 xyz, rgb as uint8)
    colors = np.round(colors * 255).astype(np.uint8)
    return points.astype(np.float64), colors


def estimate_normals(points):
    """mushroom_dataparser.py:_load_points3D_normals (transform_matrix is identity here)."""
    import open3d as o3d

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    pcd.normalize_normals()
    return np.asarray(pcd.normals, dtype=np.float32)


def rotate_z_to(n):
    """dn_model.py:rotate_vector_to_vector([0,0,1], n) + matrix_to_quaternion (wxyz)."""
    import torch

    n = torch.from_numpy(n).double()
    N = n.shape[0]
    u = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64).expand(N, 3)
    Ru = n / n.norm(dim=-1, keepdim=True)
    I3 = torch.eye(3, dtype=torch.float64).expand(N, 3, 3)
    c = (u * Ru).sum(-1)
    K = Ru[:, :, None] * u[:, None, :] - u[:, :, None] * Ru[:, None, :]
    R = I3 + K + (K @ K) / (1 + c)[:, None, None]
    same = (c - 1.0).abs() < 1e-10
    opp = (c + 1.0).abs() < 1e-10
    R[same] = torch.eye(3, dtype=torch.float64)
    R[opp] = -torch.eye(3, dtype=torch.float64)
    m = R
    tr = m[:, 0, 0] + m[:, 1, 1] + m[:, 2, 2]
    q = torch.zeros(N, 4, dtype=torch.float64)
    c0 = tr > 0
    c1 = (~c0) & (m[:, 0, 0] > m[:, 1, 1]) & (m[:, 0, 0] > m[:, 2, 2])
    c2 = (~c0) & (~c1) & (m[:, 1, 1] > m[:, 2, 2])
    c3 = ~(c0 | c1 | c2)
    S = torch.sqrt((tr + 1.0).clamp_min(0)) * 2
    q[c0] = torch.stack([0.25 * S, (m[:, 2, 1] - m[:, 1, 2]) / S, (m[:, 0, 2] - m[:, 2, 0]) / S,
                         (m[:, 1, 0] - m[:, 0, 1]) / S], -1)[c0]
    S = torch.sqrt((1.0 + m[:, 0, 0] - m[:, 1, 1] - m[:, 2, 2]).clamp_min(0)) * 2
    q[c1] = torch.stack([(m[:, 2, 1] - m[:, 1, 2]) / S, 0.25 * S, (m[:, 0, 1] + m[:, 1, 0]) / S,
                         (m[:, 0, 2] + m[:, 2, 0]) / S], -1)[c1]
    S = torch.sqrt((1.0 + m[:, 1, 1] - m[:, 0, 0] - m[:, 2, 2]).clamp_min(0)) * 2
    q[c2] = torch.stack([(m[:, 0, 2] - m[:, 2, 0]) / S, (m[:, 0, 1] + m[:, 1, 0]) / S, 0.25 * S,
                         (m[:, 1, 2] + m[:, 2, 1]) / S], -1)[c2]
    S = torch.sqrt((1.0 + m[:, 2, 2] - m[:, 0, 0] - m[:, 1, 1]).clamp_min(0)) * 2
    q[c3] = torch.stack([(m[:, 1, 0] - m[:, 0, 1]) / S, (m[:, 0, 2] + m[:, 2, 0]) / S,
                         (m[:, 1, 2] + m[:, 2, 1]) / S, 0.25 * S], -1)[c3]
    return q.float().numpy()


def init_scales(points):
    """splatfacto k_nearest_sklearn(x, 3): mean distance of the 3 nearest non-self points."""
    from sklearn.neighbors import NearestNeighbors

    nn_ = NearestNeighbors(n_neighbors=4, algorithm="auto", metric="euclidean").fit(points)
    d, _ = nn_.kneighbors(points)
    return d[:, 1:].astype(np.float32).mean(-1)


def _lightning_free_pickle():
    """The Omnidata checkpoint pickles pytorch_lightning hparams objects next to the state
    dict; map those (and only those) classes to an inert placeholder instead of installing
    pytorch_lightning."""
    import pickle
    import types

    class _Stub:
        def __init__(self, *a, **k):
            pass

        def __setstate__(self, state):
            self.state = state

    class Unpickler(pickle.Unpickler):
        def find_class(self, module, name):
            if module.split(".")[0] in ("pytorch_lightning", "lightning_fabric", "lightning"):
                return _Stub
            return super().find_class(module, name)

    mod = types.ModuleType("lightning_free_pickle")
    mod.__dict__.update({k: getattr(pickle, k) for k in dir(pickle) if not k.startswith("__")})
    mod.Unpickler = Unpickler
    return mod


def omnidata_normals(paths, device):
    import sys

    import torch
    import torchvision.transforms.functional as TF
    from PIL import Image
    from torchvision import transforms

    from omnidata_tools.torch.modules.midas import blocks

    # The official constructor asks timm for ImageNet weights that the checkpoint overwrites.
    orig = blocks._make_pretrained_vitb_rn50_384
    blocks._make_pretrained_vitb_rn50_384 = lambda pretrained, *a, **k: orig(False, *a, **k)
    from omnidata_tools.torch.modules.midas.dpt_depth import DPTDepthModel

    model = DPTDepthModel(backbone="vitb_rn50_384", num_channels=3)
    ck = torch.load(OMNI_CKPT, map_location="cpu", pickle_module=_lightning_free_pickle())
    sd = {k[6:]: v for k, v in ck["state_dict"].items()} if "state_dict" in ck else ck
    model.load_state_dict(sd)
    model.to(device).eval()
    out = []
    to_t = transforms.ToTensor()
    with torch.no_grad():
        for i, p in enumerate(paths):
            img = to_t(Image.open(p)).permute(1, 2, 0)[..., :3].to(device)
            H, W = img.shape[:2]
            x = TF.resize(img.permute(2, 0, 1), (384, 384), antialias=False)
            r = model(x[None]).clamp(min=0, max=1)[0]
            r = TF.resize(r, (H, W), antialias=False).permute(1, 2, 0)
            out.append((r.cpu().numpy() * 255).astype(np.uint8))
            if i % 50 == 0:
                print(f"omnidata {i}/{len(paths)}", flush=True, file=sys.stdout)
    return np.stack(out)


def viewmat(c2w_gl):
    """Exactly splatstudio.data.load_dataset (float32 OpenGL->OpenCV, then inverse)."""
    import torch

    c2w = torch.tensor(c2w_gl, dtype=torch.float32)
    c2w[:3, 1:3] *= -1
    return torch.linalg.inv(c2w).numpy()


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--room", default="vr_room")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--num-points", type=int, default=1_000_000)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    src = RAW / f"{a.room}_x" / "room_datasets" / a.room / "iphone"
    long_dir, short_dir = src / "long_capture", src / "short_capture"
    out = NATIVE / a.room
    side = out / "dnsplat"
    (out / "images").mkdir(parents=True, exist_ok=True)
    side.mkdir(exist_ok=True)

    lm, sm = load_meta(long_dir), load_meta(short_dir)
    test_names = [s.strip() for s in (long_dir / "test.txt").read_text().splitlines() if s.strip()]
    long_frames = sorted(lm["frames"], key=stem)
    short_frames = sorted(sm["frames"], key=stem)
    for k in ("w", "h", "fl_x", "fl_y", "cx", "cy"):
        assert lm[k] == sm[k], k
    frames, train, test, val = [], [], [], []
    for tag, fr_list, d in (("long", long_frames, long_dir), ("short", short_frames, short_dir)):
        for fr in fr_list:
            name = f"images/{tag}_{stem(fr)}.jpg"
            dst = out / name
            if not dst.exists():
                os.link(d / fr["file_path"], dst)
            frames.append({"file_path": name, "transform_matrix": fr["transform_matrix"]})
            if tag == "short":
                val.append(name)
            elif stem(fr) in test_names:
                test.append(name)
            else:
                train.append(name)
    tj = {k: lm[k] for k in ("w", "h", "fl_x", "fl_y", "cx", "cy")}
    tj.update(camera_model="PINHOLE", frames=frames, train_filenames=train,
              test_filenames=test, val_filenames=val)
    (out / "transforms.json").write_text(json.dumps(tj, indent=1))
    print(f"train {len(train)} test(within) {len(test)} val(cross/short) {len(val)}")

    # --- initial point cloud + normals + scale/quat init
    if not (out / "points.npz").exists():
        pts, cols = tsdf_pointcloud(long_dir, lm, test_names, a.num_points, a.seed)
        nrm = estimate_normals(pts)
        pts32 = pts.astype(np.float32)
        avg = init_scales(pts32)
        log_scales = np.log(np.repeat(avg[:, None], 3, 1))
        log_scales[:, 2] = np.log(avg / 10)
        quats = rotate_z_to(nrm)
        np.savez(side / "init.npz", points=pts32, log_scales=log_scales.astype(np.float32),
                 quats=quats, normals=nrm)
        np.savez(out / "points.npz", points=pts32, colors=(cols / 255.0).astype(np.float32))
        print(f"points {len(pts32)}; zero-knn {(avg == 0).sum()}")

    # --- per-training-view priors keyed by the native viewmat
    by_name = {f["file_path"]: f for f in frames}
    names = list(train)
    np.save(side / "viewmats.npy", np.stack([viewmat(by_name[n]["transform_matrix"]) for n in names]))
    (side / "names.json").write_text(json.dumps(names))
    src_of = {f"images/long_{stem(fr)}.jpg": fr for fr in long_frames}
    if not (side / "depth_mm.npy").exists():
        import cv2

        depth = np.stack([cv2.imread(str(long_dir / src_of[n]["depth_file_path"]),
                                     cv2.IMREAD_ANYDEPTH) for n in names]).astype(np.uint16)
        assert depth.shape[1:] == (lm["h"], lm["w"])
        np.save(side / "depth_mm.npy", depth)
    if not (side / "normal_omni_u8.npy").exists():
        normals = omnidata_normals([out / n for n in names], a.device)
        np.save(side / "normal_omni_u8.npy", normals)
    manifest = {"room": a.room, "seed": a.seed, "omnidata_ckpt_sha256": sha256(OMNI_CKPT),
                "files": {p.name: sha256(p) for p in sorted(side.iterdir()) if p.is_file()
                          and p.name != "manifest.json"}}
    (side / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print("done", out)


if __name__ == "__main__":
    main()
