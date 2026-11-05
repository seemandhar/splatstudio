"""Build the DNGaussian LLFF 3-view protocol (official scripts/run_llff.sh) as native scenes.

Reuses llff-fewview3-r8 for everything that is identical to the official loader and adds
what DNGaussian does differently. Output per scene: <out>/<scene>/{images/ (hard links),
transforms.json (byte copy), points.npz, dng_depth.npz, prep.json}.

Identical to llff-fewview3-r8 (checked against source/code/scene/dataset_readers.py and
utils/camera_utils.py): test = every 8th name-sorted image (eval_cam_infos, the set
metrics.py scores), train = round(np.linspace(0, n-1, 3)) of the rest, full-res images PIL
bicubic-resized to round(W/8) x round(H/8) = 504x378, PINHOLE with centred principal point,
COLMAP world frame unchanged.

Different (hence a new dataset):
  * points.npz: official --rand_pcd init (dataset_readers.py readColmapSceneInfo, LLFF branch):
    np.random.seed(0) (utils/general_utils.py safe_state, the first numpy draw of a run);
    xyz = sparse/0/points3D.bin of the *full* LLFF reconstruction, used only for its bounding
    box: shape = max - min (top-1 per axis), num_pts = int(shape.max() * 50),
    xyz = U[0,1)^3 * shape * 1.3 + (20th smallest coordinate per axis);
    colours SH2RGB(U[0,1)/255) (unused: DNGaussian's colour comes from the neural renderer).
  * dng_depth.npz: DPT-Hybrid (dpt_hybrid_384, MiDaS v3) monocular depth of the 3 TRAINING
    views only (official dpt/get_depth_map_for_llff_dtu.py: full-res RGB -> MiDaS
    dpt_transform -> prediction bicubic-resized to (H//8, W//8) -> utils_io.write_depth(bits=2):
    uint16(65535 * (d - min) / (max - min)) PNG; the loader reads it with PIL, /255, and the
    trainer uses depth_mono = 255 - png/255). Stored: viewmats (native-loader convention, used
    as lookup keys), depth_u16, depth_mono (float32), the raw float prediction, and the
    official 180-pose spiral camera centres (scene/dataset_readers.py CreateLLFFSpiral +
    utils/pose_utils.py, from the raw poses_bounds.npy) used by the near-camera pruning.

Run (DPT needs timm 0.6.12 + the MiDaS repo, kept outside the splatstudio env):
    PYTHONPATH=~/models/dpt/pylib tools/ssenv python reproductions/dngaussian/tools/prepare_dng_llff.py \
        --scenes fern flower ... --src ~/datasets/llff/fewview3-r8 \
        --raw ~/datasets/llff/nerf_llff_data --out ~/datasets/llff/fewview3-r8-dng
"""

import argparse
import hashlib
import json
import os
import shutil
import struct
import sys
from pathlib import Path

import numpy as np
import torch

MIDAS_REPO = Path.home() / "models/dpt/midas_repo"
DPT_WEIGHTS = Path.home() / "models/dpt/dpt_hybrid_384.pt"


# ---------------------------------------------------------------- random init (official)
def read_points3d_xyz(path):
    out = []
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        for _ in range(n):
            d = struct.unpack("<QdddBBBd", f.read(43))
            out.append(d[1:4])
            (tl,) = struct.unpack("<Q", f.read(8))
            f.seek(8 * tl, os.SEEK_CUR)
    return np.asarray(out, dtype=np.float64)


def topk_axis0_last(matrix, k):
    """Official topk_(matrix, K, axis=0)[-1]: the K-th largest value per column."""
    return -np.sort(-matrix, axis=0)[k - 1]


def official_random_init(xyz_sparse, seed=0):
    rng_state = np.random.get_state()
    np.random.seed(seed)
    pcd_shape = topk_axis0_last(xyz_sparse, 1) + topk_axis0_last(-xyz_sparse, 1)
    num_pts = int(pcd_shape.max() * 50)
    xyz = np.random.random((num_pts, 3)) * pcd_shape * 1.3 - topk_axis0_last(-xyz_sparse, 20)
    shs = np.random.random((num_pts, 3)) / 255.0
    np.random.set_state(rng_state)
    colors = shs * 0.28209479177387814 + 0.5  # SH2RGB
    return xyz, colors, num_pts


# ---------------------------------------------------------------- spiral (official)
def _normalize(x):
    return x / np.linalg.norm(x)


def _viewmatrix(lookdir, up, position):
    vec2 = _normalize(lookdir)
    vec0 = _normalize(np.cross(up, vec2))
    vec1 = _normalize(np.cross(vec2, vec0))
    return np.stack([vec0, vec1, vec2, position], axis=1)


def _poses_avg(poses):
    return _viewmatrix(poses[:, :3, 2].mean(0), poses[:, :3, 1].mean(0), poses[:, :3, 3].mean(0))


def _pad(p):
    bottom = np.broadcast_to([0, 0, 0, 1.0], p[..., :1, :4].shape)
    return np.concatenate([p[..., :3, :4], bottom], axis=-2)


def official_spiral_centers(poses_bounds, n_frames=180, n_rots=2, zrate=0.5):
    """CreateLLFFSpiral: camera centres of the 180 render poses (COLMAP world frame)."""
    poses_o = poses_bounds[:, :-2].reshape([-1, 3, 5])
    bounds = poses_bounds[:, -2:]
    fix = np.array([[0, -1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float32)
    poses = poses_o[:, :3, :4] @ fix
    c2w_avg = _poses_avg(poses)
    rec = (np.linalg.inv(_pad(c2w_avg)) @ _pad(poses))[..., :3, :4]
    close_depth, inf_depth = bounds.min() * 0.9, bounds.max() * 5.0
    dt = 0.75
    focal = 1 / (((1 - dt) / close_depth + dt / inf_depth))
    radii = np.concatenate([np.percentile(np.abs(rec[:, :3, 3]), 90, 0), [1.0]])
    cam2world = _poses_avg(rec)
    up = rec[:, :3, 1].mean(0)
    render = []
    for theta in np.linspace(0.0, 2.0 * np.pi * n_rots, n_frames, endpoint=False):
        t = radii * [np.cos(theta), -np.sin(theta), -np.sin(theta * zrate), 1.0]
        position = cam2world @ t
        lookat = cam2world @ [0, 0, -focal, 1.0]
        render.append(_viewmatrix(position - lookat, up, position))
    render = np.stack(render, 0)
    render = (_pad(c2w_avg) @ _pad(render))[..., :3, :4]  # backcenter_poses
    render = render @ np.linalg.inv(fix)
    return render[:, :3, 3].astype(np.float64)


# ---------------------------------------------------------------- DPT (official)
def load_dpt(device):
    sys.path.insert(0, str(MIDAS_REPO))
    import cv2  # noqa: F401
    from midas.dpt_depth import DPTDepthModel
    from midas.transforms import NormalizeImage, PrepareForNet, Resize
    from torchvision.transforms import Compose

    import timm

    if timm.__version__ != "0.6.12":
        raise RuntimeError(f"MiDaS DPT needs timm 0.6.12 (PYTHONPATH=~/models/dpt/pylib), got {timm.__version__}")
    model = DPTDepthModel(path=None, backbone="vitb_rn50_384", non_negative=True)
    sha = hashlib.sha256(DPT_WEIGHTS.read_bytes()).hexdigest()
    if not sha.startswith("501f0c75"):
        raise RuntimeError(f"unexpected DPT-Hybrid weights hash {sha}")
    model.load_state_dict(torch.load(DPT_WEIGHTS, map_location="cpu"))
    model.to(device).eval()
    import cv2 as _cv2

    transform = Compose([  # hubconf.py transforms().dpt_transform
        lambda img: {"image": img / 255.0},
        Resize(384, 384, resize_target=None, keep_aspect_ratio=True, ensure_multiple_of=32,
               resize_method="minimal", image_interpolation_method=_cv2.INTER_CUBIC),
        NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        PrepareForNet(),
        lambda sample: torch.from_numpy(sample["image"]).unsqueeze(0),
    ])
    return model, transform, sha


@torch.no_grad()
def dpt_depth(model, transform, path, device, downsampling=8):
    import cv2

    img = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]
    pred = model(transform(img).to(device))
    pred = torch.nn.functional.interpolate(pred.unsqueeze(1), size=(h // downsampling, w // downsampling),
                                           mode="bicubic", align_corners=False).squeeze()
    d = pred.cpu().numpy().astype(np.float32)
    dmin, dmax = d.min(), d.max()
    u16 = (65535 * (d - dmin) / (dmax - dmin)).astype("uint16")  # write_depth(bits=2)
    return d, u16


def native_viewmat(transform_matrix):
    """Exactly splatstudio.data.load_dataset's viewmat (float32 torch)."""
    c2w = torch.tensor(transform_matrix, dtype=torch.float32)
    c2w[:3, 1:3] *= -1
    return torch.linalg.inv(c2w).numpy()


def build(scene, src, raw, out, model, transform, sha, device):
    src, raw, out = Path(src) / scene, Path(raw) / scene, Path(out) / scene
    (out / "images").mkdir(parents=True, exist_ok=True)
    for f in sorted((src / "images").iterdir()):
        t = out / "images" / f.name
        if not t.exists():
            os.link(f, t)
    shutil.copyfile(src / "transforms.json", out / "transforms.json")
    meta = json.loads((src / "transforms.json").read_text())
    xyz_sparse = read_points3d_xyz(raw / "sparse/0/points3D.bin")
    xyz, colors, num_pts = official_random_init(xyz_sparse)
    np.savez(out / "points.npz", points=xyz.astype(np.float32), colors=colors.astype(np.float32))
    spiral = official_spiral_centers(np.load(raw / "poses_bounds.npy"))
    frames = {f["file_path"]: f for f in meta["frames"]}
    names, viewmats, u16s, monos, preds = [], [], [], [], []
    for rel in meta["train_filenames"]:  # training views only
        stem = Path(rel).stem
        jpgs = [p for p in (raw / "images").iterdir() if p.stem == stem]
        assert len(jpgs) == 1, jpgs
        d, u16 = dpt_depth(model, transform, jpgs[0], device)
        fr = frames[rel]
        assert u16.shape == (fr["h"], fr["w"]), (u16.shape, fr["w"], fr["h"])
        names.append(rel)
        viewmats.append(native_viewmat(fr["transform_matrix"]))
        u16s.append(u16)
        monos.append((255.0 - u16.astype(np.float64) / 255.0).astype(np.float32))
        preds.append(d)
    np.savez(out / "dng_depth.npz", names=np.array(names), viewmats=np.stack(viewmats),
             depth_u16=np.stack(u16s), depth_mono=np.stack(monos), dpt_prediction=np.stack(preds),
             spiral_centers=spiral)
    cams = np.stack([np.linalg.inv(native_viewmat(f["transform_matrix"]))[:3, 3] for f in meta["frames"]])
    info = {"scene": scene, "train": meta["train_filenames"], "test": meta["test_filenames"],
            "init": "official --rand_pcd (seed 0)", "init_points": int(num_pts),
            "init_bbox_min": xyz.min(0).tolist(), "init_bbox_max": xyz.max(0).tolist(),
            "depth_model": "MiDaS v3 DPT_Hybrid dpt_hybrid_384.pt sha256 " + sha,
            "depth_views": names, "spiral_frames": int(len(spiral)),
            "spiral_to_nearest_camera_max": float(np.min(np.linalg.norm(
                spiral[:, None] - cams[None], axis=-1), axis=1).max()),
            "source_dataset": str(src)}
    (out / "prep.json").write_text(json.dumps(info, indent=2) + "\n")
    return info


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scenes", nargs="+", required=True)
    p.add_argument("--src", default=str(Path.home() / "datasets/llff/fewview3-r8"))
    p.add_argument("--raw", default=str(Path.home() / "datasets/llff/nerf_llff_data"))
    p.add_argument("--out", default=str(Path.home() / "datasets/llff/fewview3-r8-dng"))
    p.add_argument("--device", default="cuda")
    a = p.parse_args()
    model, transform, sha = load_dpt(a.device)
    for s in a.scenes:
        print(json.dumps(build(s, a.src, a.raw, a.out, model, transform, sha, a.device)), flush=True)


if __name__ == "__main__":
    main()
