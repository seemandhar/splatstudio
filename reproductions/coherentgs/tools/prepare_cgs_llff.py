"""Build the CoherentGS LLFF few-view protocol (official scene/dataset_readers.py readLLFFInfo) as
native scenes: <out>/<scene>/{images/, transforms.json, points.npz, coherentgs.npz, prep.json}.

Inputs: the author-processed LLFF archive (README Google-Drive link, llff_sparse.zip, sha256
84caaa59...) extracted to --raw: <scene>/poses_bounds.npy, images_8/*.png (pixel-identical to the
standard LLFF images_8), images_8/depth_rel/<stem>.pt (Depth Anything ViT-L relative disparity),
images_8/flow/<a>_<b>.npy (FlowFormer++ flow a->b, HxWx2) and <a>_<b>w.npy (forward-backward
consistency weight, 5x5-eroded).

Exactly as the official loader:
  * images_8 (504x378); llffhold 8 on the name-sorted list; train = round(linspace(0, n-1, N))
    of the non-test images; val = remaining images (unused, never for selection).
  * poses: poses_bounds -> [:, :3, :4] @ fix_rotation (OpenCV c2w), translations * 1/(0.75 min
    bound); focal = poses[0, 2, 4] / 8; principal point (W/2, H/2).
  * per training view: disp = depth_rel; disp_norm = disp/max + 0.2; 5-iteration sparse bilateral
    filtering (official config) -> last saved map F; z_mono = 1/F; depth-layer masks from
    get_depth_bins(disparity = (F - 0.2) * max, num_bins = C) with bins in disparity space,
    S_c = (disp > b_c) & (disp <= b_{c+1}).
  * per-pixel camera points via kornia depth_to_3d convention: integer pixel (u, v),
    x = (u - W/2)/f z, y = (v - H/2)/f z.
Stored for the plugin (coherentgs.npz): names, viewmats (native loader convention, lookup
keys), K, z_mono [N,H,W], dep_mask [N,C,H,W] uint8, flows [N,N,H,W,2], flow_w [N,N,H,W].
points.npz: world points of the *unaligned* z_mono (camera-major, row-major) + pixel colours.

Run (sklearn kept outside the env):
  PYTHONPATH=~/models/coherentgs/pylib tools/ssenv python reproductions/coherentgs/tools/prepare_cgs_llff.py \
      --raw ~/datasets/llff/coherentgs-official/nerf_llff_data --out ~/datasets/llff/fewview3-cgs
"""

import argparse
import hashlib
import json
import os
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from official_prep import CONFIG, get_depth_bins, sparse_bilateral_filtering  # noqa: E402

SCENES = ["fern", "flower", "fortress", "horns", "leaves", "orchids", "room", "trex"]
FIX = np.array([[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]], dtype=np.float32)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def split(n_images, n_views):
    idx = np.arange(n_images)
    test = idx[idx % 8 == 0]
    train_all = idx[idx % 8 != 0]
    pick = [round(i) for i in np.linspace(0, train_all.shape[0] - 1, n_views)]
    train = [int(train_all[i]) for i in pick]
    val = [int(i) for i in train_all if int(i) not in train]
    return train, [int(i) for i in test], val


def native_viewmat(transform_matrix):
    """Exactly splatstudio.data.load_dataset's viewmat (float32 torch)."""
    c2w = torch.tensor(transform_matrix, dtype=torch.float32)
    c2w[:3, 1:3] *= -1
    return torch.linalg.inv(c2w).numpy()


def masks_and_depth(disp, n_bins):
    """Official readLLFFInfo per-view depth filtering and depth-layer masks."""
    disparity_max = disp.max()
    disparity_norm = (disp / disparity_max) + 0.2
    dummy_image = np.zeros((*disp.shape, 3), dtype=np.float32)  # only used for visualisation
    _, vis_depths = sparse_bilateral_filtering(disparity_norm.numpy().copy(), dummy_image, CONFIG,
                                               num_iter=CONFIG["sparse_iter"], spdb=False)
    disparity = (torch.tensor(vis_depths[-1])[None, None] - 0.2) * disparity_max
    bins = get_depth_bins(disparity=disparity, num_bins=n_bins)
    bins = [1 / x for x in bins]
    bins.reverse()
    z_mono = 1 / vis_depths[-1]
    dep = disparity[0, 0].numpy()
    masks = [np.where((dep > bins[i]) & (dep <= bins[i + 1]), 1, 0) for i in range(len(bins) - 1)]
    return z_mono.astype(np.float32), np.stack(masks).astype(np.uint8)


def build(scene, raw, out, n_views, n_bins):
    raw, out = Path(raw) / scene, Path(out) / scene
    imgdir = raw / "images_8"
    files = sorted(f for f in os.listdir(imgdir) if f.endswith(("JPG", "jpg", "png")))
    poses_arr = np.load(raw / "poses_bounds.npy", allow_pickle=False)
    poses = poses_arr[:, :-2].reshape([-1, 3, 5])
    bounds = poses_arr[:, -2:]
    focal = float(poses[0, -1, -1] / 8)
    poses = poses[:, :3, :4] @ FIX
    poses[:, :3, 3] *= 1.0 / (bounds.min() * 0.75)
    assert len(files) == len(poses), (scene, len(files), len(poses))
    with Image.open(imgdir / files[0]) as im:
        width, height = im.size
    train, test, val = split(len(files), n_views)
    (out / "images").mkdir(parents=True, exist_ok=True)
    frames, transforms = [], {}
    for i, f in enumerate(files):
        target = out / "images" / f
        if not target.exists():
            os.link(imgdir / f, target)
        c2w = np.eye(4)
        c2w[:3] = poses[i]
        gl = c2w.copy()
        gl[:3, 1:3] *= -1
        rel = f"images/{f}"
        transforms[i] = gl
        frames.append({"file_path": rel, "transform_matrix": gl.tolist(), "w": width, "h": height,
                       "fl_x": focal, "fl_y": focal, "cx": width / 2.0, "cy": height / 2.0,
                       "camera_model": "PINHOLE"})
    meta = {
        "train_filenames": [f"images/{files[i]}" for i in train],
        "val_filenames": [f"images/{files[i]}" for i in val],
        "test_filenames": [f"images/{files[i]}" for i in test],
        "source": {"scene": scene, "protocol": "CoherentGS readLLFFInfo: images_8, llffhold 8, "
                   f"train = round(linspace) {n_views} views, official poses_bounds frame"},
        "frames": frames,
    }
    (out / "transforms.json").write_text(json.dumps(meta, indent=1))

    K = np.array([[focal, 0, width / 2.0], [0, focal, height / 2.0], [0, 0, 1]], dtype=np.float64)
    u, v = np.meshgrid(np.arange(width, dtype=np.float64), np.arange(height, dtype=np.float64))
    names, viewmats, zs, masks, pts, cols, hashes = [], [], [], [], [], [], {}
    stems = [Path(files[i]).stem for i in train]
    for i, stem in zip(train, stems):
        dpath = imgdir / "depth_rel" / f"{stem}.pt"
        hashes[dpath.name] = sha(dpath)
        disp = torch.load(dpath, map_location="cpu", weights_only=True).detach().float()
        assert disp.shape == (height, width), disp.shape
        z, m = masks_and_depth(disp, n_bins)
        x = (u - K[0, 2]) / K[0, 0] * z
        y = (v - K[1, 2]) / K[1, 1] * z
        cam = np.stack([x, y, z, np.ones_like(z)], -1).reshape(-1, 4)
        world = (transforms[i] * np.array([1, -1, -1, 1]))  # back to OpenCV c2w
        pts.append((cam @ world.T)[:, :3])
        with Image.open(imgdir / files[i]) as im:
            cols.append(np.asarray(im.convert("RGB"), dtype=np.float32).reshape(-1, 3) / 255.0)
        names.append(f"images/{files[i]}")
        viewmats.append(native_viewmat(frames[i]["transform_matrix"]))
        zs.append(z)
        masks.append(m)
    n = len(train)
    flows = np.zeros((n, n, height, width, 2), np.float32)
    flow_w = np.zeros((n, n, height, width), np.float32)
    for a in range(n):
        for b in range(n):
            if a == b:
                continue
            fp = imgdir / "flow" / f"{stems[a]}_{stems[b]}.npy"
            wp = imgdir / "flow" / f"{stems[a]}_{stems[b]}w.npy"
            flows[a, b] = np.load(fp, allow_pickle=False)
            flow_w[a, b] = np.load(wp, allow_pickle=False)
            hashes[fp.name], hashes[wp.name] = sha(fp), sha(wp)
    np.savez(out / "points.npz", points=np.concatenate(pts).astype(np.float32),
             colors=np.concatenate(cols).astype(np.float32))
    np.savez(out / "coherentgs.npz", names=np.array(names), viewmats=np.stack(viewmats),
             K=K.astype(np.float32), z_mono=np.stack(zs), dep_mask=np.stack(masks), flows=flows,
             flow_w=flow_w, scale=np.float64(1.0 / (bounds.min() * 0.75)))
    prep = {"scene": scene, "train": names, "test": meta["test_filenames"], "n_val": len(val),
            "image_size": [width, height], "focal": focal, "mask_channels": n_bins,
            "pixels_per_view": width * height, "points": int(n * width * height),
            "mask_coverage": [float(m.max(0).mean()) for m in masks],
            "source_sha256": hashes,
            "archive": "llff_sparse.zip sha256 84caaa59174f003edf506d86e8a34db5a08515cba40d0bdde78f721fff0f673d",
            "depth_model": "Depth Anything ViT-L (author-computed depth_rel/*.pt)",
            "flow_model": "FlowFormer++ (author-computed flow/*.npy, *w.npy)"}
    (out / "prep.json").write_text(json.dumps(prep, indent=1))
    return scene, prep["mask_coverage"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--scenes", nargs="+", default=SCENES)
    ap.add_argument("--views", type=int, default=3)
    ap.add_argument("--bins", type=int, default=5)
    ap.add_argument("--jobs", type=int, default=8)
    a = ap.parse_args()
    with Pool(min(a.jobs, len(a.scenes))) as pool:
        for scene, cov in pool.starmap(build, [(s, a.raw, a.out, a.views, a.bins) for s in a.scenes]):
            print(scene, "mask coverage", cov, flush=True)


if __name__ == "__main__":
    main()
