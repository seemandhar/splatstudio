"""Build the SplatFields few-view Blender protocol as native SplatStudio scenes.

Mirrors the official SplatFields code (github.com/markomih/SplatFields, MIT) where possible:
  * cameras (scene/dataset_readers.py:readCamerasFromTransformsCV): Blender c2w, world scaled by
    2 / model_scale_dict[scene]; written here as an orthonormal OpenGL c2w with the translation
    scaled (a similarity of the official camera, identical projection);
  * images: RGBA composited on white, `norm*a + (1-a)`, truncated to 8 bit like the official
    `np.array(arr*255.0, dtype=np.byte)`; saved as RGB PNG (alpha = 1 for the native loader);
  * train views (readNerfSyntheticCVInfo): `KMeans(n_views, random_state=0)` on the scaled
    train camera centres, nearest camera to each centre, sorted -- computed by
    select_views.py under scikit-learn 1.0.2 and passed in with --views (views.json);
  * test: all transforms_test.json views; validation: first 8 transforms_val.json views
    (never used for selection; the runtime needs a non-empty validation split);
  * init (pts_samples='hull'): 256^3 grid in [-1,1]^3, kept where the official projection
    (getWorld2View2 / getProjectionMatrix, clip-z divide, ndc2Pix, rounding) lands on
    mask > 0 in every train view; random 100k (num_pts default), colours U[0,1)/255;
  * masks.npz: train-view alpha (uint8) + the viewmat the native loader computes, for the
    SplatFields mask loss (supp. Eq. A.2, lambda_2 = 0.1).

Run in the splatstudio env (numpy, Pillow, torch):
    tools/ssenv python -I reproductions/splatfields/tools/prepare_blender_fewview.py \
        --raw ~/datasets/nerf_synthetic/raw --out ~/datasets/nerf_synthetic/fewview6 \
        --views reproductions/splatfields/tools/views.json --n-views 6 --scenes chair hotdog ficus
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from PIL import Image

MODEL_SCALE = dict(chair=2.1, drums=2.3, ficus=2.3, hotdog=3.0, lego=2.4, materials=2.4,
                   mic=2.5, ship=2.75)
B2O = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]], dtype=np.float32)


def focal2fov(focal, pixels):
    return 2 * math.atan(pixels / (2 * focal))


def get_world2view2(R, t):
    """Official utils/graphics_utils.py:getWorld2View2 (translate 0, scale 1)."""
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0
    C2W = np.linalg.inv(Rt)
    Rt = np.linalg.inv(C2W)
    return np.float32(Rt)


def get_projection_matrix(znear, zfar, fovX, fovY):
    """Official getProjectionMatrix."""
    tanHalfFovY = math.tan(fovY / 2)
    tanHalfFovX = math.tan(fovX / 2)
    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right
    P = torch.zeros(4, 4)
    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = 1.0
    P[2, 2] = zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P


def ndc2pix(v, S):
    return ((v + 1.0) * S - 1.0) * 0.5


def composite_8bit(path):
    """Official readCamerasFromTransformsCV image + mask (white background)."""
    im = np.array(Image.open(path).convert("RGBA"))
    norm = im / 255.0
    arr = norm[:, :, :3] * norm[:, :, 3:4] + np.array([1, 1, 1]) * (1 - norm[:, :, 3:4])
    rgb = np.trunc(arr * 255.0).astype(np.uint8)  # == np.byte cast for values in [0, 255]
    return rgb, im[:, :, 3], norm[:, :, 3:4]


def read_split(scene_dir, split, ws):
    meta = json.loads((scene_dir / f"transforms_{split}.json").read_text())
    angle = meta["camera_angle_x"]
    cams = []
    for frame in meta["frames"]:
        tm_gl = np.array(frame["transform_matrix"], dtype=np.float64)
        tm = tm_gl @ B2O
        tm[:3, :4] *= ws  # official: rotation part scaled too (a similarity)
        w2c = np.linalg.inv(tm)
        R, T = np.transpose(w2c[:3, :3]), w2c[:3, 3]
        cams.append({"file": frame["file_path"], "tm_gl": tm_gl, "R": R, "T": T,
                     "center": tm[:3, 3], "angle_x": angle})
    return cams


def official_mask_test(points, cams, masks, H, W):
    """Visual-hull test of the official 'hull'/'load' branches on [N,3] float32 points."""
    pts = torch.from_numpy(np.concatenate([points, np.ones((len(points), 1))], axis=1)).float()
    keep = np.ones(len(points), dtype=bool)
    for cam, mask in zip(cams, masks):
        focal = (H / 2) / np.tan(cam["angle_x"] / 2)
        fovy, fovx = focal2fov(focal, H), focal2fov(focal, W)
        wv = torch.tensor(get_world2view2(cam["R"], cam["T"])).transpose(0, 1)
        proj = get_projection_matrix(0.01, 100.0, fovx, fovy).transpose(0, 1)
        full = wv.unsqueeze(0).bmm(proj.unsqueeze(0)).squeeze(0)
        cam_xyz = pts @ full
        uv = cam_xyz[:, :2] / cam_xyz[:, 2:3]  # official divides by clip z, not w
        uv = ndc2pix(uv, torch.tensor(np.array([H, W])))
        uv = np.round(uv.numpy()).astype(int)
        pix = (uv[:, 0] >= 0) & (uv[:, 0] < H) & (uv[:, 1] >= 0) & (uv[:, 1] < W)
        uvv = uv[pix]
        pix[pix] = mask[uvv[:, 1], uvv[:, 0]].reshape(-1) > 0
        keep &= pix
    return keep


def loader_viewmat(tm_gl_scaled):
    c2w = torch.tensor(tm_gl_scaled, dtype=torch.float32)
    c2w[:3, 1:3] *= -1
    return torch.linalg.inv(c2w).numpy()


def build(scene, raw, out, n_views, num_pts, n_val, seed, views):
    scene_dir = raw / scene
    ws = 2 / MODEL_SCALE[scene]
    train = read_split(scene_dir, "train", ws)
    indices = views["views"][scene][str(n_views)]
    chosen = [train[i] for i in indices]
    test = read_split(scene_dir, "test", ws)
    val = read_split(scene_dir, "val", ws)[:n_val]
    target = out / scene
    (target / "images").mkdir(parents=True, exist_ok=True)
    frames, splits, masks01, alphas, viewmats = [], {"train": [], "val": [], "test": []}, [], [], []
    H = W = None
    for split, cams in (("train", chosen), ("val", val), ("test", test)):
        for cam in cams:
            rgb, alpha, a01 = composite_8bit(scene_dir / (cam["file"] + ".png"))
            H, W = rgb.shape[:2]
            name = f"images/{split}_{Path(cam['file']).name}.png"
            Image.fromarray(rgb, "RGB").save(target / name)
            tm = cam["tm_gl"].copy()
            tm[:3, 3] *= ws  # orthonormal OpenGL c2w, scaled world
            focal = (H / 2) / np.tan(cam["angle_x"] / 2)
            frames.append({"file_path": name, "transform_matrix": tm.tolist(), "w": W, "h": H,
                           "fl_x": float(focal), "fl_y": float(focal), "cx": W / 2, "cy": H / 2,
                           "camera_model": "PINHOLE"})
            splits[split].append(name)
            if split == "train":
                masks01.append(a01)
                alphas.append(alpha)
                viewmats.append(loader_viewmat(tm))
    meta = {"frames": frames, "train_filenames": splits["train"],
            "val_filenames": splits["val"], "test_filenames": splits["test"]}
    (target / "transforms.json").write_text(json.dumps(meta, indent=1))
    np.savez_compressed(target / "masks.npz", viewmats=np.stack(viewmats).astype(np.float32),
                        masks=np.stack(alphas).astype(np.uint8),
                        names=np.array(splits["train"]))
    # visual-hull initialisation (official pts_samples='hull')
    g = np.linspace(-1.0, 1.0, 256)
    grid = np.stack(np.meshgrid(g, g, g), axis=-1).reshape(-1, 3)
    keep = official_mask_test(grid, chosen, masks01, H, W)
    hull = torch.from_numpy(grid).float().numpy()[keep]
    rng = np.random.RandomState(seed)
    n_hull = len(hull)
    if n_hull > num_pts:
        hull = hull[rng.choice(n_hull, num_pts, replace=False)]
    colors = rng.random_sample((len(hull), 3)) / 255.0
    np.savez(target / "points.npz", points=hull.astype(np.float32),
             colors=colors.astype(np.float32))
    prep = {"scene": scene, "n_views": n_views, "world_scale": ws,
            "train_indices": indices, "selection_sklearn": views["sklearn"],
            "train": splits["train"], "n_val": len(splits["val"]), "n_test": len(splits["test"]),
            "hull_voxels": int(n_hull), "init_points": int(len(hull)), "seed": seed,
            "image_size": [W, H]}
    (target / "prep.json").write_text(json.dumps(prep, indent=1))
    return prep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n-views", type=int, required=True)
    ap.add_argument("--views", type=Path, required=True, help="select_views.py output")
    ap.add_argument("--scenes", nargs="+", required=True)
    ap.add_argument("--num-pts", type=int, default=100_000)
    ap.add_argument("--n-val", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    views = json.loads(a.views.read_text())
    for scene in a.scenes:
        print(json.dumps(build(scene, a.raw.expanduser(), a.out.expanduser(), a.n_views,
                               a.num_pts, a.n_val, a.seed, views)), flush=True)


if __name__ == "__main__":
    main()
