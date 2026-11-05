"""Convert the Gaussian Surfels authors' DTU release (HuggingFace turandai/gaussian-surfels-dtu,
IDR layout: image/, mask/, normal/ (Omnidata, precomputed by the authors), cameras.npz) into
SplatStudio native scenes plus the sidecar the plugin reads.

Mirrors the official loader (scene/dataset_readers.py readIDRCameras/readIDRSceneInfo,
utils/camera_utils.loadCam, scene/gaussian_model.create_from_pcd + mask_prune):
  * cameras: P = world_mat @ scale_mat, cv2.decomposeProjectionMatrix -> K, w2c (IDR-normalised)
  * split (--eval, "following NeuS2"): sorted indices {8,13,16,21,26,31,34} (+56 if > 56 images)
    are test; the rest train. validation = byte copies of the test images (val_*.png) so the
    runtime's disjoint-split contract holds (validation aliases test, never used for selection).
  * images/*.png: RGB * mask (official get_gtImage with black background = the PSNR protocol)
  * gsurfels/ sidecar (train views only, in train order):
      viewmats.npy [N,4,4] OpenCV w2c; names.json
      image_s{2,4}.npy  uint8 [N,h,w,3]  PIL bicubic resize of the *unmasked* image (PILtoTorch)
      mask_s{1,2,4}.npy fp16 [N,h,w]      PIL 'F' luminance / 255, bilinear F.interpolate
      normal_s{1,2,4}.npy fp16 [N,3,h,w]  Omnidata prior, bilinear F.interpolate
      init.npz: log_scales [M,3], quats [M,4] aligned with points.npz
  * points.npz: 1e6 uniform points in [-1.2,1.2]^3 (seeded), visual-hull pruned like mask_prune
    (9x9 max-pooled masks, all training views, official world2scrn pixel mapping incl. its
    principal-point term), colours SH2RGB(rand/255)
  * cameras.npz copied (scale_mat for the DTU evaluator), meta["dtu_scan"].

usage: python prepare_dtu.py --raw ~/datasets/gsurfels_dtu/raw --out ~/datasets/gsurfels_dtu/native
       [--scans 24 37 ...]
"""

import argparse
import json
import shutil
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.spatial import cKDTree

SCANS = [24, 37, 40, 55, 63, 65, 69, 83, 97, 105, 106, 110, 114, 118, 122]
C0 = 0.28209479177387814
ZNEAR, ZFAR = 0.01, 100.0


def test_ids(n):
    ids = [8, 13, 16, 21, 26, 31, 34]
    if n > 56:
        ids.append(56)
    return ids


def load_mask(path):
    """official load_mask: imageio.imread(pilmode='F') / 255 (PIL 'F' = ITU-R 601 luminance)."""
    return np.asarray(Image.open(path).convert("F"), dtype=np.float32) / 255.0


def resize_t(arr, hw):
    """official utils/image_utils.resize_image(img, [h, w]) bilinear (no antialias)."""
    t = torch.from_numpy(np.ascontiguousarray(arr)).float()
    if tuple(t.shape[-2:]) == tuple(hw):
        return t.numpy()
    return F.interpolate(t[None], size=hw, mode="bilinear")[0].numpy()


def rotmat2quaternion(R):
    """official general_utils.rotmat2quaternion(normalize=False)."""
    tr = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2] + 1e-6
    r = torch.sqrt(1 + tr) / 2
    return torch.stack([r, (R[:, 2, 1] - R[:, 1, 2]) / (4 * r), (R[:, 0, 2] - R[:, 2, 0]) / (4 * r),
                        (R[:, 1, 0] - R[:, 0, 1]) / (4 * r)], -1)


def normal2rotation(n):
    """official general_utils.normal2rotation."""
    n = F.normalize(n, dim=-1)
    w0 = torch.tensor([[1.0, 0, 0]]).expand(n.shape)
    R0 = w0 - torch.sum(w0 * n, -1, True) * n
    R0 = R0 * torch.sign(R0[:, :1])
    R0 = F.normalize(R0, dim=-1)
    R1 = torch.cross(n, R0, dim=-1)
    R1 = R1 * torch.sign(R1[:, 1:2]) * torch.sign(n[:, 2:])
    return rotmat2quaternion(torch.stack([R0, R1, n], -1))


def hull_prune(xyz, cams, pad=4, batch=32):
    """official GaussianModel.mask_prune + utils/image_utils.world2scrn (per 32-view batch)."""
    keep = torch.ones(len(xyz), dtype=torch.bool)
    pts = xyz.clone()
    idx_all = torch.arange(len(xyz))
    pool = torch.nn.MaxPool2d(9, stride=1, padding=pad)
    for b in range(0, len(cams), batch):
        cb = cams[b:b + batch]
        masks = pool(torch.stack([torch.from_numpy(c["mask"]) for c in cb])[:, None])[:, 0]
        hom = torch.cat([pts, torch.ones_like(pts[:, :1])], 1)
        inmask = []
        outview = []
        for c, m in zip(cb, masks):
            full = torch.from_numpy((c["proj"] @ c["w2c"]).astype(np.float32))  # column-vector form
            p = hom @ full.T
            ndc = p[:, :3] / (p[:, 3:] + 1e-7)
            out = (ndc[:, 0] <= -1) | (ndc[:, 0] > 1) | (ndc[:, 1] <= -1) | (ndc[:, 1] > 1)
            W, H = c["width"], c["height"]
            reso = torch.tensor([W, H], dtype=torch.float32)
            prcp = torch.tensor([c["K"][0, 2] / W, c["K"][1, 2] / H], dtype=torch.float32)
            scrn = ((ndc[:, :2] + 1) * reso - 1) * 0.5 + reso * (prcp - 0.5)
            scrn = torch.minimum(torch.clamp(scrn, min=0), reso - 1).long()
            inmask.append(m[scrn[:, 1], scrn[:, 0]] > 0)
            outview.append(out)
        inmask = torch.stack(inmask)
        outview = torch.stack(outview)
        visible = inmask.all(0) & ~outview.all(0)
        pts = pts[visible]
        idx_all = idx_all[visible]
    keep[:] = False
    keep[idx_all] = True
    return keep


def proj_matrix(K, W, H):
    """official graphics_utils.getProjectionMatrix(znear, zfar, fov, W, H, prcp) (row form)."""
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    top, bottom = ZNEAR * cy / fy, -ZNEAR * (H - cy) / fy
    right, left = ZNEAR * (W - cx) / fx, -ZNEAR * cx / fx
    P = np.zeros((4, 4))
    P[0, 0] = 2 * ZNEAR / (right - left)
    P[1, 1] = 2 * ZNEAR / (top - bottom)
    P[0, 2] = -(right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = 1.0
    P[2, 2] = ZFAR / (ZFAR - ZNEAR)
    P[2, 3] = -(ZFAR * ZNEAR) / (ZFAR - ZNEAR)
    return P


def convert(src, dst, seed=0, num_pts=1_000_000):
    names = sorted(p.stem for p in (src / "image").glob("*.png"))
    masks = sorted((src / "mask").glob("*.png"))
    assert len(masks) == len(names), (len(masks), len(names))
    cams_npz = np.load(src / "cameras.npz")
    n = len(names)
    tids = set(test_ids(n))
    if dst.exists():
        shutil.rmtree(dst)
    (dst / "images").mkdir(parents=True)
    side = dst / "gsurfels"
    side.mkdir()
    frames, train, test, val, cams = [], [], [], [], []
    for i, name in enumerate(names):
        P = (cams_npz[f"world_mat_{i}"].astype(np.float32) @ cams_npz[f"scale_mat_{i}"].astype(
            np.float32))[:3, :4]
        K, R, t = cv2.decomposeProjectionMatrix(P)[:3]
        K = K / K[2, 2]
        c = (t[:3] / t[3])[:, 0]
        w2c = np.eye(4)
        w2c[:3, :3] = R
        w2c[:3, 3] = -R @ c
        img = np.asarray(Image.open(src / "image" / f"{name}.png").convert("RGB"))
        H, W = img.shape[:2]
        mask = load_mask(masks[i])
        masked = np.clip(np.round(img.astype(np.float32) * mask[..., None]), 0, 255).astype(np.uint8)
        fname = f"images/{name}.png"
        Image.fromarray(masked).save(dst / fname)
        c2w = np.linalg.inv(w2c)
        c2w[:3, 1:3] *= -1  # OpenCV -> OpenGL (the loader flips back)
        fr = {"file_path": fname, "transform_matrix": c2w.tolist(), "fl_x": float(K[0, 0]),
              "fl_y": float(K[1, 1]), "cx": float(K[0, 2]), "cy": float(K[1, 2]), "w": W, "h": H}
        frames.append(fr)
        if i in tids:
            test.append(fname)
            vname = f"images/val_{name}.png"
            shutil.copyfile(dst / fname, dst / vname)
            frames.append({**fr, "file_path": vname})
            val.append(vname)
        else:
            train.append(fname)
            cams.append({"name": name, "K": K, "w2c": w2c, "width": W, "height": H, "mask": mask,
                         "proj": proj_matrix(K, W, H), "path": src / "image" / f"{name}.png",
                         "normal": src / "normal" / f"{name}_normal.npy"})
    meta = {"camera_model": "PINHOLE", "frames": frames, "train_filenames": train,
            "val_filenames": val, "test_filenames": test,
            "dtu_scan": int(src.name.replace("scan", "")),
            "_doc": "Gaussian Surfels DTU (IDR-normalised); images masked on black; test = NeuS2 "
                    "split; validation = copies of test"}
    (dst / "transforms.json").write_text(json.dumps(meta, indent=1))
    shutil.copyfile(src / "cameras.npz", dst / "cameras.npz")

    # ---- sidecar
    np.save(side / "viewmats.npy", np.stack([c["w2c"] for c in cams]).astype(np.float32))
    (side / "names.json").write_text(json.dumps([c["name"] for c in cams]))
    H, W = cams[0]["height"], cams[0]["width"]
    for s in (1, 2, 4):
        h, w = (H, W) if s == 1 else (round(H / s), round(W / s))
        ms = np.stack([resize_t(c["mask"][None], (h, w))[0] for c in cams]).astype(np.float16)
        np.save(side / f"mask_s{s}.npy", ms)
        nrm = []
        for c in cams:
            m = np.load(c["normal"])
            m = m[0] if m.ndim == 4 else m
            nrm.append(resize_t(m[:3], (h, w)).astype(np.float16))
        np.save(side / f"normal_s{s}.npy", np.stack(nrm))
        if s > 1:
            ims = np.stack([np.asarray(Image.open(c["path"]).convert("RGB").resize(
                (w, h), Image.Resampling.BICUBIC)) for c in cams])
            np.save(side / f"image_s{s}.npy", ims)
        del nrm

    # ---- init (readIDRSceneInfo + create_from_pcd + mask_prune)
    rng = np.random.default_rng(seed)
    normal = rng.random((num_pts, 3)) - 0.5
    normal /= np.linalg.norm(normal, 2, 1, True)
    xyz = rng.random((num_pts, 3)) * 2.4 - 1.2
    shs = rng.random((num_pts, 3)) / 255.0
    d, _ = cKDTree(xyz).query(xyz, k=4)
    dist2 = np.clip((d[:, 1:] ** 2).mean(1), 1e-7, None)  # simple-knn distCUDA2
    log_s = np.log(np.sqrt(dist2 / 4))[:, None].repeat(3, 1)
    log_s[:, 2] -= 1e10
    quats = normal2rotation(torch.from_numpy(normal).float()).numpy()
    keep = hull_prune(torch.from_numpy(xyz).float(), cams).numpy()
    finite = np.isfinite(quats).all(1)
    keep &= finite
    colors = np.clip(shs * C0 + 0.5, 0, 1)
    np.savez(dst / "points.npz", points=xyz[keep].astype(np.float32),
             colors=colors[keep].astype(np.float32))
    np.savez(side / "init.npz", log_scales=log_s[keep].astype(np.float32),
             quats=quats[keep].astype(np.float32), points=xyz[keep].astype(np.float32))
    return {"scan": src.name, "images": n, "train": len(train), "test": len(test),
            "init_points": int(keep.sum()), "nonfinite_quats": int((~finite).sum())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--scans", type=int, nargs="*", default=SCANS)
    a = ap.parse_args()
    for s in a.scans:
        print(json.dumps(convert(a.raw / f"scan{s}", a.out / f"scan{s}")), flush=True)


if __name__ == "__main__":
    main()
