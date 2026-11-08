"""Stage 1 of Foroutan et al.: NerfAcc iNGP @ 5k -> point cloud + depth sidecar (training views).

Writes a hard-linked copy of a native SplatStudio scene (same transforms.json and images, hence
the same dataset_sha256 / protocol_id) with

- `points.npz`              the initial point cloud, loaded by the runtime like an SfM cloud:
    * `--init nerf` (default, paper Sec. 3.2 / 4.3): 500k rays drawn uniformly from all training
      rays, one inverse-CDF sample of the volume-rendering weights per ray (Eq. 2-4), colour =
      NeRF radiance at the sample in the ray direction;
    * `--init random-box` (paper Sec. 4.2, "50^3 BBox"): 50k centres uniform in a 50 x 50 x 50
      box centred at the COLMAP origin, colours SH2RGB(U[0,1)/255) as 3DGS's random init;
- `nerfinit/depth_z_f16.npy` D_NeRF (Eq. 6) of every training pixel, as camera z in world units;
- `nerfinit/{viewmats.npy,names.json,init.npz,ingp.pt,nerfinit.json}` (see sidecar.py).

    ss-nerfinit-prep --source ~/datasets/mipnerf360/native/garden \
                     --output ~/datasets/nerfinit-gs/mipnerf360/garden
"""

import argparse
import json
import os
import shutil
import time
from pathlib import Path

import numpy as np
import torch

from . import sidecar
from .ingp import INGP, RayCaster, expected_depth, gs2nerf_transform, surface_samples, train_ingp

FP16_MAX = 65000.0


def save_npy(path, array):
    tmp = path.with_name(path.name + ".tmp.npy")
    np.save(tmp, array)
    os.replace(tmp, path)


def save_npz(path, **arrays):
    tmp = path.with_name(path.stem + ".tmp.npz")
    np.savez(tmp, **arrays)
    os.replace(tmp, path)


def copy_scene(source, output):
    output.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source / "transforms.json", output / "transforms.json")
    meta = json.loads((source / "transforms.json").read_text())
    for frame in meta["frames"]:
        src = (source / frame["file_path"]).resolve()
        dst = output / frame["file_path"]
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists():
            continue
        try:
            os.link(src, dst)
        except OSError:
            shutil.copy2(src, dst)


def knn_log_scales(points, k=3, chunk=1024):
    """3DGS init: log sqrt(mean of the k smallest squared neighbour distances) (distCUDA2)."""
    out = torch.empty(len(points), device=points.device)
    for start in range(0, len(points), chunk):
        d2 = torch.cdist(points[start : start + chunk], points).square_()
        rows = torch.arange(d2.shape[0], device=points.device)
        d2[rows, rows + start] = float("inf")
        out[start : start + chunk] = d2.topk(k, dim=1, largest=False).values.mean(dim=1)
        del d2
    return out.clamp_min(1e-7).sqrt().log()


def camera_tensors(dataset, names, caster, device):
    K_inv, R, o = [], [], []
    for n in names:
        f = dataset["frames"][n]
        ki, r, oo = caster.camera(f["K"].to(device), f["viewmat"].to(device))
        K_inv.append(ki)
        R.append(r)
        o.append(oo)
    return torch.stack(K_inv), torch.stack(R), torch.stack(o)


@torch.no_grad()
def nerf_points(model, caster, K_inv, R, o, n_img, H, W, count, seed, batch=8192):
    """Uniform subset of all training rays, one weight-distribution sample per ray (Eq. 4)."""
    device = K_inv.device
    gen = torch.Generator(device=device).manual_seed(seed + 1)
    pts, cols, ts = [], [], []
    for start in range(0, count, batch):
        m = min(batch, count - start)
        ids = torch.randint(0, n_img, (m,), device=device, generator=gen)
        px = torch.randint(0, W, (m,), device=device, generator=gen)
        py = torch.randint(0, H, (m,), device=device, generator=gen)
        origins, dirs, _, _ = RayCaster.rays(K_inv[ids], R[ids], o[ids], px, py)
        p, c, t, _ = surface_samples(model, origins, dirs, generator=gen)
        pts.append(p)
        cols.append(c)
        ts.append(t)
    pts, cols, ts = torch.cat(pts), torch.cat(cols), torch.cat(ts)
    return caster.to_world(pts), cols.clamp(0, 1), ts


def random_box_points(count, size, seed):
    rng = np.random.default_rng(seed)
    pts = rng.uniform(-size / 2, size / 2, size=(count, 3)).astype(np.float32)
    shs = rng.random((count, 3)) / 255.0  # 3DGS random init: SH2RGB(random / 255)
    cols = (shs * 0.28209479177387814 + 0.5).astype(np.float32)
    return pts, cols


@torch.no_grad()
def depth_maps(model, caster, K_inv, R, o, H, W, scale, log=print):
    """Eq. 6 for every training pixel -> camera z in world units (float16)."""
    device = K_inv.device
    py, px = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device),
                            indexing="ij")
    px, py = px.reshape(-1), py.reshape(-1)
    out = np.empty((len(K_inv), H, W), dtype=np.float16)
    stats = []
    for i in range(len(K_inv)):
        origins, dirs, norm, _ = RayCaster.rays(K_inv[i], R[i], o[i], px, py)
        t = expected_depth(model, origins, dirs)
        z = (t / scale / norm).reshape(H, W)  # ray distance (nerf units) -> world camera z
        stats.append([float(z.min()), float(z.median()), float(z.max())])
        out[i] = z.clamp(0, FP16_MAX).cpu().numpy().astype(np.float16)
        if (i + 1) % 20 == 0 or i + 1 == len(K_inv):
            log(f"depth {i + 1}/{len(K_inv)}")
    return out, stats


def main(argv=None):
    p = argparse.ArgumentParser(prog="ss-nerfinit-prep", description=__doc__.split("\n")[0])
    p.add_argument("--source", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--init", choices=("nerf", "random-box"), default="nerf")
    p.add_argument("--iterations", type=int, default=5000)  # NerfAcc @ 5k (Sec. 4.3)
    p.add_argument("--num-points", type=int, default=None,
                   help="default 500000 (nerf, Sec. 4.3) / 50000 (random-box, Table 1 caption)")
    p.add_argument("--box-size", type=float, default=50.0)  # 50^3 BBox (Sec. 4.2)
    p.add_argument("--no-depth", action="store_true", help="skip the depth sidecar")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args(argv)
    from splatstudio.data import load_dataset

    count = args.num_points or (500_000 if args.init == "nerf" else 50_000)
    source, output = args.source.expanduser().resolve(), args.output.expanduser()
    side = output / sidecar.SUBDIR
    if (output / "points.npz").exists() and not args.overwrite:
        raise SystemExit(f"{output}/points.npz exists; pass --overwrite to rebuild")
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    t_load = time.perf_counter()
    dataset = load_dataset(source)
    train = dataset["splits"]["train"]
    # nerfacc's 360 loader normalises with every camera pose (similarity_from_cameras).
    c2w = np.stack([np.linalg.inv(f["viewmat"].numpy()) for f in dataset["frames"].values()])
    transform, scale = gs2nerf_transform(c2w)
    caster = RayCaster(transform, scale, device)
    shapes = {tuple(dataset["frames"][n]["image"].shape) for n in train}
    if len(shapes) != 1:
        raise SystemExit(f"Training images must share one resolution, got {sorted(shapes)}")
    images = torch.stack([dataset["frames"][n]["image"] for n in train]).to(device)
    n_img, H, W, _ = images.shape
    K_inv, R, o = camera_tensors(dataset, train, caster, device)
    load_seconds = time.perf_counter() - t_load

    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    t0 = time.perf_counter()
    model = INGP({"iterations": args.iterations}).to(device)
    history = train_ingp(model, images, K_inv, R, o, seed=args.seed)
    del images
    torch.cuda.empty_cache()
    torch.cuda.synchronize(device)
    ingp_seconds = time.perf_counter() - t0

    t1 = time.perf_counter()
    if args.init == "nerf":
        points, colors, t = nerf_points(model, caster, K_inv, R, o, n_img, H, W, count, args.seed)
        far_frac = float((t > 0.5 * model.cfg["far"]).float().mean())
    else:
        pts, cols = random_box_points(count, args.box_size, args.seed)
        points, colors = torch.from_numpy(pts).to(device), torch.from_numpy(cols).to(device)
        far_frac = None
    log_scales = knn_log_scales(points)
    torch.cuda.synchronize(device)
    sample_seconds = time.perf_counter() - t1

    t2 = time.perf_counter()
    depth, depth_stats = (None, None) if args.no_depth else depth_maps(
        model, caster, K_inv, R, o, H, W, scale)
    torch.cuda.synchronize(device)
    depth_seconds = time.perf_counter() - t2

    copy_scene(source, output)
    side.mkdir(parents=True, exist_ok=True)
    pts = points.cpu().numpy().astype(np.float32)
    sha = sidecar.points_sha256(pts)
    save_npz(output / "points.npz", points=pts, colors=colors.cpu().numpy().astype(np.float32))
    save_npz(side / "init.npz", points_sha256=np.array(sha),
             log_scales=log_scales.cpu().numpy().astype(np.float32))
    save_npy(side / "viewmats.npy",
             np.stack([dataset["frames"][n]["viewmat"].numpy() for n in train]).astype(np.float32))
    (side / "names.json").write_text(json.dumps(train) + "\n")
    if depth is not None:
        save_npy(side / "depth_z_f16.npy", depth)
    torch.save({"points_sha256": sha, "transform": torch.as_tensor(transform), "scale": scale,
                "cfg": dict(model.cfg), "state_dict": model.state_dict()}, side / "ingp.pt")
    check = load_dataset(output)
    if check["dataset_id"] != dataset["dataset_id"]:
        raise SystemExit("Copied dataset hash differs from the source")
    centre_dist = torch.linalg.vector_norm(points - points.mean(0), dim=1)
    info = {
        "source": str(source), "output": str(output.resolve()), "init": args.init,
        "points_sha256": sha, "dataset_sha256": check["dataset_id"],
        "protocol_id": check["protocol_id"], "train_views": len(train), "image_hw": [H, W],
        "num_points": count, "iterations": args.iterations, "seed": args.seed,
        "ingp_cfg": {k: list(v) if isinstance(v, tuple) else v for k, v in model.cfg.items()},
        "normalisation_scale": scale,
        "seconds": {"load": load_seconds, "ingp_train": ingp_seconds,
                    "points_and_knn": sample_seconds, "depth_render": depth_seconds,
                    "method_total": ingp_seconds + sample_seconds + depth_seconds},
        "ingp_history": history,
        "points": {"far_fraction_t_gt_half_far": far_frac,
                   "dist_from_mean_quantiles_05_50_95": [
                       float(x) for x in centre_dist.quantile(
                           torch.tensor([0.05, 0.5, 0.95], device=device))]},
        "log_scale_stats": {"median": float(log_scales.median()),
                            "p01": float(log_scales.quantile(0.01)),
                            "p99": float(log_scales.quantile(0.99))},
        "depth_z_stats_min_median_max_first5": depth_stats[:5] if depth_stats else None,
        "peak_gpu_bytes": torch.cuda.max_memory_allocated(device),
        "gpu": torch.cuda.get_device_name(device),
    }
    (side / "nerfinit.json").write_text(json.dumps(info, indent=1) + "\n")
    print(json.dumps({k: v for k, v in info.items() if k != "ingp_history"}, indent=1))


if __name__ == "__main__":
    main()
