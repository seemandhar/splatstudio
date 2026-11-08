"""ConeGS stage 1: per-scene iNGP + iNGP initialization (paper Sec. 4.1), training views only.

Writes a hard-linked copy of a native SplatStudio scene (same transforms.json and images, so the
same dataset_sha256/protocol_id) with

- `points.npz`       P_init = 1M points at the iNGP median depth of uniformly sampled training
                     pixels, colour = iNGP colour with a zeroed view direction (clamped [0,1]);
- `conegs_init.npz`  points SHA-256 + exact 3-NN log scales (Eq. 3 / simple-knn convention);
- `conegs_ingp.pt`   iNGP weights + scene normalisation, used by `conegs-density` during training;
- `conegs_init.json` timings (iNGP training counts toward the method's training time).

    ss-conegs-init --source ~/datasets/mipnerf360/native/garden \
                   --output ~/datasets/conegs-init/mipnerf360/garden
"""

import argparse
import hashlib
import json
import os
import shutil
import time
from pathlib import Path

import numpy as np
import torch

from .ingp import INGP, RayCaster, gs2nerf_transform, train_ingp


def points_sha256(points):
    return hashlib.sha256(np.ascontiguousarray(points, dtype=np.float32).tobytes()).hexdigest()


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
    """log sqrt(mean of the k smallest squared neighbour distances) (simple-knn distCUDA2)."""
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
def initial_points(model, caster, K_inv, R, o, n_img, H, W, count, seed, batch=4096):
    """Uniform (image, pixel) pairs -> median-depth points (official pretrain_model loop)."""
    device = K_inv.device
    gen = torch.Generator(device=device).manual_seed(seed + 1)
    pts, cols, kept = [], [], 0
    while kept < count:
        ids = torch.randint(0, n_img, (batch,), device=device, generator=gen)
        px = torch.randint(0, W, (batch,), device=device, generator=gen)
        py = torch.randint(0, H, (batch,), device=device, generator=gen)
        origins, dirs, norm, radii = RayCaster.rays(K_inv[ids], R[ids], o[ids], px, py)
        p, c, _, _, valid = model.median_points(origins, dirs, norm, radii)
        pts.append(p[valid])
        cols.append(c[valid])
        kept += int(valid.sum())
    pts, cols = torch.cat(pts), torch.cat(cols)
    # initialize_new_points(num_to_select=P_init): multinomial over unit weights = uniform subset
    sel = torch.randperm(len(pts), device=device, generator=gen)[:count]
    return caster.to_world(pts[sel]), cols[sel].clamp(0, 1)


def main(argv=None):
    p = argparse.ArgumentParser(prog="ss-conegs-init", description=__doc__.split("\n")[0])
    p.add_argument("--source", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--iterations", type=int, default=20000)  # num_iters_pretrain
    p.add_argument("--num-init", type=int, default=1_000_000)  # num_gaussians_initialized
    p.add_argument("--budget", type=int, default=1_000_000,
                   help="primitive budget; P_init = min(num_init, budget) (0 = no budget)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args(argv)
    from splatstudio.data import load_dataset

    source, output = args.source.expanduser().resolve(), args.output.expanduser()
    if (output / "points.npz").exists() and not args.overwrite:
        raise SystemExit(f"{output}/points.npz exists; pass --overwrite to rebuild")
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    start = time.perf_counter()
    dataset = load_dataset(source)
    train = dataset["splits"]["train"]
    # Official Scene normalises with every camera pose (train + test); poses only, no images.
    c2w = np.stack([np.linalg.inv(f["viewmat"].numpy()) for f in dataset["frames"].values()])
    transform, scale = gs2nerf_transform(c2w)
    caster = RayCaster(transform, scale, device)
    shapes = {tuple(dataset["frames"][n]["image"].shape) for n in train}
    if len(shapes) != 1:
        raise SystemExit(f"Training images must share one resolution, got {sorted(shapes)}")
    images = torch.stack([dataset["frames"][n]["image"] for n in train]).to(device)
    n_img, H, W, _ = images.shape
    K_inv, R, o = camera_tensors(dataset, train, caster, device)
    load_seconds = time.perf_counter() - start

    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    t0 = time.perf_counter()
    model = INGP({"iterations": args.iterations}).to(device)
    history = train_ingp(model, images, K_inv, R, o, seed=args.seed)
    torch.cuda.synchronize(device)
    ingp_seconds = time.perf_counter() - t0

    t1 = time.perf_counter()
    count = min(args.num_init, args.budget) if args.budget else args.num_init
    points, colors = initial_points(model, caster, K_inv, R, o, n_img, H, W, count, args.seed)
    del images
    torch.cuda.empty_cache()
    log_scales = knn_log_scales(points)
    torch.cuda.synchronize(device)
    init_seconds = time.perf_counter() - t1

    copy_scene(source, output)
    pts = points.cpu().numpy().astype(np.float32)
    sha = points_sha256(pts)
    save_npz(output / "points.npz", points=pts, colors=colors.cpu().numpy().astype(np.float32))
    save_npz(output / "conegs_init.npz", points_sha256=np.array(sha),
             log_scales=log_scales.cpu().numpy().astype(np.float32))
    torch.save({"points_sha256": sha, "transform": torch.as_tensor(transform),
                "scale": scale, "cfg": dict(model.cfg), "state_dict": model.state_dict()},
               output / "conegs_ingp.pt")
    check = load_dataset(output)
    if check["dataset_id"] != dataset["dataset_id"]:
        raise SystemExit("Copied dataset hash differs from the source")
    info = {
        "source": str(source), "output": str(output.resolve()), "points_sha256": sha,
        "dataset_sha256": check["dataset_id"], "train_views": len(train),
        "image_hw": [H, W], "num_init": count, "iterations": args.iterations,
        "seed": args.seed, "normalisation_scale": scale,
        "seconds": {"load": load_seconds, "ingp_train": ingp_seconds,
                    "init_points_and_knn": init_seconds,
                    "method_total": ingp_seconds + init_seconds},
        "ingp_history": history,
        "peak_gpu_bytes": torch.cuda.max_memory_allocated(device),
        "gpu": torch.cuda.get_device_name(device),
        "log_scale_stats": {"median": float(log_scales.median()),
                            "p01": float(log_scales.quantile(0.01)),
                            "p99": float(log_scales.quantile(0.99))},
    }
    (output / "conegs_init.json").write_text(json.dumps(info, indent=1) + "\n")
    print(json.dumps({k: v for k, v in info.items() if k != "ingp_history"}, indent=1))


if __name__ == "__main__":
    main()
