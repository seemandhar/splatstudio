"""EDGS correspondence initialization as a preprocessing step (training views only).

Writes a copy of a native SplatStudio scene with an EDGS `points.npz` plus `edgs_init.npz`
(per-point log scales, opacity logits, SH rest and the hash that ties them to the points).

    ss-edgs-init --source ~/datasets/mipnerf360/native/garden \
                 --output ~/datasets/edgs-init/mipnerf360/garden --matches-per-ref 25000

Port of `init_gaussians_with_corr` in https://github.com/CompVis/EDGS (source/corr_init.py).
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
import torch.nn.functional as F

from . import core

DEFAULT_MODELS = Path.home() / "models" / "roma"


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def points_sha256(points):
    return hashlib.sha256(np.ascontiguousarray(points, dtype=np.float32).tobytes()).hexdigest()


def load_roma(device, weights, dinov2_weights, model="outdoors"):
    from romatch import roma_indoor, roma_outdoor

    w = torch.load(weights, map_location=device, weights_only=True)
    dw = torch.load(dinov2_weights, map_location=device, weights_only=True)
    build = roma_indoor if model == "indoors" else roma_outdoor
    roma = build(device=device, weights=w, dinov2_weights=dw)
    roma.upsample_preds = False  # official corr_init.py:538-539
    roma.symmetric = False
    return roma


def to_pil(image):
    from PIL import Image

    return Image.fromarray((image.clamp(0, 1).cpu().numpy() * 255).round().astype(np.uint8))


@torch.no_grad()
def match_pair(roma, image_a, image_b, height, width, device):
    """RoMa warp A->B and certainty, bilinearly resized to the image resolution (official
    resize_batch). Returns certainty [H,W] and B coordinates in NDC [H,W,2]."""
    warp, certainty = roma.match(to_pil(image_a), to_pil(image_b), device=device)
    flow = warp[..., 2:].permute(2, 0, 1)[None]
    flow = F.interpolate(flow, size=(height, width), mode="bilinear", align_corners=False)
    cert = F.interpolate(certainty[None, None], size=(height, width), mode="bilinear",
                         align_corners=False)
    return cert[0, 0], flow[0].permute(1, 2, 0)


@torch.no_grad()
def initialize_scene(dataset, roma, cfg, device, log=print):
    names = dataset["splits"]["train"]  # training views only
    frames = [dataset["frames"][n] for n in names]
    viewmats = torch.stack([f["viewmat"] for f in frames])
    refs = core.select_reference_views(viewmats, cfg["num_refs"], seed=cfg["seed"])
    nns = core.nearest_cameras(viewmats, cfg["nns_per_ref"]).cpu().numpy()
    gen = torch.Generator(device=device).manual_seed(cfg["seed"])
    proj = [core.ndc_projection(f["K"], f["viewmat"], f["width"], f["height"]).to(device)
            for f in frames]
    centres = [core.camera_center(f["viewmat"].double()).to(device) for f in frames]
    out = {k: [] for k in ("points", "colors", "log_scales", "error", "dirs", "obs", "ref_view",
                          "obs_view")}
    timing = {"matching": 0.0, "triangulation": 0.0}
    total_sampled = 0
    for n_ref, i in enumerate(refs):
        fa = frames[i]
        H, W = fa["height"], fa["width"]
        img_a = fa["image"].to(device)
        t0 = time.perf_counter()
        certs, flows = [], []
        for j in nns[i]:
            c, fl = match_pair(roma, img_a, frames[j]["image"], H, W, device)
            certs.append(c)
            flows.append(fl)
        torch.cuda.synchronize(device)
        timing["matching"] += time.perf_counter() - t0
        t0 = time.perf_counter()
        cert = torch.stack(certs).max(dim=0).values  # c^i = max_j c^ij (supp. Sec. C)
        weights = core.sampling_weights(cert.reshape(-1), cfg["corr_threshold"])
        m = min(cfg["matches_per_ref"], int((weights > 0).sum()))
        idx = torch.multinomial(weights, m, replacement=False, generator=gen)
        total_sampled += m
        uv_a = core.pixel_centres_ndc(idx, W, H).double()
        pts, e1s, e2s = [], [], []
        for k, j in enumerate(nns[i]):
            uv_b = flows[k].reshape(-1, 2)[idx].double()
            x, e1, e2 = core.triangulate(proj[i], proj[j], uv_a, uv_b)
            pts.append(x)
            e1s.append(e1)
            e2s.append(e2)
        x, err, best = core.select_best(torch.stack(pts), torch.stack(e1s), torch.stack(e2s))
        keep = err <= cfg["proj_threshold"]  # official: opacity logit -10 -> pruned at step 1
        x, best, sel = x[keep], best[keep], idx[keep]
        color_a = img_a.reshape(-1, 3)[sel]
        # Best-neighbour colour at the matched pixel (second SH observation).
        uv_b_all = torch.stack([flows[k].reshape(-1, 2)[sel] for k in range(len(nns[i]))])
        uv_b = uv_b_all[best, torch.arange(len(sel), device=device)]
        color_b = torch.empty_like(color_a)
        dir_b = torch.empty_like(x)
        for k, j in enumerate(nns[i]):
            mk = best == k
            if mk.any():
                fb = frames[j]
                col, row = core.ndc_to_pixel(uv_b[mk], fb["width"], fb["height"])
                color_b[mk] = fb["image"].to(device)[row, col]
                dir_b[mk] = x[mk] - centres[j][None]
        out["points"].append(x.float())
        out["colors"].append(color_a)
        out["log_scales"].append(
            core.log_isotropic_scale(x, centres[i], cfg["scaling_factor"], 0.5).float())
        out["error"].append(err[keep].float())
        out["dirs"].append(torch.stack([x - centres[i][None], dir_b], dim=1).float())
        out["obs"].append(torch.stack([color_a, color_b], dim=1))
        out["ref_view"].append(torch.full((len(sel),), i, dtype=torch.int32, device=device))
        out["obs_view"].append(torch.as_tensor(nns[i], device=device)[best].to(torch.int32))
        timing["triangulation"] += time.perf_counter() - t0
        if n_ref % 20 == 0 or n_ref == len(refs) - 1:
            log(f"[edgs-init] ref {n_ref + 1}/{len(refs)} kept {int(keep.sum())}/{m}")
    res = {k: torch.cat(v) for k, v in out.items()}
    t0 = time.perf_counter()
    res["shN"] = core.fit_sh_residual(res["dirs"], res["obs"])  # Sec. 3.5 (residual form)
    timing["sh_fit"] = time.perf_counter() - t0
    stats = {"reference_views": [names[i] for i in refs], "num_reference_views": len(refs),
             "sampled": total_sampled, "kept": len(res["points"]), "timing_seconds": timing}
    return res, stats


SH_FIT = {"form": "residual, reference-direction constrained, ridge",
          "lambda": core.RIDGE_LAMBDA, "observations": "reference + best-neighbour colour"}


def sh_stats(shn):
    m = shn.abs().reshape(len(shn), -1).max(dim=1).values
    return {"max_abs": float(m.max()), "frac_gt_2": float((m > 2).float().mean()),
            "median_point_max_abs": float(m.median()),
            "p99_point_max_abs": float(m.float().quantile(0.99)) if len(m) < 1 << 24 else None}


def save_npz(path, **arrays):
    tmp = path.with_name(path.stem + ".tmp.npz")
    np.savez(tmp, **arrays)
    os.replace(tmp, path)


def refit_sh(output, device):
    """Recompute shN only, from the cached obs/dirs; points and the SHA link are untouched."""
    with np.load(output / "edgs_init.npz", allow_pickle=False) as data:
        arrays = {k: data[k] for k in data.files}
    dirs = torch.from_numpy(arrays["dirs"]).to(device)
    obs = torch.from_numpy(arrays["obs"]).to(device)
    shn = core.fit_sh_residual(dirs, obs)
    arrays["shN"] = shn.cpu().numpy().astype(np.float32)
    save_npz(output / "edgs_init.npz", **arrays)
    info = json.loads((output / "edgs_init.json").read_text())
    info.update(sh=sh_stats(shn), sh_fit=SH_FIT)
    (output / "edgs_init.json").write_text(json.dumps(info, indent=1) + "\n")
    print(json.dumps(info["sh"]))


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


def main(argv=None):
    p = argparse.ArgumentParser(prog="ss-edgs-init", description=__doc__.split("\n")[0])
    p.add_argument("--source", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--matches-per-ref", type=int, default=25000)  # full_eval.py (M360)
    p.add_argument("--nns-per-ref", type=int, default=3)  # full_eval.py
    p.add_argument("--num-refs", type=int, default=180)
    p.add_argument("--scaling-factor", type=float, default=0.001)
    p.add_argument("--proj-threshold", type=float, default=0.01)
    p.add_argument("--corr-threshold", type=float, default=0.05)
    p.add_argument("--roma-model", choices=["outdoors", "indoors"], default="outdoors")
    p.add_argument("--roma-weights", type=Path, default=DEFAULT_MODELS / "roma_outdoor.pth")
    p.add_argument("--dinov2-weights", type=Path,
                   default=DEFAULT_MODELS / "dinov2_vitl14_pretrain.pth")
    p.add_argument("--seed", type=int, default=228)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--overwrite", action="store_true",
                   help="rewrite points.npz/edgs_init.* of an existing output (SHA link kept)")
    p.add_argument("--refit-sh", action="store_true",
                   help="only refit shN from the cached observations in --output/edgs_init.npz")
    args = p.parse_args(argv)
    if args.refit_sh:
        return refit_sh(args.output.expanduser(), args.device)
    from splatstudio.data import load_dataset

    source, output = args.source.expanduser().resolve(), args.output.expanduser()
    if (output / "points.npz").exists() and not args.overwrite:
        raise SystemExit(f"{output}/points.npz exists; remove the directory to rebuild")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    start = time.perf_counter()
    dataset = load_dataset(source)
    roma = load_roma(args.device, args.roma_weights, args.dinov2_weights, args.roma_model)
    cfg = {"matches_per_ref": args.matches_per_ref, "nns_per_ref": args.nns_per_ref,
           "num_refs": args.num_refs, "scaling_factor": args.scaling_factor,
           "proj_threshold": args.proj_threshold, "corr_threshold": args.corr_threshold,
           "roma_model": args.roma_model, "seed": args.seed}
    torch.cuda.reset_peak_memory_stats(args.device)
    res, stats = initialize_scene(dataset, roma, cfg, args.device)
    copy_scene(source, output)
    points = res["points"].cpu().numpy().astype(np.float32)
    colors = res["colors"].clamp(0, 1).cpu().numpy().astype(np.float32)
    sha = points_sha256(points)
    previous = None
    if (output / "edgs_init.json").exists():
        previous = json.loads((output / "edgs_init.json").read_text()).get("points_sha256")
    save_npz(output / "points.npz", points=points, colors=colors)
    save_npz(output / "edgs_init.npz", points_sha256=np.array(sha),
             log_scales=res["log_scales"].cpu().numpy().astype(np.float32),
             opacity_logits=np.zeros(len(points), np.float32),
             shN=res["shN"].cpu().numpy().astype(np.float32),
             reprojection_error=res["error"].cpu().numpy().astype(np.float32),
             obs=res["obs"].cpu().numpy().astype(np.float32),
             dirs=res["dirs"].cpu().numpy().astype(np.float32),
             ref_view=res["ref_view"].cpu().numpy(), obs_view=res["obs_view"].cpu().numpy())
    stats["sh"] = sh_stats(res["shN"])
    stats["sh_fit"] = SH_FIT
    stats["previous_points_sha256"] = previous
    check = load_dataset(output)
    info = {
        "source": str(source), "output": str(output.resolve()), "config": cfg,
        "points_sha256": sha, "dataset_sha256": check["dataset_id"],
        "source_dataset_sha256": dataset["dataset_id"],
        "train_views": dataset["splits"]["train"], **stats,
        "seconds_total": time.perf_counter() - start,
        "peak_gpu_bytes": torch.cuda.max_memory_allocated(args.device),
        "gpu": torch.cuda.get_device_name(args.device),
        "roma": {"package_commit": "370117431ffc5dc000fb46f6e581b74bdb2c3ff8",
                 "weights": str(args.roma_weights), "weights_sha256": file_sha256(args.roma_weights),
                 "dinov2": str(args.dinov2_weights),
                 "dinov2_sha256": file_sha256(args.dinov2_weights)},
    }
    if check["dataset_id"] != dataset["dataset_id"]:
        raise SystemExit("Copied dataset hash differs from the source")
    (output / "edgs_init.json").write_text(json.dumps(info, indent=1) + "\n")
    print(json.dumps({k: v for k, v in info.items() if k not in ("train_views",
                                                              "reference_views")}, indent=1))


if __name__ == "__main__":
    main()
