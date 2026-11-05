"""Secondary speed measurement: forward rasterization time of a trained run's test views with the
Speedy-Splat (SnugBox+AccuTile) and the stock 3D-GS (3-sigma) rasterizers on the same GPU.

Timing follows the paper (CUDA events around the forward rasterizer, no backward). Numbers are
GPU/load specific (shared A6000s here, RTX A5000 in the paper); only ratios are comparable.

    tools/ssenv python -m ss_speedysplat.fps --run runs/<exp>/<scene> [--repeats 5]
"""

import argparse
import json
from pathlib import Path

import torch


def measure(params, cameras, sh_degree, rasterizer, repeats=5, background=(0.0, 0.0, 0.0)):
    from .raster import render

    dev = params["means"].device
    bg = torch.tensor(background, dtype=torch.float32, device=dev)
    times, pairs = [], 0
    with torch.no_grad():
        for cam in cameras[:2]:  # warm-up (JIT, allocator)
            render(params, cam, sh_degree, bg, rasterizer)
        for _ in range(repeats):
            for cam in cameras:
                start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
                    enable_timing=True)
                start.record()
                _, _, _, stats = render(params, cam, sh_degree, bg, rasterizer)
                stop.record()
                torch.cuda.synchronize(dev)
                times.append(start.elapsed_time(stop))
                pairs += stats["num_rendered"]
    ms = sum(times) / len(times)
    return {"ms_per_frame": ms, "fps": 1000.0 / ms,
            "tile_gaussian_pairs_per_frame": pairs / len(times)}


def main(argv=None):
    from splatstudio.data import load_dataset

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--output", default=None)
    a = ap.parse_args(argv)
    run = Path(a.run)
    cfg = json.loads((run / "config.json").read_text())
    ck = torch.load(run / "checkpoint.pt", map_location=a.device, weights_only=True)
    params = {k: v.detach() for k, v in ck["params"].items()}
    ds = load_dataset(cfg["data"], cfg.get("downscale", 1))
    cams = [{k: (v.to(a.device) if torch.is_tensor(v) else v)
             for k, v in ds["frames"][n].items() if k in ("K", "viewmat", "width", "height")}
            for n in ds["splits"][a.split]]
    out = {"run": str(run), "gaussians": len(params["means"]), "views": len(cams),
           "gpu": torch.cuda.get_device_name(a.device),
           "note": "secondary, GPU/load specific; compare ratios only"}
    for r in ("stock", "speedy"):
        out[r] = measure(params, cams, ck["sh_degree"], r, a.repeats)
    out["speedup_speedy_vs_stock"] = out["stock"]["ms_per_frame"] / out["speedy"]["ms_per_frame"]
    out["pair_reduction"] = (out["stock"]["tile_gaussian_pairs_per_frame"]
                             / out["speedy"]["tile_gaussian_pairs_per_frame"])
    text = json.dumps(out, indent=1)
    print(text)
    Path(a.output or run / "speedy_fps.json").write_text(text)


if __name__ == "__main__":
    main()
