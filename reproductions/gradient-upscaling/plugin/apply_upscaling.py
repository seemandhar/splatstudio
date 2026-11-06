"""Evaluate gradient-aware upscaling on a finished SplatStudio run (paper Tab. 1 'Spline (3DGS)'
/ '3DGS' rows, Tab. 2 and supp. render-time table).

For every test view of the run's scene at the dataset's FULL resolution (`--data`, default the
run's own data with downscale 1):
  native      gsplat render at full resolution (paper '3DGS' row when the run was trained with
              `--downscale s`)
  per factor s (render at floor(W/s) x floor(H/s), upscale to W x H):
    spline     analytic image derivatives (CUDA jet rasterizer) + gradient-aware spline (ours)
    spline_fd  the same spline with central-difference derivatives (ablation)
    bicubic    gsplat render + PyTorch bicubic (paper 'Bicubic' inference)
Metrics: SplatStudio's (PSNR, SSIM valid-conv, LPIPS-VGG), white background, clamp [0,1].
Speed (CUDA events, 1 warm-up + median of --repeats per view, same GPU, back to back):
native render ms vs low-res render (+ jets) ms + upscale ms; ratios only are meaningful on
shared GPUs.

Usage:
  tools/ssenv python apply_upscaling.py --run runs/<exp>/<method>/<scene> --out results/x.json \
      [--data ~/datasets/tandt_db/native/truck] [--factor 2 --factor 4] [--repeats 10]
"""

import argparse
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

import torch


class _Wrap:
    def __init__(self, fn):
        self.fn = fn

    def render(self, params, camera, sh_degree=3):
        return {"rgb": self.fn(camera, sh_degree)}


def _gpu_state():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used",
                              "--format=csv,noheader"], capture_output=True, text=True,
                             timeout=20, check=False).stdout
        return {"name": torch.cuda.get_device_name(), "nvidia_smi": out.strip().splitlines()}
    except (OSError, subprocess.SubprocessError) as exc:
        return {"name": torch.cuda.get_device_name(), "nvidia_smi_error": str(exc)}


def _time(fn, repeats):
    fn()
    times = []
    for _ in range(repeats):
        torch.cuda.synchronize()
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        times.append(s.elapsed_time(e))
    return statistics.median(times)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--data", default=None, help="full-resolution dataset (default: run's data)")
    ap.add_argument("--split", default="test")
    ap.add_argument("--factor", type=float, action="append", default=None)
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--max-views", type=int, default=0)
    ap.add_argument("--no-speed", action="store_true")
    args = ap.parse_args(argv)
    factors = args.factor or [2.0, 4.0]

    from splatstudio.data import load_dataset, on_device
    from splatstudio.runtime import measure

    from ss_gradient_upscaling.renderer import low_res_camera, render_gsplat, render_jets
    from ss_gradient_upscaling.upscale import bicubic_upscale, central_differences, spline_upscale

    run = Path(args.run)
    config = json.loads((run / "config.json").read_text())
    ckpt = torch.load(run / "checkpoint.pt", map_location="cuda", weights_only=True)
    params = {k: v.detach() for k, v in ckpt["params"].items()}
    degree = ckpt["sh_degree"]
    data = args.data or config["data"]
    dataset = load_dataset(data, 1)
    names = list(dataset["splits"][args.split])
    if args.max_views:
        names = names[: args.max_views]
        dataset = {**dataset, "splits": {**dataset["splits"], args.split: names}}

    def native(cam, d):
        return render_gsplat(params, cam, d)[0]

    def spline(s):
        def fn(cam, d):
            low = low_res_camera(cam, s)
            jets = render_jets(params, low, d)[0]
            return spline_upscale(jets[0], jets[1], jets[2], jets[3], cam["height"], cam["width"])
        return fn

    def spline_fd(s):
        def fn(cam, d):
            low = low_res_camera(cam, s)
            img = render_gsplat(params, low, d)[0]
            return spline_upscale(img, *central_differences(img), cam["height"], cam["width"])
        return fn

    def bicubic(s):
        def fn(cam, d):
            low = low_res_camera(cam, s)
            return bicubic_upscale(render_gsplat(params, low, d)[0], cam["height"], cam["width"])
        return fn

    report = {
        "run": str(run), "run_method": config["method"], "run_downscale": config["downscale"],
        "data": str(data), "scene": dataset["root"].name, "split": args.split,
        "views": len(names), "gaussians": len(params["means"]), "sh_degree": degree,
        "factors": factors, "gpu": _gpu_state(),
        "timing_note": "same-GPU medians on a shared GPU; compare ratios, not absolute ms",
    }
    t0 = time.perf_counter()
    with torch.no_grad():
        quality = {"native": measure(_Wrap(native), params, dataset, args.split, degree, "cuda")[0]}
        for s in factors:
            key = f"{s:g}x"
            quality[key] = {
                name: measure(_Wrap(fn(s)), params, dataset, args.split, degree, "cuda")[0]
                for name, fn in (("spline", spline), ("spline_fd", spline_fd),
                                 ("bicubic", bicubic))
            }
            print(json.dumps({key: quality[key]}), flush=True)
        report["quality"] = quality
        report["quality_seconds"] = time.perf_counter() - t0

        if not args.no_speed:
            rows = []
            for name in names:
                cam = on_device(dataset["frames"][name], "cuda")
                cam = {k: cam[k] for k in ("viewmat", "K", "width", "height")}
                row = {"frame": name, "native_ms": _time(lambda c=cam: native(c, degree),
                                                         args.repeats)}
                for s in factors:
                    low = low_res_camera(cam, s)
                    jets = render_jets(params, low, degree)[0]
                    img = render_gsplat(params, low, degree)[0]
                    H, W = cam["height"], cam["width"]
                    row[f"{s:g}x"] = {
                        "render_gsplat_ms": _time(lambda c=low: render_gsplat(params, c, degree),
                                                  args.repeats),
                        "render_jets_ms": _time(lambda c=low: render_jets(params, c, degree),
                                                args.repeats),
                        "upscale_bicubic_ms": _time(lambda i=img: bicubic_upscale(i, H, W),
                                                    args.repeats),
                        "upscale_spline_ms": _time(
                            lambda j=jets: spline_upscale(j[0], j[1], j[2], j[3], H, W),
                            args.repeats),
                        "spline_total_ms": _time(lambda c=cam, s=s: spline(s)(c, degree),
                                                 args.repeats),
                        "bicubic_total_ms": _time(lambda c=cam, s=s: bicubic(s)(c, degree),
                                                  args.repeats),
                    }
                rows.append(row)
            summary = {"native_ms": statistics.mean(r["native_ms"] for r in rows)}
            for s in factors:
                key = f"{s:g}x"
                avg = {k: statistics.mean(r[key][k] for r in rows) for k in rows[0][key]}
                avg["speedup_spline"] = summary["native_ms"] / avg["spline_total_ms"]
                avg["speedup_bicubic"] = summary["native_ms"] / avg["bicubic_total_ms"]
                avg["jets_overhead"] = avg["render_jets_ms"] / avg["render_gsplat_ms"]
                summary[key] = avg
            report["speed"] = summary
            report["speed_per_view"] = rows
            print(json.dumps({"speed": summary}), flush=True)
    report["gpu_after"] = _gpu_state()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    sys.exit(main())
