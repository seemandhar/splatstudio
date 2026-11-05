"""Render a finished SplatStudio 3DGS run with FlashGS and compare it with the reference
rasterizers - the FlashGS paper protocol (render a 30K-iteration 3DGS model; Tab. 1, Sec. 6.5).

Quality (native resolution, the run's split, SplatStudio metric code, white background):
  * gsplat  = the run's own renderer (what the baseline's eval.json reports)
  * 3dgs    = original graphdeco CUDA forward (the paper's baseline), unmodified
  * flashgs = official FlashGS kernels, unmodified (uint8 output as the kernel writes it)
  plus image-to-image PSNR between backends ("parity").
Speed (per view: 1 warm-up + median of --repeats frames, CUDA events, same GPU, inputs already
on the GPU; FlashGS buffers reserved outside the timed region as upstream example.py does):
  at the native resolution and at each --width (1920 = paper "1080p", 3840 = "4k"; same pose
  and field of view, centred principal point).

Never modifies the source run. Timings on shared GPUs are only meaningful as ratios measured
back to back on the same device; the report records GPU name and utilisation.

Usage:
  tools/ssenv python apply_flashgs.py --run runs/baseline3dgs-tandt-30000/gsplat/tandt-truck \
      --out results/truck.json [--split test] [--width 1920 --width 3840] [--repeats 10]
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
    """Minimal renderer object for splatstudio.runtime.measure."""

    def __init__(self, fn):
        self.fn = fn

    def render(self, params, camera, sh_degree=3):
        return {"rgb": self.fn(camera)}


def _gpu_state():
    try:
        dev = torch.cuda.current_device()
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,uuid,utilization.gpu,memory.used",
             "--format=csv,noheader"], capture_output=True, text=True, timeout=20, check=False).stdout
        uuid = str(getattr(torch.cuda.get_device_properties(dev), "uuid", "n/a"))
        line = next((ln for ln in out.splitlines() if uuid[:8] in ln), out.strip())
        return {"name": torch.cuda.get_device_name(dev), "nvidia_smi": line.strip()}
    except (OSError, subprocess.SubprocessError, RuntimeError) as exc:  # informational only
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


def _summary(rows, ref):
    fl = [r["ms"]["flashgs"] for r in rows]
    rf = [r["ms"][ref] for r in rows]
    ratio = [b / a for a, b in zip(fl, rf)]
    return {
        "avg_ms": {"flashgs": statistics.mean(fl), ref: statistics.mean(rf)},
        "max_ms": {"flashgs": max(fl), ref: max(rf)},
        "avg_speedup": statistics.mean(ratio),
        "max_speedup": max(ratio),
        "min_speedup": min(ratio),
        "ratio_of_avg_times": statistics.mean(rf) / statistics.mean(fl),
    }


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--width", type=int, action="append", default=None,
                    help="extra speed-test widths (default 1920 and 3840)")
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--max-views", type=int, default=0, help="0 = all views of the split")
    ap.add_argument("--no-quality", action="store_true")
    ap.add_argument("--no-ref3dgs", action="store_true")
    args = ap.parse_args(argv)
    widths = args.width if args.width is not None else [1920, 3840]

    from splatstudio.backends.gsplat import create_renderer
    from splatstudio.data import load_dataset, on_device
    from splatstudio.runtime import measure
    from splatstudio.specs import ComponentSpec

    from ss_flashgs import core

    run = Path(args.run)
    config = json.loads((run / "config.json").read_text())
    ckpt = torch.load(run / "checkpoint.pt", map_location="cuda", weights_only=True)
    params = {k: v.detach() for k, v in ckpt["params"].items()}
    degree = ckpt["sh_degree"]
    dataset = load_dataset(config["data"], config["downscale"])
    if dataset["protocol_id"] != ckpt["protocol_id"]:
        raise SystemExit("dataset/protocol changed since the run was trained")
    rspec = next(c for c in config["specification"]["components"] if c["kind"] == "renderer")
    if rspec["factory"] != "splatstudio.backends.gsplat:create_renderer":
        raise SystemExit(f"run renderer {rspec['factory']} is not the plain gsplat renderer")
    gs_renderer = create_renderer(ComponentSpec(rspec["name"], "renderer", rspec["factory"],
                                                tuple(rspec["requires"]),
                                                tuple(rspec["provides"]), rspec["config"]))
    names = list(dataset["splits"][args.split])
    if args.max_views:
        names = names[: args.max_views]
        dataset = {**dataset, "splits": {**dataset["splits"], args.split: names}}

    white = torch.ones(3)
    white_d = white.cuda()
    prep = core.prepare(params, degree)
    flash = core.FlashGSRasterizer("cuda")
    ref = None if args.no_ref3dgs else core.Reference3DGS()
    ref_in = None if ref is None else ref.inputs(params, degree)
    report = {
        "run": str(run), "scene": dataset["root"].name, "split": args.split,
        "gaussians": len(params["means"]), "sh_degree": degree, "views": len(names),
        "gpu": _gpu_state(), "repeats": args.repeats,
        "timing_note": ("same-GPU back-to-back medians on a shared RTX A6000; absolute ms are "
                        "not comparable with the paper's RTX 3090 / A100 numbers"),
    }

    def fl_rgb(cam):
        return flash.forward(prep, cam, white).float() / 255.0

    with torch.no_grad():
        if not args.no_quality:
            t0 = time.perf_counter()
            q = {"gsplat": measure(gs_renderer, params, dataset, args.split, degree, "cuda")}
            q["flashgs"] = measure(_Wrap(fl_rgb), params, dataset, args.split, degree, "cuda")
            if ref is not None:
                q["3dgs"] = measure(_Wrap(lambda c: ref.forward(ref_in, c, white_d)),
                                    params, dataset, args.split, degree, "cuda")
            q["gsplat_uint8_trunc"] = measure(
                _Wrap(lambda c: core.quantize_trunc(
                    gs_renderer.render(params, c, degree)["rgb"].clamp(0, 1)).float() / 255),
                params, dataset, args.split, degree, "cuda")
            parity = []
            for name in names:
                cam = on_device(dataset["frames"][name], "cuda")
                a = fl_rgb(cam)
                wrap = flash.edge_wrap_tiles(cam["width"])
                g = gs_renderer.render(params, cam, degree)["rgb"].clamp(0, 1)
                row = {"frame": name, "flashgs_vs_gsplat": core.psnr(a, g),
                       "pairs_flashgs": flash.last["num_rendered"], "edge_wrap_tiles": wrap}
                if ref is not None:
                    c3 = ref.forward(ref_in, cam, white_d).clamp(0, 1)
                    row["flashgs_vs_3dgs"] = core.psnr(a, c3)
                    row["flashgs_vs_3dgs_uint8_trunc"] = core.psnr(
                        a, core.quantize_trunc(c3).float() / 255)
                    row["3dgs_vs_gsplat"] = core.psnr(c3, g)
                    row["pairs_3dgs"] = ref.last["num_rendered"]
                parity.append(row)
            keys = [k for k in parity[0] if k != "frame"]
            report["quality"] = {k: v[0] for k, v in q.items()}
            report["parity_mean"] = {k: statistics.mean(r[k] for r in parity) for k in keys}
            report["parity_min_psnr"] = {k: min(r[k] for r in parity) for k in keys
                                         if not k.startswith(("pairs", "edge"))}
            report["parity_per_view"] = parity
            report["quality_seconds"] = time.perf_counter() - t0
            print(json.dumps({"quality": report["quality"], "parity": report["parity_mean"]}),
                  flush=True)

        report["speed"] = {}
        for width in [None, *widths]:
            label = "native" if width is None else ("1080p" if width == 1920 else
                                                    "4k" if width == 3840 else f"w{width}")
            rows = []
            for name in names:
                cam = on_device(dataset["frames"][name], "cuda")
                cam = {k: cam[k] for k in ("viewmat", "K", "width", "height")}
                if width is not None:
                    cam = core.scaled_camera(cam, width)
                flash.reserve(prep, cam)
                ms = {"flashgs": _time(lambda cam=cam: flash.forward(prep, cam, white, reserve=False),
                                       args.repeats)}
                pairs = {"flashgs": flash.last["num_rendered"]}
                ms["gsplat"] = _time(lambda cam=cam: core.gsplat_forward(params, cam, degree),
                                     args.repeats)
                if ref is not None:
                    ms["3dgs"] = _time(lambda cam=cam: ref.forward(ref_in, cam, white_d), args.repeats)
                    pairs["3dgs"] = ref.last["num_rendered"]
                rows.append({"frame": name, "width": cam["width"], "height": cam["height"],
                             "ms": ms, "pairs": pairs})
            entry = {"resolution": [rows[0]["width"], rows[0]["height"]],
                     "vs_gsplat": _summary(rows, "gsplat"), "per_view": rows}
            if ref is not None:
                entry["vs_3dgs"] = _summary(rows, "3dgs")
                entry["pair_reduction_vs_3dgs"] = 1 - (
                    sum(r["pairs"]["flashgs"] for r in rows) / sum(r["pairs"]["3dgs"] for r in rows))
            report["speed"][label] = entry
            print(json.dumps({label: {k: v for k, v in entry.items() if k != "per_view"}}),
                  flush=True)
    report["gpu_after"] = _gpu_state()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    sys.exit(main())
