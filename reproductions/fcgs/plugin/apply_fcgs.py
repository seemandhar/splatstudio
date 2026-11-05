"""Apply the pretrained FCGS codec to an existing SplatStudio run and evaluate the decoded model.

FCGS needs no training, so it can compress any finished 3DGS checkpoint (e.g. the shared
`runs/baseline3dgs-<dataset>-30000/gsplat/<dataset>-<scene>` runs). This tool encodes the
checkpoint's Gaussians into real bitstreams, decodes them, renders the decoded model with the
run's own (gsplat) renderer on the requested split using SplatStudio's metric code, and writes
a JSON report. It never modifies the source run.

Usage:
  tools/ssenv python apply_fcgs.py --run <run_dir> --lmd 1e-4 [--lmd 16e-4] \
      --out <report.json> [--split test] [--xyz-coder auto|gpcc|octree] [--keep-bits <dir>]
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--lmd", type=float, action="append", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--xyz-coder", default="auto")
    ap.add_argument("--keep-bits", default=None, help="directory to keep the bitstreams in")
    ap.add_argument("--skip-original", action="store_true")
    args = ap.parse_args(argv)

    from splatstudio.backends.gsplat import create_renderer
    from splatstudio.data import load_dataset
    from splatstudio.runtime import measure
    from splatstudio.specs import ComponentSpec

    from ss_fcgs import codec

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
    renderer = create_renderer(ComponentSpec(rspec["name"], "renderer", rspec["factory"],
                                             tuple(rspec["requires"]), tuple(rspec["provides"]),
                                             rspec["config"]))
    report = {"run": str(run), "scene": dataset["root"].name, "split": args.split,
              "gaussians": len(params["means"]), "results": []}
    if not args.skip_original:
        metrics, _ = measure(renderer, params, dataset, args.split, degree, "cuda")
        report["original"] = metrics
        print(json.dumps({"original": metrics}), flush=True)
    for lmd in args.lmd:
        start = time.perf_counter()
        bits = None if args.keep_bits is None else str(Path(args.keep_bits) / report["scene"])
        decoded, enc = codec.encode_decode(params, lmd, xyz_coder=args.xyz_coder,
                                           bitstream_dir=bits)
        metrics, _ = measure(renderer, decoded, dataset, args.split, degree, "cuda")
        row = {"lambda": lmd, "metrics": metrics, "codec": enc,
               "seconds": time.perf_counter() - start}
        report["results"].append(row)
        print(json.dumps(row), flush=True)
        del decoded
        torch.cuda.empty_cache()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    sys.exit(main())
