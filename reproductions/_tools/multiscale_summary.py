#!/usr/bin/env python3
"""Per-scale metrics for multi-scale (MTMT) runs, e.g. on `mipnerf360-ms`.

Reads <run>/test.json `per_view` (frame paths images/s{1,2,4,8}/...) and reports PSNR/SSIM/LPIPS
at 1x/2x/4x/8x and their average. Averaging follows the official render_ms.py + paper tables:
per scene, mean over test views at each scale (render_ms.py render_set), then the unweighted
mean over the 4 scales (supp. 360_multi_per.tex "Avg"; e.g. garden Ours 27.13/29.53/31.20/32.19
-> 30.01). Across scenes (main Tab. 2): mean over scenes at each scale, then over scales.

    python3 tools/multiscale_summary.py runs/<exp>/<method>/mipnerf360-ms-garden
    python3 tools/multiscale_summary.py --exp <exp> --method analytic-splatting [--json]
    python3 tools/multiscale_summary.py --exp <exp> --dataset mipnerf360-msgs --scales 1,4,16,64
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

RUNS = Path(os.environ.get("SPLATIFY_RUNS", Path(__file__).resolve().parent.parent / "runs"))
SCALES = (1, 2, 4, 8)
METRICS = ("psnr", "ssim", "lpips")
SCALE_RE = re.compile(r"(?:^|/)s(\d+)(?:_dup\d+)?/")


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def run_summary(run, split="test"):
    result = json.loads((Path(run) / f"{split}.json").read_text())
    by_scale = {s: [] for s in SCALES}
    for row in result["per_view"]:
        m = SCALE_RE.search(row["frame"])
        if not m:
            raise SystemExit(f"{run}: cannot infer scale from frame {row['frame']!r}")
        if int(m.group(1)) in by_scale:
            by_scale[int(m.group(1))].append(row)
    out = {"run": str(run), "split": split, "views": {s: len(v) for s, v in by_scale.items()}}
    for k in METRICS:
        per = {s: mean([r.get(k) for r in v]) for s, v in by_scale.items()}
        out[k] = {**{f"{s}x": per[s] for s in SCALES}, "avg": mean(per.values())}
    return out


def fmt(v, k):
    return "—" if v is None else (f"{v:.2f}" if k == "psnr" else f"{v:.3f}" if k == "ssim" else f"{v:.4f}")


def table(rows):
    lines = []
    for k in METRICS:
        lines.append(f"| {k.upper()} | " + " | ".join(f"{s}x" for s in SCALES) + " | avg |")
        lines.append("|---" * (len(SCALES) + 2) + "|")
        for name, r in rows:
            lines.append(f"| {name} | " + " | ".join(fmt(r[k][f"{s}x"], k) for s in SCALES)
                         + f" | {fmt(r[k]['avg'], k)} |")
        lines.append("")
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("runs", nargs="*", help="run directories containing test.json")
    p.add_argument("--exp")
    p.add_argument("--method")
    p.add_argument("--dataset", default="mipnerf360-ms")
    p.add_argument("--split", default="test")
    p.add_argument("--json", action="store_true")
    p.add_argument("--scales", default="1,2,4,8",
                   help="comma list of scales to report/average (MS-GS mipnerf360-msgs: 1,4,16,64)")
    a = p.parse_args()
    global SCALES
    SCALES = tuple(int(s) for s in a.scales.split(","))
    runs = [Path(r) for r in a.runs]
    if a.exp:
        base = RUNS / a.exp
        methods = [a.method] if a.method else sorted(d.name for d in base.iterdir() if d.is_dir())
        for m in methods:
            runs += sorted(d for d in (base / m).glob(f"{a.dataset}-*") if (d / f"{a.split}.json").exists())
    if not runs:
        sys.exit("no runs with test.json found")
    rows = [(f"{r.parent.name}/{r.name}", run_summary(r, a.split)) for r in runs]
    # cross-scene mean per method: mean over scenes at each scale, then over scales
    groups = {}
    for name, r in rows:
        groups.setdefault(name.split("/")[0], []).append(r)
    for m, rs in groups.items():
        if len(rs) > 1:
            agg = {}
            for k in METRICS:
                per = {f"{s}x": mean([r[k][f"{s}x"] for r in rs]) for s in SCALES}
                agg[k] = {**per, "avg": mean(per.values())}
            rows.append((f"{m}/mean({len(rs)} scenes)", agg))
    if a.json:
        print(json.dumps(dict(rows), indent=1))
    else:
        print(table(rows))
        print("views per scale:", {n: r["views"] for n, r in rows if "views" in r})


if __name__ == "__main__":
    main()
