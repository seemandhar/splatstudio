"""Calibrate the FreGS densification threshold by Gaussian-count matching (plan 6.2, review-1).

Like-for-like: gsplat runs IN THE SAME 15000-step sweep (same steps, means-LR schedule, seed and
recipe) as the fregs-calib-gXXXX candidates; N(15000) of each candidate is matched to gsplat's.
Densification stops at 15000, so N(15000) is the final count.

Blinding (non-destructive): only `training.jsonl` is ever opened. The metric files bench writes
(validation.json, test.json, <run>.eval.json) are left in place (bench marks a job done only if
test.json exists), are never opened or parsed, and their sha256 is recorded in a BLINDED.json
marker in every run dir and in the experiment dir, plus in the report. Do not run
`bench.py summarize` on the calibration experiment.

Sweep (3 scenes):
  python3 tools/bench.py launch --exp fregs-calib-threshold --dataset mipnerf360 \
      --scenes bicycle garden room --steps 15000 --methods gsplat \
      fregs-calib-g0003 fregs-calib-g0004 fregs-calib-g0005 fregs-calib-g0006
  tools/ssenv python reproductions/fregs/plugin/calibrate_threshold.py \
      --exp fregs-calib-threshold --output reproductions/fregs/evidence/threshold_calibration.json
"""

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

from ss_fregs.defaults import CALIBRATED, DEFAULTS, THRESHOLD_GRID

RUNS = Path("/home/sejain/splatify/runs")
READABLE = {"training.jsonl", "config.json"}  # the only files this script may open (no metrics)
METRIC_FILES = ("validation.json", "test.json")
NOTE = ("Calibration run for FreGS grow_grad2d (Gaussian-count matching). Image metrics are "
        "BLINDED: never opened, must not be summarized or used for any selection.")


def read_log(path):
    path = Path(path)
    if path.name not in READABLE:
        raise PermissionError(f"Refusing to read {path}: only training logs may be read")
    with path.open() as log:
        return [json.loads(line) for line in log]


def check_weights(run):
    """The candidate run must have been trained with the calibrated DEFAULTS w_l/w_h."""
    path = run / "config.json"
    if path.name not in READABLE:
        raise PermissionError(path)
    spec = json.loads(path.read_text())["specification"]
    cfg = next(c for c in spec["components"] if c["name"] == "fregs-frequency")["config"]
    for key in ("w_l", "w_h"):
        if cfg[key] != DEFAULTS["frequency"][key]:
            raise SystemExit(f"{run}: trained with {key}={cfg[key]}, DEFAULTS has "
                             f"{DEFAULTS['frequency'][key]}; relaunch the sweep")


def count_at(run, step):
    for row in read_log(run / "training.jsonl"):
        if row["step"] == step:
            return int(row["gaussians"])
    raise SystemExit(f"{run}: no training-log row at step {step}")


def sha256(path):
    # Hashes raw bytes for provenance only; the content is never decoded or parsed.
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def blind(run):
    files = [run / n for n in METRIC_FILES] + [run.with_name(run.name + ".eval.json")]
    hashes = {str(p): sha256(p) for p in files if p.exists()}
    marker = {"blinded": True, "note": NOTE, "metric_files_sha256": hashes,
              "timestamp": datetime.now(timezone.utc).isoformat()}
    (run / "BLINDED.json").write_text(json.dumps(marker, indent=2, sort_keys=True) + "\n")
    return hashes


def solve(grid, values):
    """Linear interpolation of the zero of mean log(N_fregs/N_gsplat) over the grid."""
    pairs = sorted(zip(grid, values))
    for (g0, v0), (g1, v1) in zip(pairs, pairs[1:]):
        if v0 == 0:
            return g0, "grid_point"
        if v0 * v1 < 0:
            return g0 + (g1 - g0) * (0 - v0) / (v1 - v0), "interpolated"
    if pairs[-1][1] == 0:
        return pairs[-1][0], "grid_point"
    return min(pairs, key=lambda gv: abs(gv[1]))[0], "out_of_grid_nearest"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--exp", required=True)
    p.add_argument("--baseline-method", default="gsplat", help="baseline run inside the same --exp")
    p.add_argument("--dataset", default="mipnerf360")
    p.add_argument("--scenes", nargs="+", default=["bicycle", "garden", "room"])
    p.add_argument("--grid", nargs="+", type=float, default=list(THRESHOLD_GRID))
    p.add_argument("--step", type=int, default=15000)
    p.add_argument("--runs-root", default=str(RUNS))
    p.add_argument("--output")
    args = p.parse_args()
    if not (CALIBRATED["w_l"] and CALIBRATED["w_h"]):
        raise SystemExit("Refusing to run: the fregs-calib-* candidates inherit w_l/w_h from DEFAULTS, "
                         "which are not calibrated yet (set CALIBRATED['w_l'] and ['w_h'] first, "
                         "reinstall, then launch the sweep)")
    exp = Path(args.runs_root) / args.exp
    report = {
        "procedure": "plan.md 6.2 (review-1): N(15000) from training.jsonl; baseline = "
        f"{args.baseline_method} in the same sweep",
        "label": "calibrated (Gaussian-count matched, no image metrics)",
        "blinded": True,
        "frequency_weights_used": {k: DEFAULTS["frequency"][k] for k in ("w_l", "w_h")},
        "assumption": "N_fregs/N_gsplat matched under a 15000-step schedule is assumed to transfer "
        "to 30000-step runs (higher means LR during densification); untested (gap G4)",
        "scenes": args.scenes,
        "step": args.step,
        "baseline": {},
        "candidates": {},
        "blinded_metric_files_sha256": {},
    }
    runs = {}
    for scene in args.scenes:
        run = exp / args.baseline_method / f"{args.dataset}-{scene}"
        report["baseline"][scene] = count_at(run, args.step)
        runs[str(run)] = run
    mean_logs = []
    for g in args.grid:
        name = f"fregs-calib-g{round(g * 1e4):04d}"
        counts, logs = {}, []
        for scene in args.scenes:
            run = exp / name / f"{args.dataset}-{scene}"
            check_weights(run)
            counts[scene] = count_at(run, args.step)
            logs.append(math.log(counts[scene] / report["baseline"][scene]))
            runs[str(run)] = run
        mean_logs.append(sum(logs) / len(logs))
        report["candidates"][name] = {"grow_grad2d": g, "gaussians": counts,
                                      "mean_log_ratio": mean_logs[-1]}
    for run in runs.values():
        report["blinded_metric_files_sha256"].update(blind(run))
    (exp / "BLINDED.json").write_text(json.dumps(
        {"blinded": True, "note": NOTE, "runs": sorted(runs),
         "timestamp": datetime.now(timezone.utc).isoformat()}, indent=2) + "\n")
    value, how = solve(args.grid, mean_logs)
    report["grow_grad2d"] = round(value, 5)
    report["solution"] = how
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
