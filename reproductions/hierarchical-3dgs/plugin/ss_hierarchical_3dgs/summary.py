"""Mip-NeRF 360 summary in the paper's terms (Sec. 6.2), from eval_taus outputs.

    tools/ssenv python -m ss_hierarchical_3dgs.summary --ours runs/<exp> \
        --baseline runs/baseline3dgs-mipnerf360-1600-30000 runs/baseline3dgs-mipnerf360-1600b-30000 \
        [--channel-baseline] [--output FILE]

Scene runs are found as `<root>/<method>/<dataset>-<scene>/` (tools/bench.py layout).
Ours: `test_tau-0.5.json` = leaves, `test_tau{0,3,6,15}.json` = opt, `test_basic_tau*.json` =
basic. Baseline: `test.json` (full-image PSNR); with --channel-baseline its checkpoint is
re-rendered once per scene (eval_taus.channel_only, cached in `<ours>/baseline_channel/`) for
the per-channel PSNR that the official render_hierarchy.py reports.

Reported, each with the full-image PSNR (runtime) and the per-channel PSNR (official
image_utils.psnr): 9-scene and 7-scene (no flowers/treehill, the original public release)
scene means; leaves - 3DGS on the 7-scene mean vs the paper's -0.15 (their 3DGS 29.11 matches a
7-scene mean, not 9); tau-row deltas (row - leaves) vs the paper's, which do not depend on the
3DGS run. A mean over missing scenes is not reported.
"""

import argparse
import json
from pathlib import Path

SCENES9 = ("bicycle", "flowers", "garden", "stump", "treehill", "room", "counter", "kitchen",
           "bonsai")
SCENES7 = tuple(s for s in SCENES9 if s not in ("flowers", "treehill"))
# paper_results.json datasets.mipnerf360 (evaluation.tex Sec. 6.2)
PAPER = {"3dgs": 29.11, "leaves": 28.96,
         "opt": {3: 28.87, 6: 28.52, 15: 27.25}, "basic": {3: 28.86, 6: 28.05, 15: 24.82}}
METRICS = ("psnr", "psnr_channel")


def scene_runs(root):
    out = {}
    for d in sorted(Path(root).glob("*/*")):
        if d.is_dir() and (d / "config.json").exists():
            scene = d.name.rsplit("-", 1)[-1]
            if scene in SCENES9:
                out[scene] = d
    return out


def _view_mean(path, key):
    if not path.exists():
        return None
    views = json.loads(path.read_text())["per_view"]
    if not views or key not in views[0]:
        return None
    return sum(v[key] for v in views) / len(views)


def _mean(per_scene, scenes):
    vals = [per_scene.get(s) for s in scenes]
    return None if any(v is None for v in vals) else sum(vals) / len(vals)


def collect(ours_root, baseline_roots, channel_baseline=False, device="cuda:0"):
    ours = scene_runs(ours_root)
    base = {}
    for r in baseline_roots:
        base.update(scene_runs(r))
    rows = {"leaves": ("", -0.5)}
    for kind, tag in (("opt", ""), ("basic", "basic_")):
        for tau in (0, 3, 6, 15):
            rows[f"{kind}_tau{tau}"] = (tag, tau)
    table = {}  # row -> metric -> scene -> value
    for row, (tag, tau) in rows.items():
        table[row] = {m: {} for m in METRICS}
        for scene, run in ours.items():
            for m in METRICS:
                v = _view_mean(run / f"test_{tag}tau{tau:g}.json", m)
                if v is not None:
                    table[row][m][scene] = v
    table["3dgs"] = {m: {} for m in METRICS}
    cache = Path(ours_root) / "baseline_channel"
    for scene, run in base.items():
        table["3dgs"]["psnr"][scene] = _view_mean(run / "test.json", "psnr")
        f = cache / f"{scene}.json"
        if channel_baseline and not f.exists():
            from .eval_taus import channel_only

            channel_only(run, f, device=device)
        v = _view_mean(f, "psnr_channel")
        if v is not None:
            table["3dgs"]["psnr_channel"][scene] = v
    return table, {s: str(r) for s, r in ours.items()}, {s: str(r) for s, r in base.items()}


def summarize(table):
    means = {row: {m: {"9": _mean(v[m], SCENES9), "7": _mean(v[m], SCENES7),
                       "n_scenes": len(v[m])} for m in METRICS}
             for row, v in table.items()}

    def d(a, b):
        return None if a is None or b is None else a - b

    head = {}
    for m in METRICS:
        lv = means["leaves"][m]
        h = {"leaves_minus_3dgs_7scene": d(lv["7"], means["3dgs"][m]["7"]),
             "leaves_minus_3dgs_9scene": d(lv["9"], means["3dgs"][m]["9"]),
             "paper_leaves_minus_3dgs": PAPER["leaves"] - PAPER["3dgs"], "tau_deltas": {}}
        for kind in ("opt", "basic"):
            for tau in (0, 3, 6, 15):
                r = means[f"{kind}_tau{tau}"][m]
                paper = PAPER[kind].get(tau)
                h["tau_deltas"][f"{kind}_tau{tau}"] = {
                    "ours_9scene": d(r["9"], lv["9"]), "ours_7scene": d(r["7"], lv["7"]),
                    "paper": None if paper is None else paper - PAPER["leaves"]}
        head[m] = h
    return means, head


def _fmt(v):
    return "  -   " if v is None else f"{v:6.2f}"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ours", required=True)
    ap.add_argument("--baseline", nargs="+", required=True)
    ap.add_argument("--channel-baseline", action="store_true",
                    help="re-render baseline checkpoints for per-channel PSNR (cached)")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--output")
    a = ap.parse_args(argv)
    table, ours, base = collect(a.ours, a.baseline, a.channel_baseline, a.device)
    means, head = summarize(table)
    out = {"per_scene": table, "means": means, "headline": head, "paper": PAPER,
           "ours_runs": ours, "baseline_runs": base,
           "scene_sets": {"9": SCENES9, "7": SCENES7}}
    if a.output:
        Path(a.output).write_text(json.dumps(out, indent=1))
    for m in METRICS:
        print(f"\n[{m}]  row: 9-scene / 7-scene mean (n scenes)")
        for row, v in means.items():
            print(f"  {row:13s} {_fmt(v[m]['9'])} {_fmt(v[m]['7'])}  ({v[m]['n_scenes']})")
        h = head[m]
        print(f"  leaves-3DGS 7-scene {_fmt(h['leaves_minus_3dgs_7scene'])}  "
              f"(9-scene {_fmt(h['leaves_minus_3dgs_9scene'])}; paper "
              f"{h['paper_leaves_minus_3dgs']:+.2f})")
        for row, t in h["tau_deltas"].items():
            print(f"  {row:13s} - leaves: 9 {_fmt(t['ours_9scene'])} 7 {_fmt(t['ours_7scene'])}"
                  f"  paper {_fmt(t['paper'])}")


if __name__ == "__main__":
    main()
