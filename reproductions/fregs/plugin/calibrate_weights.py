"""Calibrate FreGS loss weights w_l, w_h from the paper's Fig. 2 gradient ratios (plan 6.1, review-1).

Uses TRAINING views only (splits["train"]); never opens test/validation images or metric files.

Statistic (primary): the per-Gaussian view-space gradient norm that densification thresholds,
||dL/dmeans2d * (W/2, H/2)|| over visible Gaussians, pooled over the selected training views
(gsplat DefaultStrategy's grad2d). The pixel-gradient ratio mean|dL/drgb| is recorded too.
Fig. 2's "w/o FR" curve decays 0.25 -> 0.055 over training, which mean|dL_photo/drgb| (L1 sign
gradient) cannot do, so Fig. 2 is read as the screen-space statistic.

w_l: step-1 state of `ss-train --recipe 3dgs` (all SfM points, 3-NN log scales, SH degree 0),
     t = 1 (low band only); target ratio 1.7 (Fig. 2 iter 0: 0.43/0.25 and 0.405/0.248, both
     regions alike, so a whole-image target is exact here).
w_h: gsplat reference state at t = T = 15000 (train views only, by default trained here for
     15000 steps of the 30000-step schedule, i.e. the state a 30k run has at step 15000), t = T.
     Primary rule (review-2): per-band unit-gradient equalization, w_h = w_l * M_low / M_high,
     M_b = pooled mean ||d(d_ba + d_bp)/dmeans2d|| at unit weight. Eq. 14 treats both bands
     symmetrically and the paper gives no reason to favour either; a whole-image Fig. 2 ratio
     cannot separate w_h from w_l (no low-only curve). The resulting whole-image t = T ratio is
     checked against Fig. 2's 12k-15k envelope (~1.15-2.3); a miss is recorded as a deviation,
     never retuned. Whole-image solves at 1.15/1.5/2.0 are recorded as diagnostics only.
Per-term unit-weight magnitudes (amplitude vs phase, low vs high band) are recorded per scene.
Global values: w_l = median over scenes, w_h = w_l * median(M_low/M_high). Per-scene reports
(one process per scene) are merged with --from-reports. Prints a JSON report (also written with --output).

Example:
  tools/ssenv python reproductions/fregs/plugin/calibrate_weights.py --scenes bicycle garden \
      --device cuda:0 --reference-dir runs/fregs-calib-reference \
      --output reproductions/fregs/evidence/weight_calibration.json
"""

import argparse
import json
import random
import statistics
from pathlib import Path

import torch
from splatstudio.components import build
from splatstudio.data import initial_points, load_dataset, on_device
from splatstudio.methods import make_method
from splatstudio.runtime import camera_extent, knn_log_scales
from splatstudio.state import check_state, initialize

from ss_fregs import method as fregs_method
from ss_fregs.defaults import WEIGHT_TARGETS
from ss_fregs.frequency import parse_config
from ss_fregs.gradstats import (
    band_magnitudes,
    equalized_w_h,
    magnitudes,
    pooled_ratio,
    term_gradients,
)

SCENES = ("bicycle", "flowers", "garden", "stump", "treehill", "room", "counter", "kitchen", "bonsai")
DATA_ROOT = Path.home() / "datasets/mipnerf360/native"


def train_views(dataset, count):
    names = dataset["splits"]["train"]  # training views only
    if count >= len(names):
        return list(names)
    return [names[round(i * (len(names) - 1) / (count - 1))] for i in range(count)]


def step1_state(spec, dataset, seed, device):
    """Same initialization as runtime.train(recipe='3dgs', num_gaussians=10_000_000)."""
    torch.manual_seed(seed)
    random.seed(seed)
    points, colors = initial_points(dataset, 10_000_000, seed, device)
    params, optimizers = initialize(spec, points, colors)
    with torch.no_grad():
        params["scales"].copy_(knn_log_scales(points))
    return params, optimizers


def train_reference(dataset, steps, schedule, seed, device, max_gaussians=12_000_000):
    """gsplat baseline trained on TRAINING views for `steps` of a `schedule`-step run.

    Mirrors runtime.train(recipe='3dgs') (same init, Adam, means-LR decay 0.01**(step/schedule),
    SH degree schedule, DefaultStrategy), but stops early and evaluates nothing.
    """
    base = make_method("gsplat")
    renderer, density, losses = build(base)
    params, optimizers = step1_state(base, dataset, seed, device)
    scene_scale = max(camera_extent(dataset), 0.01)
    optimizers["means"].param_groups[0]["lr"] *= scene_scale
    density.initialize(params, optimizers, scene_scale)
    names = dataset["splits"]["train"]
    initial_lr = optimizers["means"].param_groups[0]["lr"]
    for step in range(1, steps + 1):
        frame = on_device(dataset["frames"][random.choice(names)], device)
        degree = min((step - 1) // 1000, 3)
        outputs = renderer.render(params, frame, sh_degree=degree)
        outputs["step"] = step
        density.before_backward(params, optimizers, step, outputs)
        sum(loss(params, outputs, frame["image"]) for _, loss in losses).backward()
        for opt in optimizers.values():
            opt.step()
            opt.zero_grad(set_to_none=True)
        optimizers["means"].param_groups[0]["lr"] = initial_lr * (0.01 ** (step / schedule))
        density.after_backward(params, optimizers, step, outputs)
        if len(params["means"]) > max_gaussians:
            raise RuntimeError("Gaussian budget exceeded in the reference run")
        if step % 1000 == 0:
            check_state(base, params, optimizers)
            print(json.dumps({"reference_step": step, "gaussians": len(params["means"])}), flush=True)
    return {
        "params": {k: v.detach() for k, v in params.items()},
        "step": steps,
        "schedule": schedule,
        "sh_degree": degree,
        "protocol_id": dataset["protocol_id"],
        "seed": seed,
        "source": "calibrate_weights.train_reference (gsplat, training views only)",
    }


def collect(renderer, params, dataset, names, degree, t, cfg, photometric, device):
    # Leaf tensors with grad so the render graph reaches means2d (checkpoints store plain tensors).
    params = {k: v.detach().requires_grad_(True) for k, v in params.items()}
    views = []
    for name in names:
        frame = on_device(dataset["frames"][name], device)
        out = renderer.render(params, frame, sh_degree=degree)
        views.append(term_gradients(out, frame["image"], t, cfg, photometric))
        del out
    return views


def solve(ratio, target, w_max=1e6):
    """Smallest bracketed w > 0 with ratio(w) = target; raises if unidentifiable.

    The ratio need not be monotone (a norm of a vector sum can dip), so a geometric grid
    brackets the first upward crossing of the target before bisecting.
    """
    r0 = ratio(0.0)
    if r0 >= target:
        raise ValueError(f"unidentifiable: ratio at w=0 is {r0:.4f} >= target {target}")
    lo, hi = 0.0, 1e-14
    while ratio(hi) <= target:
        lo, hi = hi, hi * 2
        if hi > w_max:
            raise ValueError(f"unidentifiable: no w <= {w_max:g} reaches target {target}")
    for _ in range(200):
        mid = (lo * hi) ** 0.5 if lo > 0 else hi / 2
        lo, hi = (mid, hi) if ratio(mid) <= target else (lo, mid)
        if hi - lo <= 1e-6 * hi:
            break
    return hi


def try_solve(ratio, target):
    """Sensitivity solves: record the failure loudly instead of a value."""
    try:
        return solve(ratio, target)
    except ValueError as exc:
        return str(exc)


def weights(w_l, w_h=0.0):
    return {"photo": 1.0, "low_a": w_l, "low_p": w_l, "high_a": w_h, "high_p": w_h}


def aggregate(report):
    """Global values: w_l = median scene w_l; w_h = w_l * median(M_low/M_high). Every scene must
    have the needed entries (no silent subset medians)."""
    scenes = report["scenes"]
    missing = [k for k, v in scenes.items() if "w_l" not in v]
    if not missing:
        report["w_l"] = statistics.median(v["w_l"] for v in scenes.values())
    ratios = {k: v["w_h_band_ratio"] for k, v in scenes.items() if "w_h_band_ratio" in v}
    if ratios:
        if len(ratios) != len(scenes):
            raise SystemExit(f"w_h band ratio missing for {sorted(set(scenes) - set(ratios))}")
        w_l = report.get("w_l") or report.get("w_l_used_for_w_h")
        report["w_h_band_ratio_median"] = statistics.median(ratios.values())
        report["w_h"] = w_l * report["w_h_band_ratio_median"]
        report["w_h_within_fig2_envelope_all_scenes"] = all(
            v["within_fig2_envelope"] for v in scenes.values())
    return report


def from_reports(paths, output):
    """Merge per-scene reports (later files update earlier ones) and aggregate."""
    report = {"procedure": "aggregated with --from-reports", "sources": [str(p) for p in paths],
              "scenes": {}}
    for path in paths:
        part = json.loads(Path(path).read_text())
        for scene, entry in part["scenes"].items():
            report["scenes"].setdefault(scene, {}).update(entry)
        for key in ("targets", "frequency_config", "w_l_used_for_w_h"):
            if key in part:
                report[key] = part[key]
    stale = [k for k, v in report["scenes"].items() if "w_h" in v and "w_h_band_ratio" not in v]
    if stale:
        raise SystemExit(
            f"Reports for {stale} predate review-2 (no w_h_band_ratio); rerun cheaply with cached "
            "references: --stage w_h --w-l <median w_l> --reference-dir runs/fregs-calib-reference")
    text = json.dumps(aggregate(report), indent=2, sort_keys=True)
    if output:
        Path(output).write_text(text + "\n")
    print(text)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scenes", nargs="+", default=list(SCENES))
    p.add_argument("--data-root", default=str(DATA_ROOT))
    p.add_argument("--stage", choices=["w_l", "w_h", "both"], default="both")
    p.add_argument("--w-l", type=float, help="fixed w_l for --stage w_h (default: median of w_l stage)")
    p.add_argument("--reference-dir", help="cache dir for trained 15k references (<scene>.pt)")
    p.add_argument("--reference-checkpoint", help="use this checkpoint template ({scene}) instead "
                   "of training a reference; its step is recorded (bias if != t_end)")
    p.add_argument("--reference-schedule", type=int, default=30000)
    p.add_argument("--views", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--output")
    p.add_argument("--from-reports", nargs="+", help="aggregate per-scene JSON reports only")
    args = p.parse_args()
    if args.from_reports:
        return from_reports(args.from_reports, args.output)
    if args.stage == "w_h" and args.w_l is None:
        p.error("--stage w_h requires --w-l")

    spec = fregs_method()
    cfg = parse_config(next(c for c in spec.components if c.name == "fregs-frequency").config)
    renderer, _, losses = build(spec)
    photometric = dict(losses)["photometric"]
    report = {
        "procedure": "plan.md 6.1 as revised by review-1; training views only",
        "statistic": "pooled mean ||dL/dmeans2d*(W/2,H/2)|| over visible Gaussians (primary); "
        "pooled mean |dL/drgb| (recorded)",
        "targets": dict(WEIGHT_TARGETS),
        "targets_source": "FreGS Fig. 2 (Bicycle): iter 0 ~1.7x (w_l); 12k-15k well ~1.15x, over "
        "~2.3x used only as a check envelope for the equalized w_h",
        "frequency_config": cfg,
        "views_per_scene": args.views,
        "scenes": {},
        "label": "calibrated from Fig. 2 gradient ratios on training views",
    }
    datasets = {}

    def dataset_for(scene):
        if scene not in datasets:
            datasets[scene] = load_dataset(Path(args.data_root) / scene)
        return datasets[scene]

    if args.stage in ("w_l", "both"):
        for scene in args.scenes:
            dataset = dataset_for(scene)
            names = train_views(dataset, args.views)
            params, _ = step1_state(spec, dataset, args.seed, args.device)
            views = collect(renderer, params, dataset, names, 0, 1, cfg, photometric, args.device)
            w_l = solve(lambda w: pooled_ratio(views, weights(w))[1], WEIGHT_TARGETS["w_l"])
            w_l_rgb = try_solve(lambda w: pooled_ratio(views, weights(w))[0], WEIGHT_TARGETS["w_l"])
            report["scenes"].setdefault(scene, {}).update(
                w_l=w_l,
                w_l_if_rgb_statistic=w_l_rgb,
                w_l_ratios_at_solution=pooled_ratio(views, weights(w_l)),
                w_l_unit_weight_terms=magnitudes(views),
                train_views=names,
                initial_gaussians=len(params["means"]),
            )
            print(json.dumps({"scene": scene, "w_l": w_l}), flush=True)
            del params, views
            torch.cuda.empty_cache()
        report["w_l"] = statistics.median(s["w_l"] for s in report["scenes"].values())
    w_l = args.w_l if args.w_l is not None else report.get("w_l")
    if args.stage in ("w_h", "both"):
        if w_l is None:
            raise SystemExit("No w_l available for the w_h stage")
        report["w_l_used_for_w_h"] = w_l
        for scene in args.scenes:
            dataset = dataset_for(scene)
            names = train_views(dataset, args.views)
            if args.reference_checkpoint:
                path = Path(args.reference_checkpoint.format(scene=scene))
                reference = torch.load(path, map_location=args.device, weights_only=True)
            else:
                path = Path(args.reference_dir or ".") / f"{scene}.pt"
                if path.exists():
                    reference = torch.load(path, map_location=args.device, weights_only=True)
                else:
                    reference = train_reference(dataset, cfg["t_end"], args.reference_schedule,
                                                args.seed, args.device)
                    if args.reference_dir:
                        path.parent.mkdir(parents=True, exist_ok=True)
                        torch.save(reference, path)
            if reference["protocol_id"] != dataset["protocol_id"]:
                raise SystemExit(f"{path}: reference was trained on a different dataset/protocol")
            views = collect(renderer, reference["params"], dataset, names,
                            int(reference["sh_degree"]), cfg["t_end"], cfg, photometric, args.device)
            bands = band_magnitudes(views)
            w_h = equalized_w_h(w_l, bands)  # primary rule (review-2): w_h M_high = w_l M_low
            rgb_ratio, m2d_ratio = pooled_ratio(views, weights(w_l, w_h))
            lo_env, hi_env = WEIGHT_TARGETS["fig2_envelope_tT"]
            entry = {
                "w_h_reference": str(path),
                "w_h_reference_step": int(reference["step"]),
                "w_h_reference_bias": None if int(reference["step"]) == cfg["t_end"] else
                "reference step differs from t_end; sharper renders bias w_h",
                "w_h_unit_weight_terms": magnitudes(views),
                "w_h_band_magnitudes": bands,
                "w_h_band_ratio": bands["low"] / bands["high"],
                "w_h": w_h,
                "w_h_rule": "equalized per-band unit screen gradient: w_h = w_l * M_low / M_high",
                "w_h_ratio_at_zero": pooled_ratio(views, weights(w_l))[1],
                "tT_means2d_ratio_at_w_h": m2d_ratio,
                "tT_rgb_ratio_at_w_h": rgb_ratio,
                "fig2_envelope_tT": [lo_env, hi_env],
                "within_fig2_envelope": lo_env <= m2d_ratio <= hi_env,
                "deviation": None if lo_env <= m2d_ratio <= hi_env else
                f"whole-image t=T means2d ratio {m2d_ratio:.3f} outside Fig. 2 12k-15k envelope "
                f"[{lo_env}, {hi_env}]; recorded, not retuned",
                "train_views": names,
                "sensitivity_label": "diagnostic only (not used): whole-image ratio targets",
            }
            for t in WEIGHT_TARGETS["w_h_sensitivity"]:
                entry[f"sensitivity_w_h_at_{t}"] = try_solve(
                    lambda w: pooled_ratio(views, weights(w_l, w))[1], t)
            report["scenes"].setdefault(scene, {}).update(entry)
            print(json.dumps({"scene": scene, "w_h": w_h, "tT_ratio": m2d_ratio}), flush=True)
            del reference, views
            torch.cuda.empty_cache()
        aggregate(report)
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
