#!/usr/bin/env python3
"""GPU-aware benchmark runner for SplatStudio reproductions (used by the Claude Code agents).

Subcommands
  prepare   --dataset mipnerf360 [--scenes garden bicycle]   convert COLMAP scenes to native form
  launch    --exp NAME --methods gsplat my-method --dataset mipnerf360 --scenes garden ...
            [--steps 30000] [--recipe 3dgs] [--gpus 4,6] [--min-free-gb 16] [--extra "--seed 1"]
            queue train+test-eval jobs and start a detached scheduler for the experiment
  status    --exp NAME                       job table (JSON)
  wait      --exp NAME [--timeout 540]       block until all jobs finish (exit 0) or timeout (exit 3)
  summarize --exp NAME [--paper paper_results.json] [--baseline gsplat] [--output DIR]
            per-scene table, paper comparison, baseline-adjusted gaps; writes summary.json/.md
  cancel    --exp NAME                       kill running jobs of the experiment

Runs live in $SPLATIFY_RUNS (default /home/sejain/splatify/runs)/<exp>/<method>/<dataset>-<scene>.
Jobs run inside the splatstudio env through tools/ssenv. The scheduler is a detached child of
the launching session: if that Claude Code session exits, jobs die; relaunch the same
`launch` command to resume (killed jobs are re-queued, partial runs moved to *.stale-<ts>).
"""

import argparse
import json
import os
import shlex
import signal
import subprocess
import sys
import time
from pathlib import Path

TOOLS = Path(__file__).resolve().parent
ROOT = TOOLS.parent
RUNS = Path(os.environ.get("SPLATIFY_RUNS", ROOT / "runs"))
CLAIMS = RUNS / ".gpu_claims"
SSENV = TOOLS / "ssenv"
CONVERTER = ROOT / "splatstudio" / "scripts" / "colmap_to_transforms.py"
DATASETS = json.loads((TOOLS / "datasets.json").read_text())
REFERENCE = json.loads((TOOLS / "reference_results.json").read_text())
METRICS = ("psnr", "ssim", "lpips")


def scenes_for(dataset, scenes):
    known = DATASETS[dataset]["scenes"]
    chosen = scenes or list(known)
    unknown = sorted(set(chosen) - set(known))
    if unknown:
        raise SystemExit(f"Unknown {dataset} scenes: {unknown}; known: {sorted(known)}")
    return chosen


def native_path(dataset, scene):
    return Path(DATASETS[dataset]["native_root"]) / scene


def cmd_prepare(args):
    out = {}
    for scene in scenes_for(args.dataset, args.scenes):
        target = native_path(args.dataset, scene)
        if (target / "transforms.json").exists():
            out[scene] = "exists"
            continue
        raw = Path(DATASETS[args.dataset]["raw_root"]) / scene
        if not raw.exists():
            out[scene] = f"missing raw scene {raw}"
            continue
        proc = subprocess.run(
            [str(SSENV), "python", "-I", str(CONVERTER), "--scene", str(raw), "--images",
             DATASETS[args.dataset]["scenes"][scene], "--output", str(target)],
            capture_output=True, text=True,
        )
        out[scene] = json.loads(proc.stdout) if proc.returncode == 0 else proc.stderr[-2000:]
    print(json.dumps(out, indent=2))


# ── experiment state ─────────────────────────────────────────────────────────

def exp_dir(name):
    return RUNS / name


def load_jobs(name):
    path = exp_dir(name) / "jobs.json"
    if not path.exists():
        raise SystemExit(f"No experiment {name!r} under {RUNS}")
    return json.loads(path.read_text())


def save_jobs(name, state):
    path = exp_dir(name) / "jobs.json"
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2))
    tmp.replace(path)


def alive(pid):
    if pid is None:
        return False
    try:
        # Reap our own finished children first; a zombie still answers kill(pid, 0).
        if os.waitpid(pid, os.WNOHANG)[0] == pid:
            return False
    except ChildProcessError:
        pass  # not our child (another scheduler's job, or already reaped)
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    try:  # an unreaped zombie of another process is not alive either
        return Path(f"/proc/{pid}/stat").read_text().split(")")[-1].split()[0] != "Z"
    except OSError:
        return False


def cmd_launch(args):
    root = exp_dir(args.exp)
    if (root / "jobs.json").exists():
        state = load_jobs(args.exp)
        if alive(state.get("scheduler_pid")):
            raise SystemExit("Scheduler for this experiment is still running; use a new --exp")
        # Resume: jobs killed with a previous scheduler (or failed) go back to the queue.
        # Partial run directories are moved aside, never deleted.
        for job in state["jobs"]:
            if job["status"] == "running" and (Path(job["run"]) / "test.json").exists():
                job["status"] = "done"  # finished while no scheduler was watching
                continue
            killed = job["status"] == "running" and not alive(job.get("pid"))
            if killed or (job["status"] == "failed" and args.retry_failed):
                run = Path(job["run"])
                if run.exists():
                    run.rename(run.with_name(f"{run.name}.stale-{int(time.time())}"))
                job.update(status="pending", gpu=None, pid=None)
    else:
        state = {"jobs": []}
    existing = {j["id"] for j in state["jobs"]}
    for method in args.methods:
        for scene in scenes_for(args.dataset, args.scenes):
            data = native_path(args.dataset, scene)
            if not (data / "transforms.json").exists():
                raise SystemExit(f"{data} not prepared; run: bench.py prepare --dataset {args.dataset}")
            job_id = f"{method}/{args.dataset}-{scene}"
            if job_id in existing:
                continue
            state["jobs"].append({
                "id": job_id, "method": method, "dataset": args.dataset, "scene": scene,
                "data": str(data), "run": str(root / method / f"{args.dataset}-{scene}"),
                "steps": args.steps, "recipe": args.recipe, "extra": args.extra,
                "status": "pending",
            })
    state.update(min_free_gb=args.min_free_gb, gpus=args.gpus, per_gpu=args.per_gpu)
    root.mkdir(parents=True, exist_ok=True)
    state["scheduler_pid"] = None
    save_jobs(args.exp, state)
    log = open(root / "scheduler.log", "a")
    proc = subprocess.Popen(
        [sys.executable, __file__, "_schedule", "--exp", args.exp],
        stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
    )
    print(json.dumps({"exp": args.exp, "jobs": len(state["jobs"]), "scheduler_pid": proc.pid,
                      "dir": str(root)}, indent=2))


def gpu_table():
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.free,utilization.gpu", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True,
    ).stdout
    rows = []
    for line in out.strip().splitlines():
        idx, free, util = (int(v) for v in line.split(","))
        rows.append({"index": idx, "free_gb": free / 1024, "util": util})
    return rows


def claims():
    CLAIMS.mkdir(parents=True, exist_ok=True)
    held = {}
    for path in CLAIMS.iterdir():
        gpu, _, pid = path.name.partition("-")
        if alive(int(pid)):
            held[int(gpu)] = held.get(int(gpu), 0) + 1
        else:
            path.unlink(missing_ok=True)
    return held


def oom_failure(job):
    log = Path(f"{job['run']}.log")
    tail = log.read_text(errors="replace")[-4000:] if log.exists() else ""
    return "OutOfMemoryError" in tail or "CUDA out of memory" in tail


def job_command(job):
    train = [
        str(SSENV), "ss-train", job["method"], "--data", job["data"], "--output", job["run"],
        "--steps", str(job["steps"]), "--recipe", job["recipe"], "--device", "cuda:0",
        "--num-gaussians", "10000000", "--max-gaussians", "12000000", "--log-every", "500",
    ] + (job["extra"].split() if job["extra"] else [])
    evaluate = [str(SSENV), "ss-eval", "--run", job["run"], "--split", "test", "--device", "cuda:0"]
    return f"{shlex.join(train)} && {shlex.join(evaluate)} > {shlex.quote(job['run'] + '.eval.json')}"


def cmd_schedule(args):
    while True:
        state = load_jobs(args.exp)
        state["scheduler_pid"] = os.getpid()
        jobs = state["jobs"]
        for job in jobs:
            if job["status"] == "running" and not alive(job.get("pid")):
                result = Path(job["run"]) / "test.json"
                job["status"] = "done" if result.exists() else "failed"
                job["finished"] = time.time()
                Path(CLAIMS / f"{job['gpu']}-{job['pid']}").unlink(missing_ok=True)
                if job["status"] == "failed" and oom_failure(job) and job.get("retries", 0) < 2:
                    # Shared GPUs: another process grew after placement. Re-queue, keep the log.
                    run, log = Path(job["run"]), Path(f"{job['run']}.log")
                    stamp = int(time.time())
                    if run.exists():
                        run.rename(run.with_name(f"{run.name}.oom-{stamp}"))
                    if log.exists():
                        log.rename(log.with_name(f"{log.name}.oom-{stamp}"))
                    job.update(status="pending", gpu=None, pid=None, retries=job.get("retries", 0) + 1)
                    print(f"{time.ctime()} re-queued {job['id']} after CUDA OOM", flush=True)
        pending = [j for j in jobs if j["status"] == "pending"]
        if pending:
            held = claims()
            allowed = {int(g) for g in state["gpus"].split(",")} if state.get("gpus") else None
            candidates = sorted(
                (g for g in gpu_table()
                 if (allowed is None or g["index"] in allowed)
                 and g["free_gb"] >= state["min_free_gb"]
                 and held.get(g["index"], 0) < state["per_gpu"]),
                key=lambda g: (held.get(g["index"], 0), g["util"], -g["free_gb"]),
            )
            for gpu, job in zip(candidates, pending):
                Path(job["run"]).parent.mkdir(parents=True, exist_ok=True)
                log = open(f"{job['run']}.log", "w")
                proc = subprocess.Popen(
                    ["bash", "-c", job_command(job)], stdout=log, stderr=subprocess.STDOUT,
                    env={**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu["index"])},
                    start_new_session=True,
                )
                (CLAIMS / f"{gpu['index']}-{proc.pid}").touch()
                job.update(status="running", gpu=gpu["index"], pid=proc.pid, started=time.time())
                print(f"{time.ctime()} launched {job['id']} on GPU {gpu['index']}", flush=True)
                save_jobs(args.exp, state)
                time.sleep(20)  # let the new job allocate memory before the next placement
        save_jobs(args.exp, state)
        if all(j["status"] in ("done", "failed", "cancelled") for j in jobs):
            print(f"{time.ctime()} all jobs finished", flush=True)
            return
        time.sleep(30)


def progress(job):
    log = Path(f"{job['run']}.log")
    if not log.exists():
        return None
    for line in reversed(log.read_text(errors="replace").splitlines()[-50:]):
        if line.startswith('{"step"'):
            row = json.loads(line)
            return {"step": row["step"], "gaussians": row["gaussians"], "loss": round(row["loss"], 5)}
        if "Error" in line or "error:" in line:
            return {"error": line[-300:]}
    return None


def cmd_status(args):
    state = load_jobs(args.exp)
    rows = []
    for job in state["jobs"]:
        row = {k: job.get(k) for k in ("id", "status", "gpu")}
        if job["status"] in ("running", "failed"):
            row["progress"] = progress(job)
        if job["status"] == "failed":
            row["log"] = f"{job['run']}.log"
        rows.append(row)
    counts = {}
    for job in state["jobs"]:
        counts[job["status"]] = counts.get(job["status"], 0) + 1
    print(json.dumps({"scheduler_alive": alive(state.get("scheduler_pid")), "counts": counts,
                      "jobs": rows}, indent=2))


def cmd_wait(args):
    deadline = time.time() + args.timeout
    while time.time() < deadline:
        state = load_jobs(args.exp)
        if all(j["status"] in ("done", "failed", "cancelled") for j in state["jobs"]):
            cmd_status(args)
            return 0
        if state.get("scheduler_pid") is not None and not alive(state["scheduler_pid"]):
            print("scheduler is not running; relaunch with the same --exp to resume", file=sys.stderr)
            cmd_status(args)
            return 4
        time.sleep(15)
    cmd_status(args)
    return 3


def cmd_cancel(args):
    state = load_jobs(args.exp)
    if alive(state.get("scheduler_pid")):
        os.kill(state["scheduler_pid"], signal.SIGTERM)
    for job in state["jobs"]:
        if job["status"] == "running" and alive(job.get("pid")):
            os.killpg(job["pid"], signal.SIGTERM)
        if job["status"] in ("running", "pending"):
            job["status"] = "cancelled"
    save_jobs(args.exp, state)
    print("cancelled")


# ── summaries ────────────────────────────────────────────────────────────────

def mean(values):
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def paper_value(block, metric, scene=None):
    if not block:
        return None
    if scene is None:
        return (block.get("average") or {}).get(metric)
    return ((block.get("per_scene") or {}).get(metric) or {}).get(scene)


def cmd_summarize(args):
    state = load_jobs(args.exp)
    blinded = [j["id"] for j in state["jobs"] if (Path(j["run"]) / "BLINDED.json").exists()]
    if ((exp_dir(args.exp) / "BLINDED.json").exists() or blinded) and not args.unblind:
        raise SystemExit(f"{args.exp} is a blinded calibration experiment ({len(blinded)} runs); "
                         "its test metrics must not inform method decisions. Pass --unblind only "
                         "for the final report, and say so in it.")
    paper = json.loads(Path(args.paper).read_text()) if args.paper else {}
    ours = {}
    for job in state["jobs"]:
        result = Path(job["run"]) / "test.json"
        if result.exists():
            record = json.loads(result.read_text())
            ours.setdefault(job["dataset"], {}).setdefault(job["method"], {})[job["scene"]] = {
                **record["metrics"], "gaussians": record["resources"]["gaussians"],
                "minutes": record["resources"]["training_seconds"] / 60,
            }
    report = {"exp": args.exp, "datasets": {}}
    lines = [f"# Benchmark summary: {args.exp}", ""]
    for dataset, methods in ours.items():
        pblock = (paper.get("datasets") or {}).get(dataset, {})
        p_method, p_base = pblock.get("method"), pblock.get("baseline_3dgs")
        ref = REFERENCE["3dgs"].get(dataset, {})
        ref_base = {"per_scene": {"psnr": ref.get("psnr", {})}, "average": ref.get("average", {})}
        p_base = p_base or ref_base
        scenes = sorted({s for m in methods.values() for s in m})
        complete = set(scenes) == set(DATASETS[dataset]["scenes"])
        entry = {"scenes": scenes, "all_scenes": complete, "methods": {}}
        lines += [f"## {dataset} ({len(scenes)} scenes{'' if complete else ', SUBSET'})", ""]
        header = "| method | " + " | ".join(scenes) + " | mean PSNR | mean SSIM | mean LPIPS |"
        lines += [header, "|" + "---|" * (len(scenes) + 4)]
        for method, by_scene in methods.items():
            avg = {m: mean([by_scene.get(s, {}).get(m) for s in scenes]) for m in METRICS}
            entry["methods"][method] = {"per_scene": by_scene, "mean": avg}
            cells = [f"{by_scene[s]['psnr']:.2f}" if s in by_scene else "—" for s in scenes]
            fmt = lambda v, d: f"{v:.{d}f}" if v is not None else "—"  # noqa: E731
            lines.append(f"| {method} | " + " | ".join(cells)
                         + f" | {fmt(avg['psnr'], 3)} | {fmt(avg['ssim'], 4)} | {fmt(avg['lpips'], 4)} |")
        for label, block in (("paper: method", p_method), ("paper: 3DGS baseline", p_base)):
            if block:
                cells = [f"{paper_value(block, 'psnr', s):.2f}" if paper_value(block, "psnr", s)
                         else "—" for s in scenes]
                avg = [paper_value(block, m) for m in METRICS]
                lines.append(f"| *{label}* | " + " | ".join(cells) + " | "
                             + " | ".join(f"{v}" if v is not None else "—" for v in avg) + " |")
        lines.append("")
        # Gap analysis: absolute gap and baseline-adjusted gap, scene-matched when possible.
        base = methods.get(args.baseline)
        gaps = {}
        for method, by_scene in methods.items():
            if method == args.baseline or not p_method:
                continue
            per_scene_abs, per_scene_adj = {}, {}
            for s in scenes:
                pm, pb = paper_value(p_method, "psnr", s), paper_value(p_base, "psnr", s)
                om = by_scene.get(s, {}).get("psnr")
                ob = (base or {}).get(s, {}).get("psnr")
                if pm is not None and om is not None:
                    per_scene_abs[s] = om - pm
                if None not in (pm, pb, om, ob):
                    per_scene_adj[s] = (om - ob) - (pm - pb)
            g = {"per_scene_abs": per_scene_abs, "per_scene_delta_vs_baseline": per_scene_adj}
            if complete:
                pm, pb = paper_value(p_method, "psnr"), paper_value(p_base, "psnr")
                om = entry["methods"][method]["mean"]["psnr"]
                ob = entry["methods"].get(args.baseline, {}).get("mean", {}).get("psnr")
                if pm is not None:
                    g["mean_abs"] = om - pm
                if None not in (pm, pb, ob):
                    g["mean_delta_vs_baseline"] = (om - ob) - (pm - pb)
                    g["ours_improvement"] = om - ob
                    g["paper_improvement"] = pm - pb
            if base and p_base:
                g["baseline_offset"] = {s: base[s]["psnr"] - paper_value(p_base, "psnr", s)
                                        for s in scenes if s in base and paper_value(p_base, "psnr", s)}
            gaps[method] = g
            lines.append(f"**{method} gaps (dB PSNR, ours − paper)**: "
                         + json.dumps({k: (round(v, 3) if isinstance(v, float) else
                                           {s: round(x, 3) for s, x in v.items()})
                                       for k, v in g.items()}))
            lines.append("")
        entry["gaps"] = gaps
        report["datasets"][dataset] = entry
    out = Path(args.output) if args.output else exp_dir(args.exp)
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(report, indent=2))
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("prepare")
    s.add_argument("--dataset", default="mipnerf360", choices=[k for k in DATASETS if not k.startswith("_")])
    s.add_argument("--scenes", nargs="*")
    s = sub.add_parser("launch")
    s.add_argument("--exp", required=True)
    s.add_argument("--methods", nargs="+", required=True)
    s.add_argument("--dataset", default="mipnerf360")
    s.add_argument("--scenes", nargs="*")
    s.add_argument("--steps", type=int, default=30000)
    s.add_argument("--recipe", default="3dgs", choices=["3dgs", "splatstudio"])
    s.add_argument("--gpus", default=None, help="comma-separated GPU indices to allow")
    s.add_argument("--min-free-gb", type=float, default=22)
    s.add_argument("--per-gpu", type=int, default=1)
    s.add_argument("--extra", default="", help="extra ss-train arguments")
    s.add_argument("--retry-failed", action="store_true", help="re-queue failed jobs on resume")
    s = sub.add_parser("_schedule")
    s.add_argument("--exp", required=True)
    for name in ("status", "cancel"):
        sub.add_parser(name).add_argument("--exp", required=True)
    s = sub.add_parser("wait")
    s.add_argument("--exp", required=True)
    s.add_argument("--timeout", type=int, default=540)
    s = sub.add_parser("summarize")
    s.add_argument("--exp", required=True)
    s.add_argument("--paper")
    s.add_argument("--baseline", default="gsplat")
    s.add_argument("--unblind", action="store_true", help="allow summarizing blinded runs")
    s.add_argument("--output")
    args = p.parse_args()
    fn = {"prepare": cmd_prepare, "launch": cmd_launch, "_schedule": cmd_schedule,
          "status": cmd_status, "wait": cmd_wait, "summarize": cmd_summarize,
          "cancel": cmd_cancel}[args.cmd]
    code = fn(args)
    raise SystemExit(code or 0)


if __name__ == "__main__":
    main()
