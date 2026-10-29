"""Import or train collection entries in isolated processes using stock Nerfstudio.

Run inside the Nerfstudio environment. A short fixture run establishes execution,
not a paper's contribution fidelity or convergence. No LLM or API service is used.
"""

import argparse
import copy
import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def worker(args):
    import random

    import numpy as np
    import torch
    from nerfstudio.engine.callbacks import TrainingCallbackLocation

    entry = next(
        e
        for e in importlib.metadata.distribution("splatstudio-nerfstudio-methods").entry_points
        if e.name == args.method
    )
    config = copy.deepcopy(entry.load().config)
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.set_num_threads(2)
    config.output_dir = Path(args.output) / args.method
    config.experiment_name = "analytic-spheres"
    config.timestamp = "smoke"
    config.vis = "tensorboard"
    config.max_num_iterations = args.steps
    config.pipeline.datamanager.data = Path(args.data)
    config.pipeline.datamanager.dataparser.downscale_factor = 1
    config.pipeline.model.num_downscales = 0
    config.steps_per_eval_image = 0
    config.steps_per_eval_batch = 0
    config.steps_per_eval_all_images = 0
    trainer = config.setup(local_rank=0, world_size=1)
    trainer.setup()
    initial = sum(p.numel() for p in trainer.pipeline.model.parameters())
    trace = []
    for step in range(args.steps):
        trainer.step = step
        trainer.pipeline.train()
        for callback in trainer.callbacks:
            callback.run_callback_at_location(step, TrainingCallbackLocation.BEFORE_TRAIN_ITERATION)
        loss, losses, _ = trainer.train_iteration(step)
        if not torch.isfinite(loss):
            raise RuntimeError("Nonfinite loss")
        for callback in trainer.callbacks:
            callback.run_callback_at_location(step, TrainingCallbackLocation.AFTER_TRAIN_ITERATION)
        for name, parameter in trainer.pipeline.model.named_parameters():
            if not torch.isfinite(parameter).all():
                raise RuntimeError("Nonfinite parameter: " + name)
        trace.append(
            {
                "step": step,
                "loss": float(loss.detach()),
                "terms": {k: float(v.detach()) for k, v in losses.items()},
            }
        )
    trainer.save_checkpoint(args.steps - 1)
    return {
        "status": "passed",
        "steps": args.steps,
        "parameter_elements_at_start": initial,
        "training_trace": trace,
        "overrides": {
            "num_downscales": 0,
            "dataparser.downscale_factor": 1,
            "seed": 0,
            "viewer": "disabled",
            "data": "analytic-spheres",
        },
        "scope": "Short training with callbacks and finite-state checks; late schedules untested",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data")
    parser.add_argument("--output", required=True)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--gpus", default="0")
    parser.add_argument("--method")
    parser.add_argument("--imports-only", action="store_true")
    args = parser.parse_args()
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    if args.method:
        try:
            result = worker(args)
        except Exception as exc:
            traceback.print_exc()
            result = {"status": "failed", "reason": type(exc).__name__ + ": " + str(exc)}
        (root / (args.method + ".json")).write_text(json.dumps(result, indent=2) + "\n")
        return 0 if result["status"] == "passed" else 1
    entries = importlib.metadata.distribution("splatstudio-nerfstudio-methods").entry_points
    records = []
    for entry in entries:
        record = {
            "name": entry.name,
            "entry_point": entry.value,
            "paper_reproduction": "unverified",
        }
        try:
            spec = entry.load()
            record["configuration_import"] = "passed"
            record["model_config"] = type(spec.config.pipeline.model).__name__
        except Exception as exc:
            record["configuration_import"] = "failed"
            record["reason"] = str(exc)
        records.append(record)
    external = {
        "ss-depth-reg-gs": "Requires monocular depth model weights",
        "ss-stablegs": "Requires DUSt3R depth priors and model weights",
        "ss-fsgs": "Requires monocular depth model weights",
        "ss-feature-gs": "Requires feature targets; RGB-only training would not exercise its method",
    }
    gpus = args.gpus.split(",")

    def run_group(group):
        gpu, group_records = group
        for record in group_records:
            name = record["name"]
            if args.imports_only or name in external or record["configuration_import"] != "passed":
                record["training_check"] = {
                    "status": "not_run",
                    "reason": external.get(name, "Import audit only"),
                }
                continue
            command = [
                sys.executable,
                __file__,
                "--method",
                name,
                "--data",
                args.data,
                "--output",
                str(root),
                "--steps",
                str(args.steps),
            ]
            with (root / (name + ".log")).open("w") as log:
                try:
                    subprocess.run(
                        command,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        env={
                            **os.environ,
                            "CUDA_VISIBLE_DEVICES": gpu,
                            "HF_HUB_OFFLINE": "1",
                            "OMP_NUM_THREADS": "2",
                        },
                        timeout=300,
                    )
                    report = root / (name + ".json")
                    record["training_check"] = (
                        json.loads(report.read_text())
                        if report.exists()
                        else {"status": "failed", "reason": "Worker exited without a result"}
                    )
                except subprocess.TimeoutExpired:
                    record["training_check"] = {
                        "status": "failed",
                        "reason": "Exceeded 300 second smoke limit",
                    }
            print(name, record["training_check"]["status"], flush=True)

    with ThreadPoolExecutor(max_workers=len(gpus)) as pool:
        list(pool.map(run_group, [(gpu, records[i :: len(gpus)]) for i, gpu in enumerate(gpus)]))
    import nerfstudio
    import splatstudio_zoo

    sources = Path(splatstudio_zoo.__file__).parent
    result = {
        "schema_version": 1,
        "methods": records,
        "environment": {
            name: importlib.metadata.version(name)
            for name in ("nerfstudio", "gsplat", "torch", "numpy", "splatstudio-nerfstudio-methods")
        },
        "nerfstudio_source_sha256": hashlib.sha256(
            Path(nerfstudio.__file__).read_bytes()
        ).hexdigest(),
        "source_hashes": {
            str(p.relative_to(sources)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(sources.rglob("*.py"))
        },
        "paper_reproduction": "unverified",
    }
    (root / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    return (
        1
        if any(
            r["configuration_import"] == "failed" or r["training_check"]["status"] == "failed"
            for r in records
        )
        else 0
    )


if __name__ == "__main__":
    raise SystemExit(main())
