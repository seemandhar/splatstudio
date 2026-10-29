"""Run a fixed native recipe; write every success/failure and held-out result."""

import argparse
import contextlib
import json
from pathlib import Path

from splatstudio.methods import BUILTINS
from splatstudio.registry import load
from splatstudio.runtime import evaluate, train, write_json
from splatstudio.verification import verify


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--num-gaussians", type=int, default=2000)
    parser.add_argument("--max-gaussians", type=int, default=100000)
    parser.add_argument("--seeds", default="0")
    parser.add_argument("--methods", nargs="+", default=list(BUILTINS))
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    records = []
    for name in args.methods:
        for seed in map(int, args.seeds.split(",")):
            destination = root / f"{name}-seed{seed}"
            record = {"method": name, "seed": seed, "paper_reproduction": "unverified"}
            try:
                method = load(name)
                checks = verify(method, args.device)
                write_json(root / f"{name}-verification.json", checks)
                if checks["status"] != "declared_checks_passed":
                    raise RuntimeError("Method verification did not pass")
                with (
                    (root / f"{name}-seed{seed}.log").open("w") as log,
                    contextlib.redirect_stdout(log),
                ):
                    train(
                        method,
                        args.data,
                        destination,
                        steps=args.steps,
                        seed=seed,
                        num_gaussians=args.num_gaussians,
                        max_gaussians=args.max_gaussians,
                        device=args.device,
                    )
                    result = evaluate(method, destination, device=args.device, save_images=True)
                record.update(status="passed", result=result)
            except Exception as exc:
                record.update(status="failed", reason=type(exc).__name__ + ": " + str(exc))
            records.append(record)
            write_json(root / "summary.json", {"recipe": vars(args), "runs": records})
            print(json.dumps(record), flush=True)
    return 1 if any(r["status"] != "passed" for r in records) else 0


if __name__ == "__main__":
    raise SystemExit(main())
