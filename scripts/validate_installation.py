"""Exercise real package installation and CUDA training; outputs are smoke evidence only."""

import argparse
import json
import subprocess
import sys
import uuid
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--steps", type=int, default=40)
    args = parser.parse_args()
    root = args.output.resolve()
    if root.exists() and any(root.iterdir()):
        parser.error("output must be empty")
    root.mkdir(parents=True, exist_ok=True)
    name = "installation-check-" + uuid.uuid4().hex[:8]

    def command(label, *arguments):
        result = subprocess.run(
            arguments, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
        )
        (root / f"{label}.log").write_text(result.stdout)
        if result.returncode:
            raise RuntimeError(
                f"{label} failed ({result.returncode}); see {root / (label + '.log')}"
            )
        print(f"{label}: passed", flush=True)

    cli = [sys.executable, "-m", "splatstudio.cli"]
    command("scaffold", *cli, "create", name, "--output", str(root / "plugin"))
    command(
        "install",
        sys.executable,
        "-m",
        "pip",
        "install",
        "--no-deps",
        "--no-build-isolation",
        "-e",
        str(root / "plugin"),
    )
    command("discovery", *cli, "inspect", name)
    command(
        "verification",
        *cli,
        "verify",
        name,
        "--device",
        args.device,
        "--output",
        str(root / "verification.json"),
    )
    command("dataset", *cli, "demo-data", "--output", str(root / "data"))
    command(
        "training",
        *cli,
        "train",
        name,
        "--data",
        str(root / "data"),
        "--output",
        str(root / "run"),
        "--steps",
        str(args.steps),
        "--num-gaussians",
        "192",
        "--device",
        args.device,
        "--log-every",
        "10",
    )
    if (root / "run/test.json").exists():
        raise AssertionError("Training must not evaluate test views")
    command(
        "evaluation",
        *cli,
        "eval",
        "--run",
        str(root / "run"),
        "--split",
        "test",
        "--device",
        args.device,
    )
    test = json.loads((root / "run/test.json").read_text())
    verification = json.loads((root / "verification.json").read_text())
    if test["split"] != "test" or verification["status"] != "declared_checks_passed":
        raise AssertionError("Unexpected evaluation or verification status")
    summary = {
        "status": "passed",
        "method": name,
        "steps": args.steps,
        "checks": ["scaffold", "install", "discovery", "verification", "training", "evaluation"],
        "test_metrics": test["metrics"],
        "paper_reproduction": "unverified",
        "purpose": "Software integration test on an analytic sphere fixture",
    }
    (root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
