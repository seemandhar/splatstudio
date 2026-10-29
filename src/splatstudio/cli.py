"""Command line entry points. Read-only discovery does not import GPU libraries."""

import argparse
import json
import sys
from pathlib import Path

from .registry import discover, load


def parser():
    p = argparse.ArgumentParser(prog="splatstudio")
    sub = p.add_subparsers(dest="command", required=True)
    methods = sub.add_parser("methods", help="List installed methods or the research collection")
    methods.add_argument("--catalog", action="store_true", help="Include the Nerfstudio collection")
    inspect = sub.add_parser("inspect")
    inspect.add_argument("method")
    create = sub.add_parser("create")
    create.add_argument("name")
    create.add_argument("--output")
    verify = sub.add_parser("verify")
    verify.add_argument("method")
    verify.add_argument("--device", default="cpu")
    verify.add_argument("--static-only", action="store_true")
    verify.add_argument("--output")
    train = sub.add_parser("train")
    train.add_argument("method")
    train.add_argument("--data", required=True)
    train.add_argument("--output", required=True)
    train.add_argument("--steps", type=int, default=30000)
    train.add_argument("--seed", type=int, default=0)
    train.add_argument("--downscale", type=int, default=1)
    train.add_argument("--num-gaussians", type=int, default=10000)
    train.add_argument("--max-gaussians", type=int, default=1000000)
    train.add_argument("--device", default="cuda:0")
    train.add_argument("--log-every", type=int, default=100)
    evaluate = sub.add_parser("eval")
    evaluate.add_argument("--run", required=True)
    evaluate.add_argument("--split", choices=["validation", "test"], default="test")
    evaluate.add_argument("--device", default="cuda:0")
    evaluate.add_argument("--save-images", action="store_true")
    compare = sub.add_parser("compare")
    compare.add_argument("reference")
    compare.add_argument("candidate")
    compare.add_argument("--psnr-tolerance", type=float, default=0.5)
    compare.add_argument("--output")
    demo = sub.add_parser("demo-data")
    demo.add_argument("--output", required=True)
    demo.add_argument("--size", type=int, default=32)
    prepare = sub.add_parser(
        "prepare-data", help="Undistort a Nerfstudio dataset for native training"
    )
    prepare.add_argument("--data", required=True)
    prepare.add_argument("--output", required=True)
    prepare.add_argument("--downscale", type=int, default=1)
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        code = 0
        if args.command == "methods":
            result = {"methods": discover()}
            if args.catalog:
                from importlib.resources import files

                result["research_collection"] = json.loads(
                    files("splatstudio").joinpath("catalog.json").read_text()
                )["methods"]
        elif args.command == "inspect":
            from dataclasses import asdict

            from .contracts import validate

            method = load(args.method)
            result = {"method": method.to_dict(), "issues": [asdict(i) for i in validate(method)]}
        elif args.command == "create":
            from .scaffold import create

            result = create(args.name, args.output)
        elif args.command == "verify":
            from .verification import verify

            result = verify(load(args.method), args.device, args.static_only)
            code = (
                1 if result["status"] == "failed" else 2 if result["status"] == "unverified" else 0
            )
        elif args.command == "train":
            from .runtime import train

            kwargs = vars(args).copy()
            kwargs.pop("command")
            kwargs["method"] = load(args.method)
            result = train(**kwargs)
        elif args.command == "eval":
            from .runtime import evaluate

            config = json.loads((Path(args.run) / "config.json").read_text())
            result = evaluate(
                load(config["method"]),
                args.run,
                split=args.split,
                device=args.device,
                save_images=args.save_images,
            )
        elif args.command == "compare":
            from .reporting import compare

            def records(path):
                value = json.loads(Path(path).read_text())
                return value if isinstance(value, list) else [value]

            result = compare(records(args.reference), records(args.candidate), args.psnr_tolerance)
        elif args.command == "demo-data":
            from .demo import create_dataset

            result = create_dataset(args.output, args.size)
        elif args.command == "prepare-data":
            from .preprocessing import prepare_dataset

            result = prepare_dataset(args.data, args.output, args.downscale)
        encoded = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
        if args.command in {"verify", "compare"} and args.output:
            path = Path(args.output)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(encoded)
        print(encoded, end="")
        return code
    except (ValueError, RuntimeError, ImportError, OSError, KeyError) as exc:
        print(f"splatstudio: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


def methods_main():
    return main(["methods", *sys.argv[1:]])


def create_main():
    return main(["create", *sys.argv[1:]])


def inspect_main():
    return main(["inspect", *sys.argv[1:]])


def verify_main():
    return main(["verify", *sys.argv[1:]])


def train_main():
    return main(["train", *sys.argv[1:]])


def eval_main():
    return main(["eval", *sys.argv[1:]])


def compare_main():
    return main(["compare", *sys.argv[1:]])


if __name__ == "__main__":
    raise SystemExit(main())
