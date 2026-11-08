"""Print the LightGaussian size report stored in a run's checkpoint; optionally export the zip.

Size = the official `extreme_saving.zip` bytes / 1024**2 (vectree.py). Usage:
tools/ssenv python report_size.py <run_dir> [<run_dir> ...] [--export]
(--export writes <run_dir>/extreme_saving.zip, readable by the official `load_vqgaussian`).
"""

import json
import sys
from pathlib import Path

import torch


def report(run, export=False):
    ckpt = torch.load(Path(run) / "checkpoint.pt", map_location="cpu", weights_only=True)
    state = ckpt["density_state"]["lightgaussian"]
    enc = state["encoded"]
    if enc is None:
        raise SystemExit(f"{run}: no encode step ran (training length != total_steps?)")
    out = {"run": str(run), **enc,
           "events": [e for e in state["events"] if e["lightgaussian"] != "encode"]}
    if "zip" in state:
        data = state["zip"].numpy().tobytes()
        if len(data) != enc["zip_bytes"]:
            raise SystemExit(f"{run}: stored zip has {len(data)} bytes, report {enc['zip_bytes']}")
        if export:
            path = Path(run) / "extreme_saving.zip"
            path.write_bytes(data)
            out["exported"] = str(path)
    return out


if __name__ == "__main__":
    export = "--export" in sys.argv[1:]
    for r in (a for a in sys.argv[1:] if a != "--export"):
        print(json.dumps(report(r, export), indent=1))
