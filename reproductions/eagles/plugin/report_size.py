"""Print the encoded-size report an eagles run stored in its checkpoint.

Usage: tools/ssenv python report_size.py <run_dir> [<run_dir> ...]
paper_size_MB is the official GaussianModelSQ.size() (bits / 8 / 1e6), the number the paper
reports as "Storage Mem"; actual_MB counts the range-coded bitstream + tables + fp32 parts.
"""

import json
import sys
from pathlib import Path

import torch


def report(run):
    ckpt = torch.load(Path(run) / "checkpoint.pt", map_location="cpu", weights_only=True)
    state = ckpt["density_state"]["eagles"]
    enc = state["encoded"]
    if enc is None:
        raise SystemExit(f"{run}: no encode step ran (training length != total_steps?)")
    prunes = [e for e in state["events"] if e["eagles"] == "influence_prune"]
    return {"run": str(run), **enc, "influence_prunes": prunes}


if __name__ == "__main__":
    for r in sys.argv[1:]:
        print(json.dumps(report(r), indent=1))
