"""Print the CompGS size report stored in a run's checkpoint.

Usage: tools/ssenv python report_size.py <run_dir> [<run_dir> ...]
paper_size = paper §3 formula (fp32 pos+opacity, log2 K-bit indices, RLE rotation index, fp32
codebooks); official_disk_size = released save format (ceil(log2 N)-bit indices).
"""

import json
import sys
from pathlib import Path

import torch


def report(run):
    ckpt = torch.load(Path(run) / "checkpoint.pt", map_location="cpu", weights_only=True)
    state = ckpt["density_state"]["compgs"]
    if state["encoded"] is None:
        raise SystemExit(f"{run}: no bake step ran (training length != total_steps?)")
    return {"run": str(run), **state["encoded"],
            "events": [e for e in state["events"] if e["compgs"] != "prune"]}


if __name__ == "__main__":
    for r in sys.argv[1:]:
        print(json.dumps(report(r), indent=1))
