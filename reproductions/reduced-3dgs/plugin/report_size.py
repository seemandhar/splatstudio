"""Print the encoded-size report a reduced-3dgs run stored in its checkpoint.

Usage: tools/ssenv python report_size.py <run_dir> [<run_dir> ...]
"""

import json
import sys
from pathlib import Path

import torch


def report(run):
    ckpt = torch.load(Path(run) / "checkpoint.pt", map_location="cpu", weights_only=True)
    state = ckpt["density_state"]["reduced3dgs"]
    enc = state["encoded"]
    if enc is None:
        raise SystemExit(f"{run}: no encode step ran (training length != total_steps?)")
    events = [e for e in state["events"] if e["reduced3dgs"] in ("mercy", "cull_sh")]
    return {"run": str(run), **enc,
            "mercy_rounds": sum(e["reduced3dgs"] == "mercy" for e in events),
            "cull_sh": next((e for e in events if e["reduced3dgs"] == "cull_sh"), None)}


if __name__ == "__main__":
    for r in sys.argv[1:]:
        print(json.dumps(report(r), indent=1))
