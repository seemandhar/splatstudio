"""Write and print the Compact-3DGS storage a run stored in its checkpoint.

The runtime's validation/test JSON only knows checkpoint_bytes (fp32 training state); the paper
sizes (official final_prune formulas, "Ours" and "Ours+PP") and the actual encoded container bytes
live in checkpoint["density_state"]["compact3dgs"]["encoded"]. This writes them to
<run>/compact_size.json.

Usage: tools/ssenv python report_size.py <run> [<run> ...]
"""

import json
import sys
from pathlib import Path

import torch


def report(run):
    run = Path(run)
    ckpt = torch.load(run / "checkpoint.pt", map_location="cpu", weights_only=True)
    enc = ckpt["density_state"]["compact3dgs"]["encoded"]
    (run / "compact_size.json").write_text(json.dumps(enc, indent=1, sort_keys=True) + "\n")
    return enc


if __name__ == "__main__":
    for r in sys.argv[1:]:
        e = report(r)
        sizes = e["paper_size_mb"]
        print(json.dumps({"run": r, "gaussians": e["gaussians"],
                          "paper_mb_pp": sizes.get("ours_pp", {}).get("total"),
                          "paper_mb_no_pp": sizes["ours"]["total"], "actual_mb": e["actual_mb"]}))
