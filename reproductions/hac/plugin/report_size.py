"""Write and print the HAC bitstream size a run stored in its checkpoint.

The runtime's validation/test JSON only knows checkpoint_bytes (fp32 training state); the HAC
size (official bit formula, real arithmetic-coded streams) lives in
checkpoint["density_state"]["hac"]["encoded"]. This writes it to <run>/hac_size.json.

Usage: tools/ssenv python report_size.py <run> [<run> ...]
"""

import json
import sys
from pathlib import Path

import torch


def report(run):
    run = Path(run)
    ckpt = torch.load(run / "checkpoint.pt", map_location="cpu", weights_only=True)
    enc = ckpt["density_state"]["hac"]["encoded"]
    (run / "hac_size.json").write_text(json.dumps(enc, indent=1, sort_keys=True) + "\n")
    return enc


if __name__ == "__main__":
    for r in sys.argv[1:]:
        print(json.dumps({"run": r, "size_mb": report(r)["size_mb"]["total"]}))
