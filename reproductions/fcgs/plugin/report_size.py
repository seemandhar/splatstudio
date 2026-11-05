"""Print the FCGS size report a `fcgs` / `fcgs-lowrate` run stored in its checkpoint.

Usage: tools/ssenv python report_size.py <run_dir> [<run_dir> ...]

`size_MB` is the paper's size (sum of all bitstream bits / 8 / 2^20, as the official
encode script prints it). Check `xyz_coder_used`: "octree" means the GPCC (tmc3) binary was
not available and positions were coded with the lossless substitute coder.
"""

import json
import sys
from pathlib import Path

import torch


def report(run):
    ckpt = torch.load(Path(run) / "checkpoint.pt", map_location="cpu", weights_only=True)
    enc = ckpt["density_state"]["fcgs"]["encoded"]
    if enc is None:
        raise SystemExit(f"{run}: no FCGS encode ran (training length != total_steps?)")
    return {"run": str(run), **enc}


if __name__ == "__main__":
    for r in sys.argv[1:]:
        print(json.dumps(report(r), indent=1))
