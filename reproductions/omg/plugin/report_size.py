"""Print and save the OMG storage report of a finished run (<run>/omg_size.json).

Usage: tools/ssenv python report_size.py <run>
"""

import json
import sys
from pathlib import Path

import torch


def main(run):
    run = Path(run)
    ck = torch.load(run / "checkpoint.pt", map_location="cpu", weights_only=True)
    rep = ck["density_state"]["omg"]["encoded"]
    if rep is None:
        raise SystemExit("run has no OMG encode report (not a completed 30k OMG run)")
    rep = {**rep, "gaussians_final": int(ck["params"]["means"].shape[0])}
    (run / "omg_size.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main(sys.argv[1])
