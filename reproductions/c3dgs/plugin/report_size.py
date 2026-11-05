"""Print the c3dgs size report stored in a run's checkpoint; optionally export the npz.

The size is the paper's: len(point_cloud.npz) / 2**20 (compress.py: os.path.getsize / 1024**2).
Usage: tools/ssenv python report_size.py <run_dir> [<run_dir> ...] [--export]
(--export writes <run_dir>/point_cloud.npz, loadable by the official GaussianModel.load_npz).
"""

import json
import sys
from pathlib import Path

import torch


def report(run, export=False):
    ckpt = torch.load(Path(run) / "checkpoint.pt", map_location="cpu", weights_only=True)
    state = ckpt["density_state"]["c3dgs"]
    enc = state["encoded"]
    if enc is None:
        raise SystemExit(f"{run}: no encode step ran (training length != total_steps?)")
    compress = next(e for e in state["events"] if e["c3dgs"] == "compress")
    out = {"run": str(run), **{k: v for k, v in enc.items() if k != "qparams"},
           "compress": compress}
    if "npz" in state:
        data = state["npz"].numpy().tobytes()
        if len(data) != enc["npz_bytes"]:
            raise SystemExit(f"{run}: stored npz has {len(data)} bytes, report {enc['npz_bytes']}")
        if export:
            path = Path(run) / "point_cloud.npz"
            path.write_bytes(data)
            out["exported"] = str(path)
            out["exported_MB"] = path.stat().st_size / 2**20
    return out


if __name__ == "__main__":
    export = "--export" in sys.argv[1:]
    for r in (a for a in sys.argv[1:] if a != "--export"):
        print(json.dumps(report(r, export), indent=1))
