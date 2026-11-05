"""Print the MesonGS bitstream report stored in a run's checkpoint; optionally export bins.zip.

Size = bytes of the zipped official bins/ directory (repo csv "Size [Bytes]"; paper MB = /2**20).
Usage: tools/ssenv python report_size.py <run_dir> [<run_dir> ...] [--export]
(--export writes <run_dir>/bins.zip; its members are the official save_npz files.)
Writes <run_dir>/mesongs_size.json.
"""

import json
import sys
from pathlib import Path

import torch


def report(run, export=False):
    ckpt = torch.load(Path(run) / "checkpoint.pt", map_location="cpu", weights_only=True)
    state = ckpt["density_state"]["mesongs"]
    enc = state["encoded"]
    if enc is None:
        raise SystemExit(f"{run}: no encode step ran (training length != total_steps?)")
    compress = next(e for e in state["events"] if e["mesongs"] == "compress")
    out = {"run": str(run), **enc, "compress": compress}
    if "bitstream" in state:
        data = state["bitstream"].numpy().tobytes()
        if len(data) != enc["zip_bytes"]:
            raise SystemExit(f"{run}: stored bitstream {len(data)} bytes != report")
        if export:
            path = Path(run) / "bins.zip"
            path.write_bytes(data)
            out["exported"] = str(path)
    (Path(run) / "mesongs_size.json").write_text(json.dumps(out, indent=1))
    return out


if __name__ == "__main__":
    export = "--export" in sys.argv[1:]
    for r in (a for a in sys.argv[1:] if a != "--export"):
        print(json.dumps(report(r, export), indent=1))
