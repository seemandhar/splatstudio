"""Print the RDO-Gaussian bitstream report a run stored in its checkpoint, and optionally write
the official `bitstream/` files (index_bitstream.bin, header.bin, codebook.npz, logits.npz,
position.npz) next to it.

Usage: tools/ssenv python report_size.py [--write] <run_dir> [<run_dir> ...]
Writes <run_dir>/rdo_size.json. The paper's rate is `size_MB` (bytes / 1e6), never
checkpoint_bytes.
"""

import json
import sys
from pathlib import Path

import torch


def report(run, write=False):
    run = Path(run)
    ckpt = torch.load(run / "checkpoint.pt", map_location="cpu", weights_only=True)
    state = ckpt["density_state"]["rdo"]
    enc = state["encoded"]
    if enc is None:
        raise SystemExit(f"{run}: no encode step ran (training length != total_steps?)")
    if write:
        out = run / "bitstream"
        out.mkdir(exist_ok=True)
        for name, data in state["bitstream"].items():
            (out / name).write_bytes(data.numpy().tobytes())
        on_disk = sum((out / n).stat().st_size for n in state["bitstream"])
        if on_disk != enc["total_bytes"]:
            raise SystemExit(f"{run}: written bitstream {on_disk} B != reported "
                             f"{enc['total_bytes']} B")
    stats = [e for e in state["events"] if e.get("rdo") == "stats"]
    result = {"run": str(run), **enc, "last_stats": stats[-1] if stats else None}
    (run / "rdo_size.json").write_text(json.dumps(result, indent=1) + "\n")
    return result


if __name__ == "__main__":
    args = sys.argv[1:]
    write = "--write" in args
    for r in [a for a in args if a != "--write"]:
        res = report(r, write)
        print(json.dumps({k: res[k] for k in ("run", "size_MB", "gaussians",
                                              "gaussian_prune_ratio", "sh_prune_ratio",
                                              "components_MB")}, indent=1))
