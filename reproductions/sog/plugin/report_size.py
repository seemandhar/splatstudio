"""Write the encoded SOG files stored in a run's checkpoint and report the paper's size.

Usage: tools/ssenv python report_size.py <run_dir> [<run_dir> ...]

Writes <run>/compression/jxl_quant_sh/{*.jxl, compression_info.csv, compression_config.yml}
and prints Size [Bytes] = sum of the files in that directory (official
compression/decompress.py:get_size_of_files_in_dir) and MB = bytes / 1e6 (paper tables).
"""

import json
import os
import sys
from pathlib import Path

import torch


def size_of_files_in_dir(directory):
    return sum(os.path.getsize(Path(d) / f) for d, _, fs in os.walk(directory) for f in fs)


def report(run):
    run = Path(run)
    ckpt = torch.load(run / "checkpoint.pt", map_location="cpu", weights_only=True)
    sog = ckpt["density_state"]["sog"]
    if sog["encoded"] is None:
        raise SystemExit(f"{run}: no encode step ran (training length != total_steps?)")
    out = run / "compression" / sog["encoded"]["experiment"]
    out.mkdir(parents=True, exist_ok=True)
    for name, data in sog["files"].items():
        (out / name).write_bytes(bytes(data.numpy()))
    size = size_of_files_in_dir(out)
    if size != sog["encoded"]["size_bytes"]:
        raise SystemExit(f"{run}: written size {size} != recorded {sog['encoded']['size_bytes']}")
    sorts = [e for e in sog["events"] if e["sog"] == "sort"]
    return {"run": str(run), "size_bytes": size, "size_mb": size / 1e6,
            "gaussians": sog["encoded"]["gaussians"], "grid_side": sog["encoded"]["grid_side"],
            "file_bytes": sog["encoded"]["file_bytes"], "codec": sog["encoded"]["codec"],
            "sorts": len(sorts), "sort_seconds_total": round(sum(e["seconds"] for e in sorts), 1),
            "compressed_dir": str(out)}


if __name__ == "__main__":
    for r in sys.argv[1:]:
        print(json.dumps(report(r), indent=1))
