"""Per-scene settings of the official 3dgs-mcmc configs/<scene>.json (commit 7b4fc9f).
cap_max = final Gaussian count of the original 3DGS run (paper Sec. 4, README)."""

import sys
from pathlib import Path

CAPS = {
    "bicycle": 5_900_000, "garden": 5_200_000, "stump": 4_750_000,
    "room": 1_500_000, "counter": 1_200_000, "kitchen": 1_800_000, "bonsai": 1_300_000,
    "truck": 2_600_000, "train": 1_100_000,
    "drjohnson": 3_400_000, "playroom": 2_500_000,
}
# README: opacity_reg 0.001 for Deep Blending, 0.01 elsewhere.
OPACITY_REG = {"drjohnson": 0.001, "playroom": 0.001}


def argv_value(flag, argv=None):
    argv = sys.argv if argv is None else argv
    for i, arg in enumerate(argv):
        if arg == flag and i + 1 < len(argv):
            return argv[i + 1]
        if arg.startswith(flag + "="):
            return arg.split("=", 1)[1]
    return None


def scene_name():
    root = argv_value("--data")
    return Path(root).name if root else None
