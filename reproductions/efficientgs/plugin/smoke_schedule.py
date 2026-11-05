"""Compressed-schedule smoke run: every EfficientGS event inside 2000 real-data steps.

Diagnostic only (not a benchmark): densification until 1300, top-K pruning at 1300/1500/1700,
SH allocation at 1350 and sparse increments at 1400/1500/1600 (interval 100).

Usage: tools/ssenv python smoke_schedule.py <method> <data> <output>
"""

import sys
from dataclasses import replace

from splatstudio.registry import load as load_method
from splatstudio.runtime import train

COMPRESSED = {
    "densify_until": 1300,
    "prune_steps": [1300, 1500, 1700],
    "sh_alloc_step": 1350,
    "sh_interval": 100,
}


def main(name, data, output):
    method = load_method(name)
    comps = tuple(
        replace(c, config={**c.config, **COMPRESSED}) if c.kind == "density" else c
        for c in method.components
    )
    method = replace(method, name=f"{name}-compressed", components=comps)
    result = train(method, data, output, steps=2000, recipe="3dgs", num_gaussians=10_000_000,
                   max_gaussians=12_000_000, log_every=100, device="cuda:0")
    print(result["metrics"], result["resources"])


if __name__ == "__main__":
    main(*sys.argv[1:4])
