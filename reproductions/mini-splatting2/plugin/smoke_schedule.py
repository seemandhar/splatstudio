"""Compressed-schedule smoke run: every Mini-Splatting2 event inside 2000 real-data steps.

Diagnostic only (not a benchmark): aggressive clones every 100 steps from 200, depth reinit at
600, densification until 900, simplification at 900 (sampling / importance pruning) and 1300,
culling off at (1300 + 2000) // 2 = 1650, half resolution and SH 0 until 900.

Usage: tools/ssenv python smoke_schedule.py <method> <data> <output>
"""

import sys
from dataclasses import replace

from splatstudio.registry import load as load_method
from splatstudio.runtime import train

COMPRESSED = {
    "total_steps": 2000,
    "densify_from": 200,
    "densify_until": 900,
    "clone_from": 200,
    "clone_every": 100,
    "depth_reinit_iter": 600,
    "simp_iteration1": 900,
    "simp_iteration2": 1300,
    "warmup_until": 900,
    "sh_from": 900,
    "lr_max_steps": 3333,
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
