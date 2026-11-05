"""Compressed-schedule smoke run: every Mini-Splatting event inside 2000 real-data steps.

Diagnostic only (not a benchmark): depth reinit at 1200, restart
(simplification + centre reinit, LR/SH restart) at 1300, CDF prune at 1700, num_depth 1M.

Usage: tools/ssenv python smoke_schedule.py <method> <data> <output>
"""

import sys
from dataclasses import replace

from splatstudio.registry import load as load_method
from splatstudio.runtime import train

COMPRESSED = {
    # one depth reinit at 1200: with ~160 views sampled with replacement, all views have been
    # rendered (GT colours cached) by then with high probability; 500 steps is too early.
    "densify_until": 1300,
    "reinit_every": 1200,
    "restart_step": 1300,
    "simp_iteration2": 1700,
    "num_depth": 1_000_000,
    "lr_max_steps": 2000,
    "lr_restart_offset": 333,
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
