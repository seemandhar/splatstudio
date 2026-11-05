"""Compressed-schedule smoke run: every GaussianSpa event inside 3000 real-data steps.

Diagnostic only (not a benchmark): depth reinit at 1200, 15k-style sampling + reinit at 1300,
ADMM (z init) at 1500, sparsifying/multiplier updates every 50 steps until 2500, hard prune at
2500, light tuning to 3000. num_depth 1M.

Usage: tools/ssenv python smoke_schedule.py <method> <data> <output>
"""

import sys
from dataclasses import replace

from splatstudio.registry import load as load_method
from splatstudio.runtime import train

STEPS = 3000
WINDOW = {"spa_start": 1500, "spa_stop": 2500, "spa_interval": 50}
COMPRESSED = {
    "densify_until": 1300,
    "reinit_every": 1200,
    "restart_step": 1300,
    "num_depth": 1_000_000,
    "lr_max_steps": STEPS,
    "lr_restart_offset": 333,
    "log_every_updates": 2,
    **WINDOW,
}


def main(name, data, output):
    method = load_method(name)
    comps = []
    for c in method.components:
        if c.kind == "density":
            c = replace(c, config={**c.config, **COMPRESSED})
        elif c.name == "spa-penalty":
            c = replace(c, config={**c.config, **WINDOW})
        comps.append(c)
    method = replace(method, name=f"{name}-compressed", components=tuple(comps))
    result = train(method, data, output, steps=STEPS, recipe="3dgs", num_gaussians=10_000_000,
                   max_gaussians=12_000_000, log_every=50, device="cuda:0")
    print(result["metrics"], result["resources"])


if __name__ == "__main__":
    main(*sys.argv[1:4])
