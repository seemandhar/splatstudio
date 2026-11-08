"""Compressed-schedule smoke run: the whole LP-3DGS schedule inside 2000 real-data steps.

Diagnostic only (not a benchmark): densification stops at 1000, mask window 1200-1700 (score
every 20 steps), one-time prune at 1700, 300 fine-tune steps. Mask-loss window follows.

Usage: tools/ssenv python smoke_schedule.py <data> <output>
"""

import sys
from dataclasses import replace

from splatstudio.registry import load as load_method
from splatstudio.runtime import train

DENSITY = {"refine_stop_iter": 1000, "mask_start": 1200, "mask_iters": 500}
LOSS = {"mask_start": 1200, "mask_iters": 500}


def main(data, output):
    method = load_method("lp-3dgs")
    comps = []
    for c in method.components:
        if c.kind == "density":
            c = replace(c, config={**c.config, **DENSITY})
        elif c.name == "lp-mask-loss":
            c = replace(c, config={**c.config, **LOSS})
        comps.append(c)
    method = replace(method, name="lp-3dgs-compressed", components=tuple(comps))
    result = train(method, data, output, steps=2000, recipe="3dgs", num_gaussians=10_000_000,
                   max_gaussians=12_000_000, log_every=100, device="cuda:0")
    print(result["metrics"], result["resources"])


if __name__ == "__main__":
    main(*sys.argv[1:3])
