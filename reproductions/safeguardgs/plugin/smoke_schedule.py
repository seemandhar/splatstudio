"""Compressed-schedule smoke run: the SafeguardGS prune event inside 2000 real-data steps.

Diagnostic only (not a benchmark): densification stops at 1000 (base refine_stop_iter), the
pruning event moves from 20000 to 1500, training ends at 2000. `--no-prune` gives the matching
reference without pruning; `--pixel` uses the paper-text per-pixel selection.

Usage: tools/ssenv python smoke_schedule.py <method> <data> <output> [--pixel] [--no-prune]
"""

import sys
from dataclasses import replace

from splatstudio.registry import load as load_method
from splatstudio.runtime import train


def main(name, data, output, *flags):
    method = load_method(name)
    over = {"prune_steps": [] if "--no-prune" in flags else [1500]}
    if "--pixel" in flags:
        over["selection"] = "pixel"

    def patch(c):
        if c.kind != "density":
            return c
        base = {**c.config["base"], "refine_stop_iter": 1000}
        return replace(c, config={**c.config, **over, "base": base})

    comps = tuple(patch(c) for c in method.components)
    method = replace(method, name=f"{name}-compressed", components=comps)
    result = train(method, data, output, steps=2000, recipe="3dgs", num_gaussians=10_000_000,
                   max_gaussians=12_000_000, log_every=100, device="cuda:0")
    print(result["metrics"], result["resources"])


if __name__ == "__main__":
    main(*sys.argv[1:])
