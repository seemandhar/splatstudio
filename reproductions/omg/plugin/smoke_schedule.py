"""Compressed-schedule smoke run: every Mini-Splatting and OMG event inside 2000 real-data steps.

Diagnostic only, not a benchmark. The schedule (after the Mini-Splatting port's smoke_schedule.py):
- depth reinit at 1200;
- importance sampling + reinit + construct_net at 1300;
- LD prune at 1700;
- SVQ at 1900 (K-means after step 1899);
- encode -> decode -> bake at 2000.

The validation metrics printed at the end are measured on the decoded model.

Usage: tools/ssenv python smoke_schedule.py <method> <data> <output>
"""

import json
import sys
from dataclasses import replace

from splatstudio.registry import load as load_method
from splatstudio.runtime import train

COMPRESSED = {
    "densify_until": 1300,
    "reinit_every": 1200,
    "restart_step": 1300,
    "simp_iteration2": 1700,
    "num_depth": 1_000_000,
    "lr_max_steps": 2000,
    "lr_restart_offset": 333,
    "net_itr": 1300,
    "svq_itr": 1900,
    "total_steps": 2000,
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
    print(json.dumps({"metrics": result["metrics"], "resources": result["resources"]}))


if __name__ == "__main__":
    main(*sys.argv[1:4])
