"""Compressed-schedule smoke run: every Compact-3DGS phase inside N (default 2000) real-data steps.

Diagnostic only (not a benchmark): densification + mask pruning 500..N/2, mask pruning every
N/20 afterwards, net lr warm-up 100 and decay at N/6, N/2, 5N/6, R-VQ for the last N/30 steps,
encode -> decode -> bake at N; then evaluates the baked checkpoint on the test split with
`runtime.evaluate` (what `ss-eval` runs) after clearing the live bank, so evaluation sees only the
checkpoint.

Usage: tools/ssenv python smoke_schedule.py <data> <output> [steps]
"""

import json
import sys
from dataclasses import replace

from splatstudio.registry import load as load_method
from splatstudio.runtime import evaluate, train

from ss_compact_3dgs import model


def compressed(steps):
    f = steps / 30000
    half = max(int(15000 * f), 600)
    return {"total_steps": steps, "densify_until": half,
            "base": {"absgrad": False, "grow_grad2d": 0.0002, "refine_stop_iter": half},
            "mask_prune_every": max(int(1000 * f), 1), "rvq_iter": steps - max(int(1000 * f), 1),
            "net_lr_milestones": [int(5000 * f), int(15000 * f), int(25000 * f)]}


def main(data, output, steps="2000"):
    steps = int(steps)
    method = load_method("compact-3dgs")
    comps = tuple(
        replace(c, config={**c.config, **compressed(steps)}) if c.kind == "density" else c
        for c in method.components
    )
    method = replace(method, name="compact-3dgs-compressed", components=comps)
    result = train(method, data, output, steps=steps, recipe="3dgs", num_gaussians=10_000_000,
                   max_gaussians=12_000_000, log_every=100, device="cuda:0")
    model.ACTIVE["bank"] = None
    test = evaluate(method, output, split="test", device="cuda:0")
    print(json.dumps({"validation": result["metrics"], "test": test["metrics"],
                      "resources": result["resources"]}))


if __name__ == "__main__":
    main(*sys.argv[1:4])
