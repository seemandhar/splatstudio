"""Compressed-schedule smoke run: every Reduced-3DGS event inside 2000 real-data steps.

Diagnostic only (not a benchmark): densification 500..1000 (gsplat refine_stop_iter 1000),
mercy every 200 steps (skipped at none: no opacity reset before 3000) up to step 1700, dead-point
pruning every 100 steps from 1000 and at the "save" step 700, SH culling at 1500 (band 1 is
active from step 1001 in the runtime schedule; at 1000 every Gaussian is still RGB-only), encode ->
decode at 2000; then evaluates the decoded checkpoint on the test split (runtime.evaluate,
which is what `ss-eval` runs for a registered method).

Usage: tools/ssenv python smoke_schedule.py <data> <output>
"""

import json
import sys
from dataclasses import replace

from splatstudio.registry import load as load_method
from splatstudio.runtime import evaluate, train

DENSITY = {"total_steps": 2000, "densify_until": 1000, "cull_sh_step": 1500,
           "mercy_interval": 200, "mercy_finetune": 300, "save_steps": [700],
           "base": {"absgrad": False, "grow_grad2d": 0.0002, "refine_stop_iter": 1000}}


def main(data, output):
    method = load_method("reduced-3dgs")
    comps = tuple(
        replace(c, config={**c.config, **DENSITY}) if c.kind == "density" else c
        for c in method.components
    )
    method = replace(method, name="reduced-3dgs-compressed", components=comps)
    result = train(method, data, output, steps=2000, recipe="3dgs", num_gaussians=10_000_000,
                   max_gaussians=12_000_000, log_every=100, device="cuda:0")
    test = evaluate(method, output, split="test", device="cuda:0")
    print(json.dumps({"validation": result["metrics"], "test": test["metrics"],
                      "resources": result["resources"]}))


if __name__ == "__main__":
    main(*sys.argv[1:3])
