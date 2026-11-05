"""Compressed-schedule smoke run: every HAC phase inside 2000 real-data steps.

Diagnostic only (not a benchmark): Scaffold-GS phase 1..300, noise quantization 301..1000,
anchor-bound update at 1000, hash-grid context + rate loss 1001..2000, anchor growing every
100 steps from 150 to 1500 (paused in [300, 400)), encode -> decode -> bake at 2000; then
evaluates the baked checkpoint on the test split with `runtime.evaluate` (what `ss-eval` runs)
after clearing the live bank, so evaluation sees only the checkpoint.

Usage: tools/ssenv python smoke_schedule.py <data> <output> [steps]
"""

import json
import sys
from dataclasses import replace

from splatstudio.registry import load as load_method
from splatstudio.runtime import evaluate, train

from ss_hac import model


def compressed(steps):
    f = steps / 30000
    return {"total_steps": steps, "noise_start": int(3000 * f), "entropy_start": int(10000 * f),
            "start_stat": int(500 * f), "update_from": int(1500 * f),
            "update_until": int(15000 * f), "densify_pause": [int(3000 * f), int(4000 * f)]}


def main(data, output, steps="2000"):
    steps = int(steps)
    method = load_method("hac")
    comps = tuple(
        replace(c, config={**c.config, **compressed(steps)}) if c.kind == "density" else c
        for c in method.components
    )
    method = replace(method, name="hac-compressed", components=comps)
    result = train(method, data, output, steps=steps, recipe="3dgs", num_gaussians=10_000_000,
                   max_gaussians=12_000_000, log_every=100, device="cuda:0")
    model.ACTIVE["bank"] = None
    test = evaluate(method, output, split="test", device="cuda:0")
    print(json.dumps({"validation": result["metrics"], "test": test["metrics"],
                      "resources": result["resources"]}))


if __name__ == "__main__":
    main(*sys.argv[1:4])
