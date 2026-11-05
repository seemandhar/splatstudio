"""Compressed-schedule smoke run: every EAGLES event inside 2000 real-data steps.

Diagnostic only (not a benchmark): progressive scale 0.3 -> 1 over steps 1..1400
(0.7 * (2000 + 1)), densification every 125 steps 500..1000 (gsplat refine_stop_iter 1000),
opacity reset of the opacity latent at 600, influence pruning at 700 and 1400, encode ->
entropy decode -> bake at 2000; then evaluates the baked checkpoint on the test split with
`runtime.evaluate` (what `ss-eval` runs for a registered method) in this process after clearing
the decoder bank, so the evaluation sees only the checkpoint's base fields.

Usage: tools/ssenv python smoke_schedule.py <data> <output>
"""

import json
import sys
from dataclasses import replace

from splatstudio.registry import load as load_method
from splatstudio.runtime import evaluate, train

from ss_eagles import decoders

DENSITY = {"total_steps": 2000, "opacity_reset_every": 600, "opacity_reset_until": 1000,
           "infl_prune_every": 700, "infl_prune_until": 1800,
           "base": {"absgrad": False, "grow_grad2d": 0.0002, "refine_start_iter": 500,
                    "refine_stop_iter": 1000, "refine_every": 125, "reset_every": 600}}


def main(data, output):
    method = load_method("eagles")
    comps = tuple(
        replace(c, config={**c.config, **DENSITY}) if c.kind == "density" else c
        for c in method.components
    )
    method = replace(method, name="eagles-compressed", components=comps)
    result = train(method, data, output, steps=2000, recipe="3dgs", num_gaussians=10_000_000,
                   max_gaussians=12_000_000, log_every=100, device="cuda:0")
    decoders.ACTIVE["bank"] = None
    test = evaluate(method, output, split="test", device="cuda:0")
    print(json.dumps({"validation": result["metrics"], "test": test["metrics"],
                      "resources": result["resources"]}))


if __name__ == "__main__":
    main(*sys.argv[1:3])
