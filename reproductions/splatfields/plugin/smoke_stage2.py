"""Short SplatFields smoke run that still bakes: total_steps (bake step) set to the run length.

Diagnostic only (not a benchmark). Trains `splatfields` with the density's `total_steps`
replaced by --steps (network LR schedule unchanged: 8e-4 -> 1.6e-6 over 40k), so the last step
bakes the predicted Gaussians; then clears the network slot and evaluates the baked checkpoint
on the test split with `runtime.evaluate` (what `ss-eval` runs), i.e. without the networks.

Usage: tools/ssenv python smoke_stage2.py <data> <output> [steps]
"""

import json
import sys
from dataclasses import replace

from splatstudio.registry import load as load_method
from splatstudio.runtime import evaluate, train

from ss_splatfields import network


def main(data, output, steps="2000"):
    steps = int(steps)
    method = load_method("splatfields")
    comps = tuple(replace(c, config={**c.config, "total_steps": steps})
                  if c.kind == "density" else c for c in method.components)
    method = replace(method, name="splatfields-smoke", components=comps)
    sys.argv += ["--data", data]  # mask-loss data root (as under ss-train)
    result = train(method, data, output, steps=steps, recipe="3dgs", num_gaussians=10_000_000,
                   max_gaussians=12_000_000, log_every=200, device="cuda:0")
    network.ACTIVE["net"] = None
    test = evaluate(method, output, split="test", device="cuda:0")
    print(json.dumps({"validation": result["metrics"], "test": test["metrics"],
                      "resources": result["resources"]}))


if __name__ == "__main__":
    main(*sys.argv[1:4])
