"""Smoke test of the full pipeline in 2000 steps: 3DGS 1000 (densify until 1000), prune at 1000
and 1500 with the paper ratios, refine with the official fine-tune LR schedule. Not a benchmark."""

import json
import sys
from dataclasses import replace

from splatstudio.runtime import evaluate, train

from ss_pup3dgs import method

data, out = sys.argv[1], sys.argv[2]
m = method()
comps = []
for c in m.components:
    if c.kind == "density":
        cfg = dict(c.config, base_steps=1000, prune_steps=[1000, 1500])
        cfg["base"] = dict(cfg["base"], refine_stop_iter=1000)
        c = replace(c, config=cfg)
    comps.append(c)
m = replace(m, name="pup-3dgs-smoke", components=tuple(comps))
train(m, data, out, steps=2000, recipe="3dgs", num_gaussians=10000000, max_gaussians=12000000,
      log_every=100, device="cuda:0")
print(json.dumps(evaluate(m, out, split="test", device="cuda:0")["metrics"]))
