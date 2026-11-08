"""Smoke test of the post-training pipeline in 2000 steps: 3DGS 1500 (densify until 1500),
8-cycle GaussianPOP prune (P=0.8) at 1500, 500 fine-tune steps. Not a benchmark."""

import json
import sys
from dataclasses import replace

from splatstudio.runtime import evaluate, train

from ss_gaussianpop import method

data, out = sys.argv[1], sys.argv[2]
m = method()
comps = []
for c in m.components:
    if c.kind == "density":
        cfg = dict(c.config, base_steps=1500, prune_step=1500)
        cfg["base"] = dict(cfg["base"], refine_stop_iter=1500)
        c = replace(c, config=cfg)
    comps.append(c)
m = replace(m, name="gaussianpop-smoke", components=tuple(comps))
train(m, data, out, steps=2000, recipe="3dgs", num_gaussians=10000000, max_gaussians=12000000,
      log_every=100, device="cuda:0")
print(json.dumps(evaluate(m, out, split="test", device="cuda:0")["metrics"]))
