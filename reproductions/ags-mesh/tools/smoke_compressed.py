"""Schedule-compressed smoke run (<= 2000 steps): same ags-mesh spec but with the depth switch,
normal start and ANR switch moved to ns 600 / 800 / 1400 so every phase of the official
schedule (7000 / 7000 / 15000 at 30k) is exercised. Smoke only, never a benchmark config."""

import os
import sys
from dataclasses import replace

from splatstudio import runtime

import ss_agsmesh

data, out = sys.argv[1], sys.argv[2]
mode = sys.argv[3] if len(sys.argv) > 3 else "train"
os.environ["AGSMESH_DATA"] = data
m = ss_agsmesh.method()
comps = []
for c in m.components:
    if c.name == "ags-depth":
        c = replace(c, config={**c.config, "depth_mask_step": 600})
    if c.name == "ags-normal":
        c = replace(c, config={**c.config, "normal_start": 800, "normal_mask_step": 1400})
    comps.append(c)
m = replace(m, components=tuple(comps))
if mode == "train":
    runtime.train(m, data, out, steps=2000, num_gaussians=10_000_000, max_gaussians=12_000_000,
                  device="cuda:0", log_every=100, recipe="splatstudio")
else:  # evaluate with the same (compressed) spec; mode = test | validation
    r = runtime.evaluate(m, out, split=mode, device="cuda:0")
    print(mode, r["metrics"])
