"""Smoke test with a compressed schedule so every SparseGS mechanism fires within <= 2000 steps
(warp targets at 799, SDS after 1300, floater pruning at the last step). Evidence only: the
benchmark uses the registered `sparsegs` method with the paper schedule.

    SGS_DATA=<scene> tools/ssenv python reproductions/sparsegs/tools/smoke_compressed.py <scene> <out> [steps]
"""
import json
import sys
from dataclasses import replace

from splatstudio.runtime import evaluate, train

from ss_sparsegs import method

data, out = sys.argv[1], sys.argv[2]
steps = int(sys.argv[3]) if len(sys.argv) > 3 else 2000
off = set(sys.argv[4].split(",")) - {""} if len(sys.argv) > 4 else set()  # ablation: prune,sds,warp
prune_step = int(sys.argv[5]) if len(sys.argv) > 5 else steps
mode_range = sys.argv[6] if len(sys.argv) > 6 else "official"
m = method()
comps = []
for c in m.components:
    cfg = dict(c.config)
    if c.name == "sparsegs-density":
        cfg.update(total_steps=steps, prune_step=prune_step, warp_start=799, mode_range=mode_range, prune="prune" not in off,
                   warp="warp" not in off)
    if c.name == "sparsegs-warp":
        cfg.update(start=799, weight=0.0 if "warp" in off else cfg["weight"])
    if c.name == "sparsegs-sds":
        cfg.update(total_steps=steps, weight=0.0 if "sds" in off else cfg["weight"])
    comps.append(replace(c, config=cfg))
m = replace(m, name="sparsegs-smoke", components=tuple(comps))
res = train(m, data, out, steps=steps, num_gaussians=10000000, max_gaussians=12000000,
            device="cuda:0", log_every=100, recipe="3dgs")
test = evaluate(m, out, split="test", device="cuda:0")
print(json.dumps({"validation": res["metrics"], "test": test["metrics"],
                  "resources": res["resources"]}))
