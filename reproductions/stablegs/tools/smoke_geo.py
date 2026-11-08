"""Smoke test of the post-15k phase in a short run: the stablegs spec with the dual/geometric
switch and the α freeze moved from 15000 to `--switch` (smoke only; not a benchmark config)."""
import argparse
import sys
from dataclasses import replace

from splatstudio.registry import load as load_method
from splatstudio.runtime import train

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True)
ap.add_argument("--output", required=True)
ap.add_argument("--steps", type=int, default=2000)
ap.add_argument("--switch", type=int, default=1000)
a = ap.parse_args()
sys.argv += ["--data", a.data]  # density reloads images from --data
m = load_method("stablegs")
comps = tuple(
    replace(c, config={**c.config, "dual_from": a.switch, "ho_iteration": a.switch})
    if c.kind == "density" else c for c in m.components)
m = replace(m, name="stablegs-smoke-geo", components=comps)
r = train(m, a.data, a.output, steps=a.steps, num_gaussians=10000000, max_gaussians=12000000,
          device="cuda:0", log_every=100, recipe="3dgs")
print(r["metrics"])
