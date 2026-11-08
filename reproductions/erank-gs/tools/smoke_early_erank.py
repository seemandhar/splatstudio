"""Smoke helper: erank-gs with the erank loss started at step 500 (instead of 7000) so a
2000-step smoke exercises the loss. Evidence only; never used for benchmark numbers."""
import sys
from dataclasses import replace

from splatstudio import runtime
from splatstudio.registry import load

data, out = sys.argv[1], sys.argv[2]
m = load("erank-gs")
comps = tuple(replace(c, config={**c.config, "start": 500}) if c.name == "erank" else c
              for c in m.components)
m = replace(m, name="erank-gs", components=comps)
runtime.train(m, data, out, steps=2000, num_gaussians=10_000_000, max_gaussians=12_000_000,
              log_every=200, recipe="3dgs", device="cuda:0")
