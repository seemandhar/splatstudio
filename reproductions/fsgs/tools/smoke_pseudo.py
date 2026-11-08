"""Short mechanism smoke: FSGS with the pseudo-view loss started early (step 300) so a <=1000-step
run exercises the DPT-gradient path at real resolution; logs unpooling/densify counts.
Not a benchmark configuration (pseudo start differs from the paper's 2000)."""

import json
import sys
from dataclasses import replace

import ss_fsgs.density as dens
from splatstudio.runtime import train
from ss_fsgs import method

data, out, steps = sys.argv[1], sys.argv[2], int(sys.argv[3])
m = method()
comps = [replace(c, config={**c.config, "start": 300}) if c.name == "fsgs-pseudo" else c
         for c in m.components]
m = replace(m, components=tuple(comps))
orig = dens.FSGSDensity.densify_and_prune


def logged(self, params, optimizers, step):
    row = orig(self, params, optimizers, step)
    print("DENSIFY", json.dumps(row), flush=True)
    return row


dens.FSGSDensity.densify_and_prune = logged
r = train(m, data, out, steps=steps, recipe="3dgs", num_gaussians=10_000_000,
          max_gaussians=12_000_000, log_every=100, device="cuda:0")
print("RESULT", json.dumps(r["metrics"]), r["resources"]["training_seconds"])
