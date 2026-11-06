"""Smoke-only check that the extra-point step runs inside real training: the defocus method with
pts_iter moved from 2500 to 1000 (diagnostic, never a benchmark config), 1500 steps."""
import json
import sys
from dataclasses import replace

from splatstudio.runtime import train
from ss_deblurring_3dgs import method

m = method(sys.argv[3] if len(sys.argv) > 3 else "defocus")
comps = tuple(replace(c, config={**c.config, "pts_iter": 1000}) if c.kind == "density" else c
              for c in m.components)
m = replace(m, components=comps)
res = train(m, sys.argv[1], sys.argv[2], steps=1500, recipe="3dgs", num_gaussians=10000000,
            max_gaussians=12000000, log_every=250, device="cuda:0")
print(json.dumps({"validation_psnr": res["metrics"].get("psnr"), "gaussians":
                  res["resources"]["gaussians"]}))
