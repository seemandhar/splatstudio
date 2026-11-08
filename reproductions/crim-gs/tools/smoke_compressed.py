"""Schedule-compressed smoke run (evidence only, not a benchmark): the real crim-gs spec with the
kernel stages moved to 300 / 600 so that 1000 steps exercise the sharp, averaged and
pixel-weight/mask paths on a real scene. Usage: smoke_compressed.py DATA OUTPUT STEPS"""
import sys
from dataclasses import replace

from splatstudio.runtime import train
from ss_crim_gs import method

m = method()
comps = tuple(replace(c, config={**c.config, "kernel_start_warp": 300,
                                 "kernel_start_pixel_weight": 600})
              if c.name == "crimgs-density" else c for c in m.components)
m = replace(m, name="crim-gs", components=comps)
print(train(m, sys.argv[1], sys.argv[2], steps=int(sys.argv[3]), recipe="3dgs",
            num_gaussians=10000000, max_gaussians=12000000, log_every=100,
            device="cuda:0")["metrics"])
