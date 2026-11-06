"""Diagnostic-only training with density-config overrides (never a benchmark config).
    python diag_override.py MODE DATA OUT STEPS key=value ..."""
import json
import sys
from dataclasses import replace

from splatstudio.runtime import train
from ss_deblurring_3dgs import method

mode, data, out, steps = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
over = {k: json.loads(v) for k, v in (a.split("=", 1) for a in sys.argv[5:])}
m = method(mode)
m = replace(m, components=tuple(replace(c, config={**c.config, **over}) if c.kind == "density"
                                else c for c in m.components))
res = train(m, data, out, steps=steps, recipe="3dgs", num_gaussians=10000000,
            max_gaussians=12000000, log_every=500, device="cuda:0")
print(json.dumps({"validation_psnr": res["metrics"].get("psnr"), "gaussians":
                  res["resources"]["gaussians"], "overrides": over}))
