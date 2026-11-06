"""Diagnostic only: train the defocus method for N steps and log blur-factor statistics of the
training render every 50 steps (fraction of ds/dr above 1, at the 1.1 cap, and MLP grad norm).
    python diag_deltas.py DATA OUT STEPS [mode] [key=value ...]"""
import json
import sys
from dataclasses import replace

import torch

import ss_deblurring_3dgs.density as D
from splatstudio.runtime import train
from ss_deblurring_3dgs import method

data, out, steps = sys.argv[1], sys.argv[2], int(sys.argv[3])
mode = sys.argv[4] if len(sys.argv) > 4 and "=" not in sys.argv[4] else "defocus"
over = {k: json.loads(v) for k, v in (a.split("=", 1) for a in sys.argv[4:] if "=" in a)}
m = method(mode)
m = replace(m, components=tuple(replace(c, config={**c.config, **over}) if c.kind == "density"
                                else c for c in m.components))
orig = D.Density.after_backward
log = open(out + ".deltas.jsonl", "w")


def patched(self, params, optimizers, step, outputs):
    if step % 50 == 0 or step in (1, 2499, 2501, 2502):
        d = outputs["deltas"]
        g = sum(float(p.grad.norm()) ** 2 for p in self.bank.net.parameters()
                if p.grad is not None) ** 0.5
        row = {"step": step, "mlp_grad": g}
        for k in ("ds", "dr"):
            v = d[k].detach()
            row[k] = {"gt1": float((v > 1).float().mean()), "cap": float((v >= 1.0999).float().mean()),
                      "mean": float(v.mean())}
        bias = {n: float(p.detach().abs().mean()) for n, p in self.bank.net.named_parameters()
                if n in ("s.bias", "r.bias")}
        row["bias"] = bias
        log.write(json.dumps(row) + "\n")
        log.flush()
    return orig(self, params, optimizers, step, outputs)


D.Density.after_backward = patched
train(m, data, out, steps=steps, recipe="3dgs", num_gaussians=10000000, max_gaussians=12000000,
      log_every=500, device="cuda:0")
