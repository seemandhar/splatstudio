"""Diagnostic smoke: erank-gs-adc with density config overrides (key=value ...), 2000 steps.
Evidence only (diagnosing the 2k-step gap of the fixed ADC); never used for benchmarks."""
import json
import sys
from dataclasses import replace

from splatstudio import runtime
from splatstudio.registry import load

data, out, *kv = sys.argv[1:]
over = {k: json.loads(v) for k, v in (a.split("=", 1) for a in kv)}
m = load("erank-gs-adc")
comps = tuple(replace(c, config={**c.config, **over}) if c.name == "erank-density" else c
              for c in m.components)
m = replace(m, components=comps)
runtime.train(m, data, out, steps=2000, num_gaussians=10_000_000, max_gaussians=12_000_000,
              log_every=200, recipe="3dgs", device="cuda:0")
print(json.dumps(runtime.evaluate(m, out, split="test", device="cuda:0")["metrics"]))
