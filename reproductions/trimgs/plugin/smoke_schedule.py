"""Compressed-schedule smoke run: every Trim3DGS event inside 2000 real-data steps.

Diagnostic only (not a benchmark): pre-training 1000 steps (densification 500..1000), then a
1000-step fine-tune with scale split + trimming every 200 steps (t = 200..800), kNN refresh
every 100 steps, normal loss from step 1001.

Usage: tools/ssenv python smoke_schedule.py <method> <data> <output>
"""

import sys
from dataclasses import replace

from splatstudio.registry import load as load_method
from splatstudio.runtime import train

DENSITY = {"pretrain_steps": 1000, "finetune_steps": 1000, "densify_interval": 200,
           "trim_interval": 200, "refine_stop_iter": 1000}
LOSS = {"start_step": 1000, "knn_interval": 100}


def main(name, data, output):
    method = load_method(name)
    comps = tuple(
        replace(c, config={**c.config, **DENSITY}) if c.kind == "density"
        else replace(c, config={**c.config, **LOSS}) if c.name == "trim-normal" else c
        for c in method.components
    )
    method = replace(method, name=f"{name}-compressed", components=comps)
    result = train(method, data, output, steps=2000, recipe="3dgs", num_gaussians=10_000_000,
                   max_gaussians=12_000_000, log_every=100, device="cuda:0")
    print(result["metrics"], result["resources"])


if __name__ == "__main__":
    main(*sys.argv[1:4])
