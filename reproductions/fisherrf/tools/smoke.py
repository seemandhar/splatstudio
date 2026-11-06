"""Short active-selection smoke test through the native runtime (splatstudio.runtime.train).

A scaled schedule (4 init -> 8 views, 1 view / 25 epochs, 2000 effective iterations) of the
registered method, so the whole acquire/restore loop runs in a few minutes.
"""
import argparse
import json
import time

from splatstudio.runtime import evaluate, train

from ss_fisherrf import build

ap = argparse.ArgumentParser()
ap.add_argument("--selector", default="fisher")
ap.add_argument("--data", required=True)
ap.add_argument("--output", required=True)
ap.add_argument("--steps", type=int, default=40000)
a = ap.parse_args()
m = build(f"fisherrf-smoke-{a.selector}", selector=a.selector, total_views=8, num_init_views=4,
          interval_epochs=25, total_iterations=2000)
t = time.time()
train(m, a.data, a.output, steps=a.steps, recipe="3dgs", num_gaussians=10_000_000,
      max_gaussians=12_000_000, log_every=2000, device="cuda:0")
res = evaluate(m, a.output, split="test", device="cuda:0")
print(json.dumps({"selector": a.selector, "metrics": res["metrics"],
                  "gaussians": res["resources"]["gaussians"], "seconds": time.time() - t}))
