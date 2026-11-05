"""Short smoke run of an MVGS preset with a compressed/truncated stage schedule.

ss-train refuses --steps that differ from the full schedule (39000 / 30000), so smoke tests call
splatstudio.runtime.train directly with the stage list replaced, then evaluate on the test split.
Usage: smoke.py METHOD DATA OUTPUT STAGES_JSON
"""

import json
import sys
from dataclasses import replace
from pathlib import Path

from splatstudio.registry import load as load_method
from splatstudio.runtime import evaluate, train

name, data, output, stages = sys.argv[1], sys.argv[2], sys.argv[3], json.loads(sys.argv[4])
base = load_method(name)
comps = tuple(
    replace(c, config={**c.config, "stages": stages, "data": str(Path(data).expanduser())})
    if c.kind == "density" else c
    for c in base.components
)
method = replace(base, components=comps)
steps = sum(s[2] for s in stages)
result = train(method, Path(data).expanduser(), output, steps=steps, recipe="3dgs",
               num_gaussians=10_000_000, max_gaussians=12_000_000, log_every=200,
               device="cuda:0")
test = evaluate(method, output, split="test", device="cuda:0")
print(json.dumps({"validation": result["metrics"], "test": test["metrics"],
                  "resources": result["resources"]}))
