"""Evaluate the smoke_early_erank run with the same modified specification."""
import json
import sys
from dataclasses import replace

from splatstudio import runtime
from splatstudio.registry import load

m = load("erank-gs")
comps = tuple(replace(c, config={**c.config, "start": 500}) if c.name == "erank" else c
              for c in m.components)
m = replace(m, components=comps)
print(json.dumps(runtime.evaluate(m, sys.argv[1], split="test", device="cuda:0"), default=str))
