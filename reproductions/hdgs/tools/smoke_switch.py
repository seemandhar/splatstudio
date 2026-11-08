"""Short stage-switch smoke: the installed `hdgs` spec with stage1_steps shortened so a run of
`--steps` crosses Fisher pruning + texture init + stage-2 training, then evaluates the run in a
fresh process (texture reloaded from checkpoint.pt via --run). Usage:
  smoke_switch.py train <out> <data> <stage1_steps> <steps>
  smoke_switch.py eval <out> <stage1_steps> --run <out>
"""

import json
import sys
from dataclasses import replace

from splatstudio import runtime
from splatstudio.registry import load as load_method


def short_method(s1):
    m = load_method("hdgs")
    comps = tuple(replace(c, config={**c.config, "stage1_steps": s1}) if c.name == "hdgs-density"
                  else c for c in m.components)
    return replace(m, name="hdgs", components=comps)


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "train":
        out, data, s1, steps = sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5])
        res = runtime.train(short_method(s1), data, out, steps=steps, recipe="3dgs",
                            num_gaussians=10000000, max_gaussians=12000000, log_every=250,
                            device="cuda:0")
        print(json.dumps(res["metrics"]))
    else:
        out, s1 = sys.argv[2], int(sys.argv[3])
        res = runtime.evaluate(short_method(s1), out, split="test", device="cuda:0")
        print(json.dumps({"metrics": res["metrics"], "resources": res["resources"]}))
