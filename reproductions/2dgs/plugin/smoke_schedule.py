"""Smoke run with both 2DGS regularisers switched on early (step > 500) and the paper's
unbounded lambda_dist = 100, so a <= 2000-step run exercises the normal-consistency and
distortion gradients through the official kernel. Not a paper configuration.
Usage (inside tools/ssenv): python smoke_schedule.py --data <scene> --output <run> [--steps 2000]
"""

import argparse
from dataclasses import replace

from splatstudio.runtime import train

import ss_twodgs


def early_method(start):
    m = ss_twodgs.method_paperdist()
    comps = tuple(replace(c, config={**c.config, "start": start})
                  if c.name in {"2dgs-normal", "2dgs-distortion"} else c for c in m.components)
    return replace(m, name="2dgs-smoke-early", components=comps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--start", type=int, default=500)
    a = ap.parse_args()
    res = train(early_method(a.start), a.data, a.output, steps=a.steps, recipe="3dgs",
                num_gaussians=10000000, max_gaussians=12000000, log_every=200, device="cuda:0")
    print(res["metrics"], res["resources"])


if __name__ == "__main__":
    main()
