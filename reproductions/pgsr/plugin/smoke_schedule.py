"""Smoke run with the geometric regularisers switched on early (step > 500 instead of > 7000)
so that a <= 2000-step run exercises the single-view, multi-view geometric and NCC losses,
the neighbour renders and the plane-geometry path of the density controller.

Not a paper configuration. Usage (inside tools/ssenv):
    python smoke_schedule.py --data <scene> --output <run> [--steps 2000]
"""

import argparse
from dataclasses import replace

from splatstudio.runtime import train

import ss_pgsr


def early_method(start, data):
    m = ss_pgsr.method()
    comps = []
    for c in m.components:
        cfg = dict(c.config)
        for key in ("single_view_weight_from_iter", "multi_view_weight_from_iter",
                    "geometry_from_iter"):
            if key in cfg:
                cfg[key] = start
        if c.name == "pgsr-density":
            cfg["data"] = data
        comps.append(replace(c, config=cfg))
    return replace(m, name="pgsr-smoke-early", components=tuple(comps))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--start", type=int, default=500)
    a = ap.parse_args()
    res = train(early_method(a.start, a.data), a.data, a.output, steps=a.steps, recipe="3dgs",
                num_gaussians=10000000, max_gaussians=12000000, log_every=100, device="cuda:0")
    print(res["metrics"], res["resources"])


if __name__ == "__main__":
    main()
