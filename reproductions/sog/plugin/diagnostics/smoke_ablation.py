"""Diagnostic 2000-step ablations for the smoke red-flag check (not benchmark configurations).

Usage: tools/ssenv python smoke_ablation.py <variant> <data> <output>
  noencode : sog-smoke with encode disabled (isolates codec loss)
  densonly : gsplat with SOG's densification config only (no sort/neighbor/log/codec)
"""

import sys
from dataclasses import replace

from splatstudio.methods import make_method
from splatstudio.runtime import evaluate, train

from ss_sog import smoke
from ss_sog.density import DEFAULTS


def variant(name):
    if name == "noencode":
        m = smoke()
        comps = tuple(replace(c, config={**c.config, "encode": False}) if c.kind == "density"
                      else c for c in m.components)
        return replace(m, name="sog-smoke-noencode", components=comps)
    if name == "densonly":
        m = make_method("gsplat-sogdens")
        r, d, p = m.components
        return replace(m, components=(r, replace(d, config=dict(DEFAULTS["base"])), p))
    raise SystemExit(name)


if __name__ == "__main__":
    m = variant(sys.argv[1])
    train(m, sys.argv[2], sys.argv[3], steps=2000, recipe="3dgs", num_gaussians=10000000,
          max_gaussians=12000000, log_every=500, device="cuda:0")
    print(evaluate(m, sys.argv[3], split="test", device="cuda:0")["metrics"])
