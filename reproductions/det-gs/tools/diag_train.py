"""Gap-analysis diagnostics: train a det-gs variant on the train-only diagnostic split and evaluate
held-out *training* views (validation + test halves). Not method results.

  ssenv python -I reproductions/det-gs/tools/diag_train.py VARIANT OUTPUT [--steps 7000] [--data D]

Variants: gsplat, full, nodepth, notv, noedge, disparity, sepstep (DNGaussian-style separate
backward + optimizer step for the depth loss, after the photometric step).
"""

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import torch
from splatstudio import ComponentSpec
from splatstudio.methods import make_method
from splatstudio.runtime import evaluate, train

ap = argparse.ArgumentParser()
ap.add_argument("variant")
ap.add_argument("output")
ap.add_argument("--steps", type=int, default=7000)
ap.add_argument("--data", default=str(Path("~/datasets/detgs-depth/mipnerf360-diag/garden").expanduser()))
args = ap.parse_args()
sys.argv += ["--data", args.data]  # priors lookup

from ss_det_gs import method as det_method  # noqa: E402


def drop(m, name):
    return replace(m, components=tuple(c for c in m.components if c.name != name))


def set_cfg(m, name, **kw):
    return replace(m, components=tuple(replace(c, config={**c.config, **kw}) if c.name == name else c
                                       for c in m.components))


v = args.variant
if v == "gsplat":
    m = make_method("gsplat")
else:
    m = det_method()
    if v == "nodepth":
        m = drop(m, "det-gs-depth")
    elif v == "notv":
        m = drop(m, "det-gs-tv")
    elif v == "noedge":
        m = drop(m, "det-gs-edge")
    elif v == "disparity":
        m = set_cfg(m, "det-gs-depth", space="disparity")
    elif v == "sepstep":
        cfg = next(c for c in m.components if c.name == "det-gs-depth").config
        m = drop(m, "det-gs-depth")
        m = replace(m, components=tuple(
            ComponentSpec(c.name, c.kind, "diag_sepstep:create_density", c.requires, c.provides,
                          {"base": c.config, "depth": dict(cfg)}) if c.kind == "density" else c
            for c in m.components))
    elif v != "full":
        raise SystemExit(f"unknown variant {v}")
    if v != "full":
        m = replace(m, name=f"det-gs-diag-{v}")
sys.path.insert(0, str(Path(__file__).resolve().parent))
res = train(m, args.data, args.output, steps=args.steps, num_gaussians=10_000_000,
            max_gaussians=12_000_000, device="cuda:0", log_every=500, recipe="3dgs")
test = evaluate(m, args.output, split="test", device="cuda:0")
nv, nt = len(res["per_view"]), len(test["per_view"])
out = {"variant": v, "steps": args.steps, "gaussians": res["resources"]["gaussians"],
       "minutes": res["resources"]["training_seconds"] / 60,
       "heldout_train_views": {k: (res["metrics"][k] * nv + test["metrics"][k] * nt) / (nv + nt)
                               for k in ("psnr", "ssim", "lpips")}}
Path(args.output, "diag.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out))
