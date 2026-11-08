"""Per-loss gradient norms (means, opacities) of a det-gs checkpoint on a few training views.
Usage: ssenv python reproductions/det-gs/tools/grad_diag.py --data <scene copy> --run <run dir> --out <json>"""

import argparse
import json
import random
import sys
from pathlib import Path

import torch
from splatstudio.data import load_dataset
from splatstudio.registry import resolve

from ss_det_gs import method

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True)
ap.add_argument("--run", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--views", type=int, default=6)
args = ap.parse_args()
sys.argv += ["--data", args.data]
dev = "cuda"
ck = torch.load(Path(args.run) / "checkpoint.pt", map_location=dev, weights_only=False)
params = {k: torch.nn.Parameter(v.detach().to(dev)) for k, v in ck["params"].items()}
m = method()
comps = {c.name: c for c in m.components}
renderer = resolve(comps["det-gs-renderer"].factory)(comps["det-gs-renderer"])
names = ["photometric", "det-gs-depth", "det-gs-edge", "det-gs-tv"]
fns = {n: resolve(comps[n].factory)(comps[n]) for n in names}
ds = load_dataset(args.data)
random.seed(0)
views = random.sample(ds["splits"]["train"], args.views)
rows = []
for v in views:
    f = {k: (t.to(dev) if torch.is_tensor(t) else t) for k, t in ds["frames"][v].items()}
    row = {"view": v}
    for n in names:
        out = renderer.render(params, f, sh_degree=ck["sh_degree"])
        out["step"] = 5  # a depth step
        val = fns[n](params, out, f["image"])
        g = torch.autograd.grad(val, [params["means"], params["opacities"]], allow_unused=True)
        row[n] = {"value": float(val),
                  "grad_means": float(g[0].norm()) if g[0] is not None else 0.0,
                  "grad_opacities": float(g[1].norm()) if g[1] is not None else 0.0}
    rows.append(row)
summ = {n: {k: sum(r[n][k] for r in rows) / len(rows) for k in ("value", "grad_means", "grad_opacities")}
        for n in names}
for n in names[1:]:
    summ[n]["ratio_means_vs_photometric"] = summ[n]["grad_means"] / summ["photometric"]["grad_means"]
    summ[n]["ratio_opacities_vs_photometric"] = (summ[n]["grad_opacities"]
                                                 / max(summ["photometric"]["grad_opacities"], 1e-30))
res = {"run": args.run, "step": ck["step"], "gaussians": len(params["means"]), "mean": summ, "views": rows,
       "note": "weighted losses as summed in training (beta, phi, gamma/eta applied); depth evaluated as on a depth step"}
Path(args.out).write_text(json.dumps(res, indent=1))
print(json.dumps({"step": ck["step"], "mean": summ}, indent=1))
