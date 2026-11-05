"""Apply SparseGS floater pruning (both index conventions) to a trained run's checkpoint and
measure test PSNR before/after (diagnostic evidence only).
    SGS_DATA=<scene> tools/ssenv python reproductions/sparsegs/tools/prune_offline.py <run>
"""
import json
import sys
from types import SimpleNamespace

import torch
from splatstudio.data import load_dataset
from splatstudio.runtime import measure

from ss_sparsegs.density import SparseGSDensity
from ss_sparsegs.renderer import Renderer

run = sys.argv[1]
cfg = json.load(open(f"{run}/config.json"))
ck = torch.load(f"{run}/checkpoint.pt", map_location="cuda:0", weights_only=True)
ds = load_dataset(cfg["data"], 1)
rnd = Renderer(SimpleNamespace(config={}))
cams = [{k: ds["frames"][n][k].to("cuda:0") if torch.is_tensor(ds["frames"][n][k]) else ds["frames"][n][k]
         for k in ("K", "viewmat", "width", "height")} for n in ds["splits"]["train"]]
out = {}
for mode in (None, "official", "paper"):
    params = torch.nn.ParameterDict({k: torch.nn.Parameter(v.clone()) for k, v in ck["params"].items()})
    if mode:
        opts = {k: torch.optim.Adam([p], lr=0.0) for k, p in params.items()}
        d = SparseGSDensity(SimpleNamespace(config={"mode_range": mode}))
        d.initialize(params, opts, 1.0)
        d.set_training_cameras(cams)
        res = d.prune_floaters(params, opts, rnd)
    with torch.no_grad():
        m, _ = measure(rnd, params, ds, "test", ck["sh_degree"], "cuda:0")
    out[mode or "none"] = {"test": m, "gaussians": len(params["means"]),
                           "pruned": None if not mode else res["pruned"]}
    print(mode, out[mode or "none"], flush=True)
json.dump(out, open(f"{run}/prune_offline.json", "w"), indent=1)
