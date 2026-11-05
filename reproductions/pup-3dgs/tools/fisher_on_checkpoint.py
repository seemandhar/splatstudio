"""Real-scale check: PUP Fisher + one-shot prune on a trained SplatStudio checkpoint (no refine).

    tools/ssenv python tools/fisher_on_checkpoint.py --run runs/.../mipnerf360-bicycle --ratio 0.8
"""

import argparse
import json
import time
from pathlib import Path

import torch
from splatstudio.data import load_dataset, on_device
from splatstudio.methods import make_method
from splatstudio.components import build
from splatstudio.runtime import measure

from ss_pup3dgs import fisher

p = argparse.ArgumentParser()
p.add_argument("--run", required=True)
p.add_argument("--ratio", type=float, default=0.8)
p.add_argument("--output", required=True)
a = p.parse_args()
run = Path(a.run)
cfg = json.loads((run / "config.json").read_text())
ck = torch.load(run / "checkpoint.pt", map_location="cuda", weights_only=True)
params = {k: v.cuda() for k, v in ck["params"].items()}
ds = load_dataset(cfg["data"], cfg["downscale"])
cams = [{k: on_device(ds["frames"][n], "cuda")[k] for k in ("K", "viewmat", "width", "height")}
        for n in ds["splits"]["train"]]
torch.cuda.reset_peak_memory_stats()
torch.cuda.synchronize()
t0 = time.perf_counter()
sym = fisher.scene_fisher(params, cams, 4)
torch.cuda.synchronize()
t1 = time.perf_counter()
scores = fisher.log_det_scores(sym)
mask = fisher.prune_mask(scores, a.ratio)
torch.cuda.synchronize()
t2 = time.perf_counter()
renderer, _, _ = build(make_method("gsplat"))
before, _ = measure(renderer, params, ds, "test", 3, "cuda")
pruned = {k: v[~mask] for k, v in params.items()}
after, _ = measure(renderer, pruned, ds, "test", 3, "cuda")
# Reference: opacity-ranked pruning at the same ratio (a simple heuristic, for context only).
o = params["opacities"]
omask = fisher.prune_mask(o, a.ratio)
opa, _ = measure(renderer, {k: v[~omask] for k, v in params.items()}, ds, "test", 3, "cuda")
out = {"run": str(run), "gaussians": len(scores), "train_views": len(cams), "ratio": a.ratio,
       "pruned": int(mask.sum()), "zero_fisher": int(torch.isneginf(scores).sum()),
       "fisher_seconds": t1 - t0, "score_seconds": t2 - t1,
       "peak_memory_gb": torch.cuda.max_memory_allocated() / 2**30,
       "test_before": before, "test_after_prune_no_refine": after,
       "test_after_opacity_prune_no_refine": opa}
Path(a.output).write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))
