"""Train-view diagnostics for a run: PSNR on every 4th training view, Gaussian stats.

    ssenv python tools/train_view_eval.py <run_dir> [--data <scene>]
Uses the built-in gsplat rasterization (same as the plugin renderer RGB path)."""
import argparse
import json
from pathlib import Path

import torch
from splatstudio.backends.gsplat import Renderer
from splatstudio.data import load_dataset, on_device
from splatstudio.specs import ComponentSpec

p = argparse.ArgumentParser()
p.add_argument("run")
p.add_argument("--data")
p.add_argument("--every", type=int, default=4)
a = p.parse_args()
run = Path(a.run)
cfg = json.loads((run / "config.json").read_text())
ck = torch.load(run / "checkpoint.pt", map_location="cuda", weights_only=True)
params = ck["params"]
ds = load_dataset(a.data or cfg["data"])
r = Renderer(ComponentSpec("g", "renderer", "x", (), (), {}))
names = ds["splits"]["train"][:: a.every]
ps = []
with torch.no_grad():
    for n in names:
        f = on_device(ds["frames"][n], "cuda")
        out = r.render(params, f, sh_degree=ck["sh_degree"])
        mse = ((out["rgb"].clamp(0, 1) - f["image"]) ** 2).mean().clamp_min(1e-12)
        ps.append(float(-10 * torch.log10(mse)))
    s = params["scales"].exp().amax(-1)
    op = params["opacities"].sigmoid()
    ext = cfg.get("scene_scale")
res = {"run": str(run), "train_views": len(names), "train_psnr": sum(ps) / len(ps),
       "gaussians": len(s), "scale_max_q50_q99": [float(s.median()), float(s.quantile(0.99))
                                                    if len(s) < 16_000_000 else None],
       "frac_scale_gt_0.1": float((s > 0.1).float().mean()),
       "frac_opacity_lt_0.05": float((op < 0.05).float().mean())}
print(json.dumps(res))
