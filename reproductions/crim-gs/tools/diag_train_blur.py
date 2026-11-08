"""Gap diagnostic (train views only): blur-reconstruction PSNR of the blurry training images,
sharp-render PSNR vs the blurry GT, trajectory extent and mask mean, from a finished run.
Usage: diag_train_blur.py RUN_DIR [OUT_JSON]"""
import json
import math
import sys
from pathlib import Path

import torch
from splatstudio.components import build
from splatstudio.data import load_dataset, on_device
from splatstudio.registry import load as load_method

from ss_crim_gs import kernel as kernel_mod
from ss_crim_gs.kernel import CRiMKernel

run = Path(sys.argv[1])
cfg = json.loads((run / "config.json").read_text())
ck = torch.load(run / "checkpoint.pt", map_location="cuda", weights_only=True)
method = load_method(cfg["method"])
renderer, density, _ = build(method)
ds = load_dataset(cfg["data"], cfg["downscale"])
names = ds["splits"]["train"]
cams = [{k: on_device(ds["frames"][n], "cuda")[k] for k in ("K", "viewmat", "width", "height")}
        for n in names]
kcfg = {k[len("kernel_"):]: v for k, v in density.cfg.items()
        if k.startswith("kernel_") and k != "kernel_export_every"}
kern = CRiMKernel(cams, kcfg, "cuda")
kern.load_state_dict({k: v.cuda() for k, v in ck["density_state"]["kernel"]["state_dict"].items()})
kern.step = 39999
kernel_mod.ACTIVE = kern
params = {k: torch.nn.Parameter(v) for k, v in ck["params"].items()}
deg = ck["sh_degree"]


def psnr(a, b):
    return float(-10 * torch.log10((a.clamp(0, 1) - b).square().mean()))


rows = []
for n in names:
    fr = on_device(ds["frames"][n], "cuda")
    with torch.enable_grad():
        out = renderer.render(params, fr, deg)
    gt = fr["image"]
    vm = out["subframe_viewmats"].detach()
    c2w = torch.linalg.inv(vm)
    centers = c2w[:, :3, 3]
    c0 = torch.linalg.inv(fr["viewmat"])[:3, 3]
    R0 = torch.linalg.inv(fr["viewmat"])[:3, :3]
    ang = [math.degrees(math.acos(max(-1.0, min(1.0, (float(torch.trace(R0.T @ c[:3, :3])) - 1) / 2))))
           for c in c2w]
    rows.append({"frame": n, "blur_psnr": psnr(out["rgb"].detach(), gt),
                 "sharp_vs_blurgt_psnr": psnr(out["sharp"].detach(), gt),
                 "mean_render_psnr": psnr(out["subframes"].detach().mean(0), gt),
                 "mask_mean": float(out["mask"].mean()),
                 "traj_len": float((centers[1:] - centers[:-1]).norm(dim=1).sum()),
                 "center_offset_tau0": float((centers[0] - c0).norm()),
                 "max_rot_deg": max(ang)})
    del out
summ = {k: sum(r[k] for r in rows) / len(rows) for k in rows[0] if k != "frame"}
centers = torch.stack([torch.linalg.inv(c["viewmat"])[:3, 3] for c in cams])
summ["scene_cam_spread"] = float((centers - centers.mean(0)).norm(dim=1).max())
res = {"run": str(run), "summary": summ, "per_view": rows}
print(json.dumps(summ, indent=1))
if len(sys.argv) > 2:
    Path(sys.argv[2]).write_text(json.dumps(res, indent=1))
