"""SWAG evaluation: NeRF-W protocol (paper §5.3, "we optimize an embedding on the left half of
each test image and report metrics on the right half").

Unspecified details (plan.md A7/A8/A10): the test embedding starts at the mean training
embedding and only it is optimised, 128 Adam steps at lr 0.1, loss 0.8 L1 + 0.2 D-SSIM of the
eval-mode (U = 0.5) render on the left W - W//2 columns; metrics on columns [W//2:]: PSNR (float),
SplatStudio SSIM, LPIPS-Alex and LPIPS-VGG. Also: metrics without optimisation (mean embedding,
same right half) and the C3 transient-Gaussian statistics over the training images.

    python -m ss_swag.evaluate --run RUN [--split test] [--iters 128] [--lr 0.1]
writes RUN/<split>_swag.json.
"""

import argparse
import json
from pathlib import Path

import torch
from splatstudio.components import build
from splatstudio.data import load_dataset, on_device
from splatstudio.losses import ssim
from splatstudio.registry import load
from splatstudio.runtime import specification_id, write_json

from . import bank as bank_mod
from .bank import SwagBank
from .density import transient_variance

OPT_DEFAULTS = {"iters": 128, "lr": 0.1, "lambda_dssim": 0.2}
_LP = {}


def lpips_net(net, device):
    if (net, device) not in _LP:
        import lpips

        _LP[(net, device)] = lpips.LPIPS(net=net, verbose=False).to(device).eval()
    return _LP[(net, device)]


@torch.no_grad()
def metrics(pred, gt, device):
    mse = (pred - gt).square().mean().clamp_min(1e-12)
    x, y = (t.permute(2, 0, 1)[None] * 2 - 1 for t in (pred, gt))
    return {"psnr": float(-10 * torch.log10(mse)), "ssim": float(ssim(pred, gt)),
            "lpips": float(lpips_net("alex", device)(x, y)),
            "lpips_vgg": float(lpips_net("vgg", device)(x, y))}


def left_half(frame):
    w = int(frame["width"])
    cam = {k: v for k, v in frame.items() if k != "image"}
    cam["width"] = w // 2
    return cam, frame["image"][:, : w // 2]


def optimize_embedding(renderer, params, bank, frame, degree, cfg=None):
    cfg = {**OPT_DEFAULTS, **(cfg or {})}
    cam, gt = left_half(frame)
    emb = torch.nn.Parameter(bank.mean_embedding().clone())
    opt = torch.optim.Adam([emb], lr=float(cfg["lr"]))
    lam = float(cfg["lambda_dssim"])
    losses = []
    was_training = bank.training
    bank.eval()  # U = 0.5, no training-view lookup
    bank.set_override(emb)
    try:
        with torch.enable_grad():
            for _ in range(int(cfg["iters"])):
                opt.zero_grad(set_to_none=True)
                img = renderer.render(params, cam, degree)["rgb"]
                loss = (1 - lam) * (img - gt).abs().mean() + lam * (1 - ssim(img, gt))
                loss.backward()
                opt.step()
                losses.append(float(loss))
    finally:
        bank.set_override(None)
        bank.train(was_training)
    return emb.detach(), losses


def evaluate(run, split="test", device="cuda:0", opt_cfg=None, transient_stats=True):
    run = Path(run)
    config = json.loads((run / "config.json").read_text())
    method = load(config["method"])
    ckpt = torch.load(run / "checkpoint.pt", map_location=device, weights_only=True)
    if specification_id(method) != ckpt["specification_id"]:
        raise ValueError(f"{run}: installed method specification differs from the checkpoint")
    dataset = load_dataset(config["data"], config["downscale"])
    if dataset["protocol_id"] != ckpt["protocol_id"]:
        raise ValueError("Dataset or evaluation protocol changed since training")
    renderer, _, _ = build(method)
    if bank_mod.ACTIVE is not None:
        raise RuntimeError("evaluation must not run with a live training bank")
    bank = SwagBank.from_export(ckpt["density_state"]["bank"], device)
    for p in bank.parameters():
        p.requires_grad_(False)
    bank_mod.ACTIVE = bank
    params = {k: v.detach() for k, v in ckpt["params"].items()}
    degree = ckpt["sh_degree"]
    rows = []
    try:
        for name in dataset["splits"][split]:
            frame = on_device(dataset["frames"][name], device)
            gt, w = frame["image"], int(frame["width"])
            with torch.no_grad():
                nopt = renderer.render(params, frame, degree)["rgb"].clamp(0, 1)
            emb, hist = optimize_embedding(renderer, params, bank, frame, degree, opt_cfg)
            bank.set_override(emb)
            with torch.no_grad():
                pred = renderer.render(params, frame, degree)["rgb"].clamp(0, 1)
            bank.set_override(None)
            row = {"frame": name, "width": w,
                   "nerfw": metrics(pred[:, w // 2:], gt[:, w // 2:], device),
                   "no_optimisation": metrics(nopt[:, w // 2:], gt[:, w // 2:], device),
                   "full_image_optimised": metrics(pred, gt, device),
                   "optimisation_loss": [hist[0], hist[-1]]}
            rows.append(row)
            print(json.dumps({"frame": name, **row["nerfw"]}), flush=True)
        trans = None
        if transient_stats and renderer.cfg["transient"]:
            cams = [{k: on_device(dataset["frames"][n], device)[k]
                     for k in ("K", "viewmat", "width", "height")}
                    for n in dataset["splits"]["train"]]
            order = [bank.view_index(c["viewmat"]) for c in cams]
            cams = [cams[order.index(i)] for i in range(len(cams))]
            var = transient_variance(renderer, params, bank, cams, degree)
            trans = {"gaussians": len(var),
                     **{f"var_gt_{t:g}": int((var > t).sum()) for t in (1e-8, 1e-6, 1e-4, 1e-2)}}
    finally:
        bank_mod.ACTIVE = None
    keys = ("nerfw", "no_optimisation", "full_image_optimised")
    summary = {k: {m: sum(r[k][m] for r in rows) / len(rows) for m in rows[0][k]} for k in keys}
    result = {"run": str(run.resolve()), "split": split, "metrics": summary,
              "per_view": rows, "transient": trans,
              "test_time_optimisation": {**OPT_DEFAULTS, **(opt_cfg or {})},
              "note": "HEADLINE = metrics.nerfw (right half after left-half embedding "
                      "optimisation); lpips = Alex, lpips_vgg also given."}
    write_json(run / f"{split}_swag.json", result)
    return result


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run", required=True)
    ap.add_argument("--split", default="test", choices=["validation", "test"])
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--iters", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--no-transient-stats", action="store_true")
    a = ap.parse_args(argv)
    cfg = {k: v for k, v in (("iters", a.iters), ("lr", a.lr)) if v is not None}
    r = evaluate(a.run, a.split, a.device, cfg, not a.no_transient_stats)
    print(json.dumps({"metrics": r["metrics"], "transient": r["transient"]}, indent=1))


if __name__ == "__main__":
    main()
