"""Splatfacto-W evaluation: NeRF-W right-half protocol.

Official code (``get_image_metrics_and_images`` with ``eval_right_half=True``): each test view is
rendered with its own embedding -- trained jointly on its left half (columns < W//2) because the
code's training split contains every image with the test images' right halves masked -- and
PSNR / SSIM / LPIPS (torchmetrics AlexNet) are computed on columns >= W//2. Training data for this:
``~/datasets/phototourism/sfw/<scene>`` (tools/prepare_sfw.py).

Test views without a trained embedding (a run on the plain NeRF-W split) fall back to the paper's
wording: optimise only an embedding (init: mean training embedding; Adam, ``--iters`` steps at
``--lr``) on the left half, then score the right half. This fallback is labelled in the output.

    python -m ss_splatfactow.evaluate --run RUN [--split test]
writes RUN/<split>_sfw.json (headline: metrics.nerfw).
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
from .bank import SfwBank

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


def right_half(img):
    return img[:, img.shape[1] // 2:]


def left_half(frame):
    w = int(frame["width"])
    cam = {k: v for k, v in frame.items() if k != "image"}
    cam["width"] = w // 2
    return cam, frame["image"][:, : w // 2]


def optimize_embedding(renderer, params, bank, frame, cfg=None):
    cfg = {**OPT_DEFAULTS, **(cfg or {})}
    cam, gt = left_half(frame)
    emb = torch.nn.Parameter(bank.appearance_embeds.weight.detach().mean(0).clone())
    opt = torch.optim.Adam([emb], lr=float(cfg["lr"]))
    lam = float(cfg["lambda_dssim"])
    losses = []
    bank.set_override(emb)
    try:
        with torch.enable_grad():
            for _ in range(int(cfg["iters"])):
                opt.zero_grad(set_to_none=True)
                img = renderer.render(params, cam)["rgb"]
                loss = (1 - lam) * (img - gt).abs().mean() + lam * (1 - ssim(img, gt))
                loss.backward()
                opt.step()
                losses.append(float(loss))
    finally:
        bank.set_override(None)
    return emb.detach(), losses


def evaluate(run, split="test", device="cuda:0", opt_cfg=None, save_images=False):
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
    bank = SfwBank.from_export(ckpt["density_state"]["bank"], device)
    for p in bank.parameters():
        p.requires_grad_(False)
    bank_mod.ACTIVE = bank
    params = {k: v.detach() for k, v in ckpt["params"].items()}
    rows = []
    try:
        for name in dataset["splits"][split]:
            frame = on_device(dataset["frames"][name], device)
            gt = frame["image"]
            idx = bank.view_index(frame["viewmat"])
            hist = None
            if idx is not None:
                mode = "code: jointly trained left-half embedding"
            else:
                mode = "paper text: embedding-only left-half optimisation"
                emb, hist = optimize_embedding(renderer, params, bank, frame, opt_cfg)
                bank.set_override(emb)
            with torch.no_grad():
                pred = renderer.render(params, frame)["rgb"].clamp(0, 1)
            bank.set_override(None)
            row = {"frame": name, "width": int(frame["width"]), "embedding": mode,
                   "view_index": idx,
                   "nerfw": metrics(right_half(pred), right_half(gt), device),
                   "full_image": metrics(pred, gt, device)}
            if hist:
                row["optimisation_loss"] = [hist[0], hist[-1]]
            if save_images:
                from PIL import Image

                folder = run / "images_sfw" / split
                folder.mkdir(parents=True, exist_ok=True)
                img = torch.cat([gt, pred], dim=1).cpu().numpy()
                Image.fromarray((img * 255).round().astype("uint8")).save(
                    folder / f"{len(rows):04d}.png")
            rows.append(row)
            print(json.dumps({"frame": name, **row["nerfw"]}), flush=True)
    finally:
        bank_mod.ACTIVE = None
    keys = ("nerfw", "full_image")
    summary = {k: {m: sum(r[k][m] for r in rows) / len(rows) for m in rows[0][k]} for k in keys}
    result = {"run": str(run.resolve()), "split": split, "metrics": summary, "per_view": rows,
              "embedding_modes": sorted({r["embedding"] for r in rows}),
              "test_time_optimisation": {**OPT_DEFAULTS, **(opt_cfg or {})},
              "note": "HEADLINE = metrics.nerfw (columns >= W//2); lpips = Alex (torchmetrics "
                      "default, as the official code), lpips_vgg also given."}
    write_json(run / f"{split}_sfw.json", result)
    return result


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run", required=True)
    ap.add_argument("--split", default="test", choices=["validation", "test"])
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--iters", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--save-images", action="store_true")
    a = ap.parse_args(argv)
    cfg = {k: v for k, v in (("iters", a.iters), ("lr", a.lr)) if v is not None}
    r = evaluate(a.run, a.split, a.device, cfg, a.save_images)
    print(json.dumps({"metrics": r["metrics"], "modes": r["embedding_modes"]}, indent=1))


if __name__ == "__main__":
    main()
