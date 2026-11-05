"""Masked evaluation of the CoherentGS paper (Sec. 5.1): pixels whose rendered opacity is below
1e-3 are "occluded" and masked out; the paper uses this same mask for every method.

The paper does not give the fill convention; the official loss_utils.ssim(mask=) fills masked
pixels with 1 (white) in both images, which is used here for PSNR / SSIM / LPIPS
("masked_white"). PSNR over the valid pixels only is reported as well ("psnr_valid").
Unmasked metrics are identical to ss-eval's. Usage:

    python -m ss_coherentgs.evaluate --run RUN [--split test] [--device cuda:0]
        [--mask-run OTHER_RUN]   # use OTHER_RUN's CoherentGS alpha as the mask (other methods)
writes RUN/<split>_masked.json.
"""

import argparse
import json
from pathlib import Path

import torch
from splatstudio.components import build
from splatstudio.data import load_dataset, on_device
from splatstudio.losses import ssim
from splatstudio.registry import load
from splatstudio.runtime import lpips_metric, specification_id, write_json


def _load(run, device):
    run = Path(run)
    config = json.loads((run / "config.json").read_text())
    method = load(config["method"])
    ckpt = torch.load(run / "checkpoint.pt", map_location=device, weights_only=True)
    if specification_id(method) != ckpt["specification_id"]:
        raise ValueError(f"{run}: installed method specification differs from the checkpoint")
    renderer, _, _ = build(method)
    return config, ckpt, renderer


def metrics(rgb, gt, perceptual):
    mse = (rgb - gt).square().mean().clamp_min(1e-12)
    row = {"psnr": float(-10 * torch.log10(mse)), "ssim": float(ssim(rgb, gt))}
    if perceptual is not None:
        x, y = (t.permute(2, 0, 1)[None] * 2 - 1 for t in (rgb, gt))
        row["lpips"] = float(perceptual(x, y))
    return row


@torch.no_grad()
def evaluate(run, split="test", device="cuda:0", mask_run=None, threshold=1e-3):
    config, ckpt, renderer = _load(run, device)
    dataset = load_dataset(config["data"], config["downscale"])
    if dataset["protocol_id"] != ckpt["protocol_id"]:
        raise ValueError("Dataset or evaluation protocol changed since training")
    mask_src = _load(mask_run, device) if mask_run else None
    perceptual = lpips_metric(device)
    rows = []
    for name in dataset["splits"][split]:
        frame = on_device(dataset["frames"][name], device)
        out = renderer.render(ckpt["params"], frame, sh_degree=ckpt["sh_degree"])
        rgb, gt = out["rgb"].clamp(0, 1), frame["image"]
        if mask_src is None:
            alpha = out["alpha"]
        else:
            _, mck, mren = mask_src
            alpha = mren.render(mck["params"], frame, sh_degree=mck["sh_degree"])["alpha"]
        valid = (alpha >= threshold).float()  # [H, W, 1]
        white_r = rgb * valid + (1 - valid)
        white_g = gt * valid + (1 - valid)
        row = {"frame": name, "valid_fraction": float(valid.mean()),
               "unmasked": metrics(rgb, gt, perceptual),
               "masked_white": metrics(white_r, white_g, perceptual)}
        se = ((rgb - gt).square() * valid).sum() / (3 * valid.sum()).clamp_min(1)
        row["masked_white"]["psnr_valid"] = float(-10 * torch.log10(se.clamp_min(1e-12)))
        rows.append(row)
    summary = {}
    for kind in ("unmasked", "masked_white"):
        keys = rows[0][kind].keys()
        summary[kind] = {k: sum(r[kind][k] for r in rows) / len(rows) for k in keys}
    summary["valid_fraction"] = sum(r["valid_fraction"] for r in rows) / len(rows)
    result = {"run": str(Path(run).resolve()), "split": split, "threshold": threshold,
              "mask_run": str(Path(mask_run).resolve()) if mask_run else None,
              "metrics": summary, "per_view": rows,
              "note": "masked_white: alpha < 1e-3 pixels set to white in render and GT "
                      "(official loss_utils.ssim(mask=) convention); paper Sec. 5.1"}
    write_json(Path(run) / f"{split}_masked.json", result)
    return result


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run", required=True)
    ap.add_argument("--split", default="test", choices=["validation", "test"])
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--mask-run")
    a = ap.parse_args(argv)
    r = evaluate(a.run, a.split, a.device, a.mask_run)
    print(json.dumps(r["metrics"], indent=1))


if __name__ == "__main__":
    main()
