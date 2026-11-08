"""WE-GS evaluation (paper §5.3 / Ha-NeRF, CR-NeRF, GS-W protocol): every test image is encoded
feed-forward by the trained spatial-attention encoder (l_k and M_k from the test image itself, no
test-time optimisation) and the full image is scored. Also reported: the right half of the same
render (columns >= W//2, the NeRF-W column split, for comparison with SWAG/Splatfacto-W runs;
NOT the WE-GS protocol) and the mean predicted static mask.

    python -m ss_wegs.evaluate --run RUN [--split test]
writes RUN/<split>_wegs.json (HEADLINE = metrics.full). Metric code mirrors ss_swag.evaluate
(PSNR float, SplatStudio SSIM, LPIPS-Alex headline + LPIPS-VGG).
"""

import argparse
import json
import time
from pathlib import Path

import torch
from splatstudio.components import build
from splatstudio.data import load_dataset, on_device
from splatstudio.losses import ssim
from splatstudio.registry import load
from splatstudio.runtime import specification_id, write_json

from . import bank as bank_mod
from .bank import WEGSBank

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


@torch.no_grad()
def render_feedforward(renderer, params, frame, degree):
    """Encode the frame's own image (no optimisation) and render it."""
    out = renderer.render(params, frame, degree)
    return out["rgb"].clamp(0, 1), out


def evaluate(run, split="test", device="cuda:0"):
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
    bank = WEGSBank.from_export(ckpt["density_state"]["bank"], device)
    bank_mod.ACTIVE = bank
    params = {k: v.detach() for k, v in ckpt["params"].items()}
    degree = ckpt["sh_degree"]
    rows, times = [], []
    try:
        for name in dataset["splits"][split]:
            frame = on_device(dataset["frames"][name], device)
            gt, w = frame["image"], int(frame["width"])
            torch.cuda.synchronize(device)
            t0 = time.perf_counter()
            pred, out = render_feedforward(renderer, params, frame, degree)
            torch.cuda.synchronize(device)
            times.append(time.perf_counter() - t0)
            row = {"frame": name, "width": w, "height": int(frame["height"]),
                   "full": metrics(pred, gt, device),
                   "right_half_reference_not_nerfw": metrics(pred[:, w // 2:], gt[:, w // 2:], device),
                   "static_mask_mean": float(out["static_mask"].mean())}
            rows.append(row)
            print(json.dumps({"frame": name, **row["full"]}), flush=True)
    finally:
        bank_mod.ACTIVE = None
    summary = {k: {m: sum(r[k][m] for r in rows) / len(rows) for m in rows[0][k]}
               for k in ("full", "right_half_reference_not_nerfw")}
    result = {"run": str(run.resolve()), "split": split, "metrics": summary,
              "static_mask_mean": sum(r["static_mask_mean"] for r in rows) / len(rows),
              "gaussians": len(params["means"]),
              "seconds_per_view_incl_encoding": sum(times) / len(times),
              "per_view": rows,
              "note": "HEADLINE = metrics.full (feed-forward appearance from the test image, "
                      "full image; WE-GS/Ha-NeRF/GS-W protocol); lpips = Alex, lpips_vgg also "
                      "given; right_half_reference_not_nerfw scores columns >= W//2 of the same render whose "
                      "appearance was encoded from the FULL test image: NOT the NeRF-W protocol, "
                      "do not compare with SWAG/WildGaussians right-half numbers."}
    write_json(run / f"{split}_wegs.json", result)
    return result


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run", required=True)
    ap.add_argument("--split", default="test", choices=["validation", "test"])
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args(argv)
    r = evaluate(a.run, a.split, a.device)
    print(json.dumps({"metrics": r["metrics"], "static_mask_mean": r["static_mask_mean"]},
                     indent=1))


if __name__ == "__main__":
    main()
