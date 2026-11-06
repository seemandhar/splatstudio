"""WildGaussians evaluation protocols (official train.py / datasets/phototourism.py,
NerfBaselines metrics).

* ``default`` (NeRF On-the-go): render every test view with the zero appearance embedding
  (official get_embedding(None)), metrics on the full image.
* ``nerfw`` (Photo Tourism, NeRF-W protocol): per test view the appearance embedding starts at
  zero and is optimised for 128 Adam steps (lr 0.1) on the **left** W - W//2 columns
  (horizontal_half_dataset(left=True): same intrinsics, cropped width) with
  0.8 L1 + 0.2 (1 - SSIM) of the toned render, the gradient masked by the eval-mode uncertainty
  mask of the left GT (appearance_optim_type "dssim+l1-scaled"); then the full view is rendered
  with that embedding and the metrics use columns [W//2:] (NerfWEvaluationProtocol.evaluate).
  For odd W one column is shared by optimisation and metrics (official); ``strict`` metrics use
  only columns [W - W//2:], which were never seen by the optimisation.

Metrics (NerfBaselines compute_metrics): images converted to uint8 by truncation
(convert_image_dtype), PSNR per image from the MSE, SSIM = dm_pix (valid separable Gaussian
11 / 1.5, clipped variances), LPIPS-Alex (paper) and LPIPS-VGG; the runtime's SSIM is also given.

    python -m ss_wildgaussians.evaluate --run RUN [--protocol auto|default|nerfw] [--split test]
Also scores runs of other methods (no bank, e.g. the gsplat control) with the same metrics.
writes RUN/<split>_wild.json.
"""

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from splatstudio.components import build
from splatstudio.data import load_dataset, on_device
from splatstudio.losses import ssim as ssim_runtime
from splatstudio.registry import load
from splatstudio.runtime import specification_id, write_json

from . import bank as bank_mod
from .bank import WildBank
from .losses import ssim_map
from .uncertainty import predict_sigma

OPT_DEFAULTS = {"iters": 128, "lr": 0.1, "lambda_dssim": 0.2}


def to_uint8_float(x):
    """convert_image_dtype(float -> uint8) (clip(x*255).astype(uint8): truncation) -> /255."""
    return torch.clamp(x * 255.0, 0, 255).floor() / 255.0


def dmpix_ssim(a, b, kernel_size=11, sigma=1.5, k1=0.01, k2=0.03):
    """NerfBaselines dmpix_ssim for HWC images in [0, 1] (valid separable Gaussian filter)."""
    hw = kernel_size // 2
    shift = (2 * hw - kernel_size + 1) / 2
    f = torch.exp(-0.5 * ((torch.arange(kernel_size, dtype=torch.float64) - hw + shift)
                          / sigma) ** 2)
    f = (f / f.sum()).to(a.device, torch.float32)
    x, y = (t.permute(2, 0, 1)[:, None] for t in (a, b))  # [3,1,H,W]

    def filt(z):
        z = F.conv2d(z, f.view(1, 1, 1, -1))
        return F.conv2d(z, f.view(1, 1, -1, 1))

    mu0, mu1 = filt(x), filt(y)
    s00 = filt(x * x) - mu0 * mu0
    s11 = filt(y * y) - mu1 * mu1
    s01 = filt(x * y) - mu0 * mu1
    eps = torch.finfo(torch.float32).eps ** 2
    s00, s11 = s00.clamp_min(eps), s11.clamp_min(eps)
    s01 = torch.sign(s01) * torch.minimum(torch.sqrt(s00 * s11), s01.abs())
    c1, c2 = k1 ** 2, k2 ** 2
    m = ((2 * mu0 * mu1 + c1) * (2 * s01 + c2)) / ((mu0 * mu0 + mu1 * mu1 + c1) * (s00 + s11 + c2))
    return float(m.mean())


_LP = {}


def lpips_net(net, device):
    if (net, device) not in _LP:
        import lpips

        _LP[(net, device)] = lpips.LPIPS(net=net, verbose=False).to(device).eval()
    return _LP[(net, device)]


@torch.no_grad()
def official_metrics(pred, gt, device):
    """compute_metrics(image_to_srgb(pred, uint8), image_to_srgb(gt, uint8)) on HWC tensors."""
    p, g = to_uint8_float(pred.clamp(0, 1)), to_uint8_float(gt.clamp(0, 1))
    mse = (p - g).square().mean()
    x, y = (t.permute(2, 0, 1)[None] * 2 - 1 for t in (p, g))
    return {"psnr": float(-10 * torch.log10(mse.clamp_min(1e-12))), "ssim": dmpix_ssim(p, g),
            "lpips": float(lpips_net("alex", device)(x, y)),
            "lpips_vgg": float(lpips_net("vgg", device)(x, y)),
            "ssim_runtime": float(ssim_runtime(p, g))}


def left_half_camera(frame):
    w = int(frame["width"])
    wl = w - w // 2
    cam = {k: v for k, v in frame.items() if k != "image"}
    cam["width"] = wl
    return cam, frame["image"][:, :wl]


def optimize_embedding(renderer, params, bank, frame, sh_degree, cfg=None, log=None):
    """WildGaussians.optimize_embedding on the left half of `frame`; returns (embedding, losses)."""
    cfg = {**OPT_DEFAULTS, **(cfg or {})}
    cam, gt = left_half_camera(frame)
    with torch.no_grad():
        mask = None
        if bank.uncertainty:
            sigma = predict_sigma(bank.head, bank.features, gt.permute(2, 0, 1)[None])
            mult = (1 / (2 * sigma.pow(2))).clamp_max(3.0)
            mask = (mult > 1).float()[0, 0][..., None]  # [H, W, 1]
    emb = torch.nn.Parameter(torch.zeros_like(bank.image_embeddings[0]))
    opt = torch.optim.Adam([emb], lr=cfg["lr"])
    bank.set_override(emb)
    losses = []
    try:
        with torch.enable_grad():
            for _ in range(int(cfg["iters"])):
                opt.zero_grad()
                img = renderer.render(params, cam, sh_degree)["rgb"]
                if mask is not None:  # scale_grads(image, mask)
                    img = img * mask + img.detach() * (1 - mask)
                lam = cfg["lambda_dssim"]
                loss = ((1 - lam) * (img - gt).abs().mean()
                        + lam * (1 - ssim_map(img.permute(2, 0, 1), gt.permute(2, 0, 1)).mean()))
                loss.backward()
                opt.step()
                losses.append(float(loss))
    finally:
        bank.set_override(None)
    return emb.detach(), losses


def evaluate(run, split="test", protocol="auto", device="cuda:0", opt_cfg=None):
    run = Path(run)
    config = json.loads((run / "config.json").read_text())
    method = load(config["method"])
    ckpt = torch.load(run / "checkpoint.pt", map_location=device, weights_only=True)
    if specification_id(method) != ckpt["specification_id"]:
        raise ValueError(f"{run}: installed method specification differs from the checkpoint")
    dataset = load_dataset(config["data"], config["downscale"])
    if dataset["protocol_id"] != ckpt["protocol_id"]:
        raise ValueError("Dataset or evaluation protocol changed since training")
    if protocol == "auto":
        protocol = dataset["meta"].get("wildgaussians", {}).get("evaluation_protocol", "default")
    renderer, _, _ = build(method)
    if bank_mod.ACTIVE is not None:
        raise RuntimeError("evaluation must not run with a live training bank")
    state = ckpt.get("density_state") or {}
    bank = None
    if isinstance(state, dict) and "bank" in state:
        bank = WildBank.from_export(state["bank"], device)
    # runs of other methods (e.g. the gsplat control) are scored with the same official metrics
    bank_mod.ACTIVE = bank
    params = {k: v.detach() for k, v in ckpt["params"].items()}
    degree = ckpt["sh_degree"]
    can_optimise = bank is not None and bank.appearance
    if protocol == "nerfw" and bank is not None and not bank.appearance:
        raise ValueError("nerfw protocol needs an appearance-enabled method")
    rows = []
    try:
        for name in dataset["splits"][split]:
            frame = on_device(dataset["frames"][name], device)
            gt, w = frame["image"], int(frame["width"])
            row = {"frame": name, "width": w}
            with torch.no_grad():
                nopt = renderer.render(params, frame, degree)["rgb"].clamp(0, 1).nan_to_num(0)
            if protocol == "nerfw" and not can_optimise:  # e.g. gsplat control: nothing to tune
                row["official"] = official_metrics(nopt[:, w // 2:], gt[:, w // 2:], device)
                row["strict"] = official_metrics(nopt[:, w - w // 2:], gt[:, w - w // 2:], device)
            elif protocol == "nerfw":
                emb, hist = optimize_embedding(renderer, params, bank, frame, degree, opt_cfg)
                bank.set_override(emb)
                with torch.no_grad():
                    pred = renderer.render(params, frame, degree)["rgb"].clamp(0, 1).nan_to_num(0)
                bank.set_override(None)
                row["official"] = official_metrics(pred[:, w // 2:], gt[:, w // 2:], device)
                row["strict"] = official_metrics(pred[:, w - w // 2:], gt[:, w - w // 2:], device)
                row["no_optimisation"] = official_metrics(nopt[:, w // 2:], gt[:, w // 2:],
                                                          device)
                row["optimisation_loss"] = [hist[0], hist[-1]]
            else:
                row["official"] = official_metrics(nopt, gt, device)
            rows.append(row)
            print(json.dumps({"frame": name, **row["official"]}), flush=True)
    finally:
        bank_mod.ACTIVE = None
    keys = [k for k in ("official", "strict", "no_optimisation") if k in rows[0]]
    summary = {k: {m: sum(r[k][m] for r in rows) / len(rows) for m in rows[0][k]} for k in keys}
    result = {"run": str(run.resolve()), "split": split, "protocol": protocol,
              "metrics": summary, "per_view": rows,
              "test_time_optimisation": {**OPT_DEFAULTS, **(opt_cfg or {})}
              if protocol == "nerfw" else None,
              "note": "HEADLINE = metrics.official (NerfBaselines conventions: uint8, dm_pix SSIM, "
                      "LPIPS-Alex). nerfw: 'strict' excludes the column shared with the "
                      "optimisation half for odd widths; 'no_optimisation' = zero embedding."}
    write_json(run / f"{split}_wild.json", result)
    return result


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run", required=True)
    ap.add_argument("--split", default="test", choices=["validation", "test"])
    ap.add_argument("--protocol", default="auto", choices=["auto", "default", "nerfw"])
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--iters", type=int, default=None)
    a = ap.parse_args(argv)
    r = evaluate(a.run, a.split, a.protocol, a.device,
                 {"iters": a.iters} if a.iters else None)
    print(json.dumps(r["metrics"], indent=1))


if __name__ == "__main__":
    main()
