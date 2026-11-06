"""Official BAGS / Deblur-NeRF evaluation (source/code render.py + metrics.py -t deblur-nerf).

render.py renders the final model's sharp test views (no BPN) and saves them with
torchvision.save_image (8-bit: (x*255 + 0.5).clamp(0, 255)); metrics.py maps images to [-1, 1]
and computes skimage 0.19 PSNR (data_range 2 -> identical to [0,1] PSNR), skimage SSIM on the
[-1, 1] images with no data_range (float -> dtype range span = 2, so C1 = (0.01*2)^2,
C2 = (0.03*2)^2; 7x7 uniform window, sample covariance, multichannel mean over the valid
region) and lpips.LPIPS() (AlexNet) on [-1, 1]. The luminance term (2 ux uy + C1)/(ux^2+uy^2+C1)
is NOT shift-invariant, so this differs from SSIM on [0, 1] with data_range 1 (review-1).
skimage is not in the env: its SSIM is re-implemented below (the border crop of skimage makes
its 'reflect' filtering equivalent to a valid convolution).
The runtime ss-eval conventions (float PSNR, Gaussian valid SSIM, LPIPS-VGG) are reported too.

    python -m ss_bags.evaluate --run RUN [--split test] [--device cuda:0]
writes RUN/<split>_official.json.
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

_LP = {}


def lpips_net(net, device):
    key = (net, device)
    if key not in _LP:
        import lpips

        _LP[key] = lpips.LPIPS(net=net, verbose=False).to(device).eval()
    return _LP[key]


def quantize(img):
    """torchvision.utils.save_image -> PNG -> to_tensor round trip."""
    return (img * 255 + 0.5).clamp(0, 255).to(torch.uint8).float() / 255.0


def ssim_skimage(a, b, win=7, data_range=1.0):
    """skimage.metrics.structural_similarity(multichannel=True) on HWC arrays as given, with an
    explicit data_range (C1 = (0.01 R)^2, C2 = (0.03 R)^2)."""
    x, y = (t.permute(2, 0, 1)[None].double() for t in (a, b))
    n = win * win
    cov_norm = n / (n - 1)

    def f(t):
        return F.avg_pool2d(t, win, stride=1)

    ux, uy = f(x), f(y)
    vx = cov_norm * (f(x * x) - ux * ux)
    vy = cov_norm * (f(y * y) - uy * uy)
    vxy = cov_norm * (f(x * y) - ux * uy)
    c1, c2 = (0.01 * data_range) ** 2, (0.03 * data_range) ** 2
    s = ((2 * ux * uy + c1) * (2 * vxy + c2)) / ((ux * ux + uy * uy + c1) * (vx + vy + c2))
    return float(s.mean())


def ssim_official(a, b):
    """metrics.py -t deblur-nerf: images in [0, 1] mapped to [-1, 1], skimage data_range 2."""
    return ssim_skimage(a * 2 - 1, b * 2 - 1, data_range=2.0)


@torch.no_grad()
def metrics(rgb, gt, device):
    q = quantize(rgb.clamp(0, 1))
    mse = (q - gt).square().mean().clamp_min(1e-12)
    xa, ya = (t.permute(2, 0, 1)[None] * 2 - 1 for t in (q, gt))
    mse_f = (rgb.clamp(0, 1) - gt).square().mean().clamp_min(1e-12)
    xf = rgb.clamp(0, 1).permute(2, 0, 1)[None] * 2 - 1
    return {"psnr": float(-10 * torch.log10(mse)), "ssim": ssim_official(q, gt),
            "lpips_alex": float(lpips_net("alex", device)(xa, ya)),
            "runtime_psnr": float(-10 * torch.log10(mse_f)),
            "runtime_ssim": float(ssim_runtime(rgb.clamp(0, 1), gt)),
            "runtime_lpips_vgg": float(lpips_net("vgg", device)(xf, ya))}


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
    if bank_mod.ACTIVE is not None:
        raise RuntimeError("evaluation must not run with a live training bank")
    renderer, _, _ = build(method)
    params, degree = ckpt["params"], ckpt["sh_degree"]
    rows = []
    with torch.no_grad():
        for name in dataset["splits"][split]:
            frame = on_device(dataset["frames"][name], device)
            rgb = renderer.render(params, frame, degree)["rgb"]
            rows.append({"frame": name, **metrics(rgb, frame["image"], device)})
    keys = [k for k in rows[0] if k != "frame"]
    summary = {k: sum(r[k] for r in rows) / len(rows) for k in keys}
    result = {"run": str(run.resolve()), "split": split, "metrics": summary, "per_view": rows,
              "note": "psnr / ssim / lpips_alex = official metrics.py -t deblur-nerf on 8-bit "
                      "renders (headline); runtime_* = ss-eval conventions"}
    write_json(run / f"{split}_official.json", result)
    return result


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run", required=True)
    ap.add_argument("--split", default="test", choices=["validation", "test"])
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args(argv)
    print(json.dumps(evaluate(a.run, a.split, a.device)["metrics"], indent=1))


if __name__ == "__main__":
    main()
