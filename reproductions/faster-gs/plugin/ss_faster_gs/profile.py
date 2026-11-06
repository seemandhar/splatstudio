"""Separate forward / backward / optimizer-update timing: Faster-GS vs the runtime gsplat path.

Each phase is timed between device synchronisations. Both paths start from identical copies of the same Gaussians, render the same training views on
the same GPU, use the same loss (runtime photometric, timed separately), and update all six
fields. Faster-GS = vendored rasterizer (densification statistics on, as while densifying) +
FusedAdam. gsplat = `splatstudio.backends.gsplat.Renderer` (means2d grad retained, as
DefaultStrategy does) + `torch.optim.Adam` (runtime default, foreach) and, for reference,
`torch.optim.Adam(fused=True)`.

    python -m ss_faster_gs.profile --run <run dir with checkpoint.pt> [--views 50] [--warmup 5]

writes `<run>/fastergs_profile.json`. GPUs here are shared: compare paths within one file only.
"""

import argparse
import json
import time
from pathlib import Path

import torch

from .optim import FusedAdam
from .raster import rasterize

RATES = {"means": 1.6e-5, "scales": 0.005, "quats": 0.001, "opacities": 0.025, "sh0": 0.0025,
         "shN": 0.000125}


def _clone(params):
    return {k: torch.nn.Parameter(v.detach().clone().float().contiguous()) for k, v in params.items()}


def _loss():
    from splatstudio.losses import create_photometric
    from splatstudio.specs import ComponentSpec

    return create_photometric(ComponentSpec("photometric", "loss", "x", (), (), {}))


def _timed(fn, times):
    """Wall time of one phase between device synchronisations (includes launch overhead)."""
    torch.cuda.synchronize()
    start = time.perf_counter()
    out = fn()
    torch.cuda.synchronize()
    times.append((time.perf_counter() - start) * 1000.0)
    return out


def _run_path(name, params, cameras, targets, sh_degree, iters, warmup, loss_fn):
    from splatstudio.backends.gsplat import Renderer
    from splatstudio.specs import ComponentSpec

    p = _clone(params)
    if name == "faster-gs":
        opts = {k: FusedAdam([v], lr=RATES[k], eps=1e-15) for k, v in p.items()}
        info = torch.zeros((2, len(p["means"])), device=p["means"].device)
        bg = torch.zeros(3, device=p["means"].device)

        def render(cam):
            return rasterize(p, cam, sh_degree, bg, info).permute(1, 2, 0)
    else:
        fused = name == "gsplat-fusedadam"
        opts = {k: torch.optim.Adam([v], lr=RATES[k], eps=1e-15, fused=fused or None)
                for k, v in p.items()}
        gs = Renderer(ComponentSpec("gsplat", "renderer", "x", (), (), {}))

        def render(cam):
            out = gs.render(p, cam, sh_degree)
            out["info"]["means2d"].retain_grad()
            return out["rgb"]

    timings = {"forward": [], "loss": [], "backward": [], "update": []}
    first = None
    for i in range(warmup + iters):
        cam, target = cameras[i % len(cameras)], targets[i % len(targets)]
        ev = {k: [] for k in timings}
        rgb = _timed(lambda: render(cam), ev["forward"])
        if first is None:
            first = rgb.detach().clamp(0, 1).clone()
        loss = _timed(lambda: loss_fn(p, {"rgb": rgb}, target), ev["loss"])
        _timed(loss.backward, ev["backward"])

        def update():
            for o in opts.values():
                o.step()
                o.zero_grad(set_to_none=True)

        _timed(update, ev["update"])
        if i >= warmup:
            for k in timings:
                timings[k].append(ev[k][0])
    res = {f"{k}_ms": sum(v) / len(v) for k, v in timings.items()}
    res["total_ms"] = sum(res[f"{k}_ms"] for k in timings)
    return res, first


def profile_params(params, cameras, targets, sh_degree=3, iters=50, warmup=5,
                   paths=("faster-gs", "gsplat", "gsplat-fusedadam")):
    loss_fn = _loss()
    out = {"gaussians": len(params["means"]), "iters": iters, "warmup": warmup,
           "sh_degree": sh_degree, "gpu": torch.cuda.get_device_name(), "images": {}}
    for name in paths:
        res, img = _run_path(name, params, cameras, targets, sh_degree, iters, warmup, loss_fn)
        out[name] = res
        out["images"][name] = img
    if "gsplat" in out and "faster-gs" in out:
        out["speedup_vs_gsplat"] = {
            k: out["gsplat"][k] / out["faster-gs"][k]
            for k in ("forward_ms", "backward_ms", "update_ms", "total_ms")}
    return out


def main(argv=None):
    from splatstudio.data import load_dataset, on_device

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", required=True)
    ap.add_argument("--views", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--output", default=None)
    args = ap.parse_args(argv)
    run = Path(args.run)
    config = json.loads((run / "config.json").read_text())
    ckpt = torch.load(run / "checkpoint.pt", map_location=args.device, weights_only=True)
    torch.cuda.set_device(args.device)
    dataset = load_dataset(config["data"], config["downscale"])
    names = dataset["splits"]["train"]
    gen = torch.Generator().manual_seed(0)
    pick = [names[int(i)] for i in torch.randint(len(names), (args.views,), generator=gen)]
    frames = [on_device(dataset["frames"][n], args.device) for n in pick]
    cams = [{k: f[k] for k in ("K", "viewmat", "width", "height")} for f in frames]
    res = profile_params(ckpt["params"], cams, [f["image"] for f in frames],
                         int(ckpt["sh_degree"]), iters=args.views, warmup=args.warmup)
    res.pop("images")
    res.update(run=str(run), method=config["method"], data=config["data"])
    path = Path(args.output) if args.output else run / "fastergs_profile.json"
    path.write_text(json.dumps(res, indent=2) + "\n")
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
