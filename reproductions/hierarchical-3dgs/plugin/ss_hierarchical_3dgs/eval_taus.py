"""Evaluate one trained run at several target granularities (render_hierarchy.py --taus).

    tools/ssenv python -m ss_hierarchical_3dgs.eval_taus --run RUN [--taus -0.5 0 3 6 15]
    tools/ssenv python -m ss_hierarchical_3dgs.eval_taus --run BASELINE_RUN --channel-only \
        --output FILE      # per-channel PSNR of any run (e.g. the gsplat baseline), no LPIPS

Uses the same checks as `splatstudio.runtime.evaluate` (method specification, source
manifest, protocol) and the runtime's `measure`; only the renderer's `eval_tau` differs.
Writes RUN/test_tau<tau>.json and prints a summary. tau = -0.5 renders all leaves (threshold 0,
the paper's "leaves" row); tau = 0 is the official render_hierarchy.py finest row (threshold of
1 px, render_hierarchy.py:57/129). Every per_view row also carries `psnr_channel`, the official
utils/image_utils.psnr on [3,H,W] (per-channel PSNR, then the mean over channels), and
`metrics.psnr_channel` its view mean; `metrics.psnr` stays the runtime's full-image PSNR.

--basic: rebuild the *unoptimized* hierarchy ("Ours (tau)") from the leaves of an optimized
run and evaluate it (writes RUN/test_basic_tau<tau>.json). During hierarchy optimization the
leaves are anchors and receive no update (probe h3dgs-hierarchy-optimization), so they are the
trained leaves of step `leaves_steps` up to the half-precision storage of the .hier file; the
creator is deterministic given leaves and training cameras.
"""

import argparse
import json
from dataclasses import replace
from pathlib import Path

import torch
from splatstudio.data import load_dataset, on_device
from splatstudio.provenance import source_manifest
from splatstudio.registry import load, resolve
from splatstudio.runtime import measure, specification_id, write_json

from . import hier


def channel_psnr(rgb, target):
    """utils/image_utils.psnr(img [3,H,W], gt).mean() of render_hierarchy.py:111 (HWC inputs)."""
    x = rgb.clamp(0, 1).permute(2, 0, 1).reshape(3, -1)
    y = target.clamp(0, 1).permute(2, 0, 1).reshape(3, -1)
    mse = (x - y).square().mean(1).clamp_min(1e-12)
    return float((20 * torch.log10(1.0 / mse.sqrt())).mean())


def basic_params(params, dataset, device):
    """Leaves of a hierarchy -> fresh creator run -> unoptimized hierarchy state.

    Guard: every leaf must be an anchor of the optimized hierarchy (anchors are frozen by
    train_post), otherwise the leaves are not the step-`leaves_steps` leaves and the rebuilt
    hierarchy would not be the basic one."""
    nodes, _, anchors = hier.unpack(params)
    leaf = nodes[nodes[:, 0] == 0, 2].long()
    not_anchor = int((~anchors[leaf]).sum())
    if not_anchor:
        raise ValueError(f"{not_anchor} of {len(leaf)} leaves are not anchors (not frozen during "
                         "hierarchy optimization); cannot rebuild the basic hierarchy")
    eps = 1e-6
    leaves = {"means": params["means"][leaf], "scales": params["scales"][leaf],
              "quats": params["quats"][leaf],
              "opacities": torch.logit(params["opacities"][leaf].abs().clamp(eps, 1 - eps)),
              "sh0": params["sh0"][leaf], "shN": params["shN"][leaf]}
    cams = [{k: on_device(dataset["frames"][n], device)[k] for k in ("K", "viewmat", "width",
                                                                      "height")}
            for n in dataset["splits"]["train"]]
    h = hier.build(leaves, cams)
    n = len(h["means"])
    return {"means": h["means"], "scales": h["scales"], "quats": h["rots"],
            "opacities": h["alpha"][:, 0], "sh0": h["shs"][:, :1], "shN": h["shs"][:, 1:],
            hier.FIELD: hier.pack(h["nodes"], h["boxes"], h["anchors"], n)}


def _load(run, device):
    """Same checks as splatstudio.runtime.evaluate."""
    config = json.loads((run / "config.json").read_text())
    method = load(config["method"])
    ckpt = torch.load(run / "checkpoint.pt", map_location=device, weights_only=True)
    if specification_id(method) != ckpt["specification_id"]:
        raise ValueError("Installed method specification differs from the checkpoint")
    if source_manifest(method) != config["source_manifest"]:
        raise ValueError("SDK, plugin, or renderer source changed since training")
    dataset = load_dataset(config["data"], config["downscale"])
    if dataset["protocol_id"] != ckpt["protocol_id"]:
        raise ValueError("Dataset or evaluation protocol changed since training")
    rspec = next(c for c in method.components if c.kind == "renderer")
    return method, ckpt, dataset, rspec


@torch.no_grad()
def channel_only(run, output, split="test", device="cuda"):
    """Full-image and per-channel PSNR of any run's checkpoint (e.g. the gsplat baseline)."""
    run = Path(run)
    method, ckpt, dataset, rspec = _load(run, device)
    renderer = resolve(rspec.factory)(rspec)
    params = {k: v.to(device) for k, v in ckpt["params"].items()}
    views = []
    for name in dataset["splits"][split]:
        frame = on_device(dataset["frames"][name], device)
        rgb = renderer.render(params, frame, sh_degree=ckpt["sh_degree"])["rgb"].clamp(0, 1)
        mse = (rgb - frame["image"]).square().mean().clamp_min(1e-12)
        views.append({"frame": name, "psnr": float(-10 * torch.log10(mse)),
                      "psnr_channel": channel_psnr(rgb, frame["image"])})
    result = {"method": method.name, "run": str(run), "scene": dataset["root"].name,
              "split": split, "steps": ckpt["step"],
              "metrics": {k: sum(v[k] for v in views) / len(views)
                          for k in ("psnr", "psnr_channel")},
              "per_view": views}
    write_json(Path(output), result)
    return result["metrics"]


def evaluate_taus(run, taus, split="test", device="cuda", basic=False):
    run = Path(run)
    method, ckpt, dataset, rspec = _load(run, device)
    params = ckpt["params"]
    tag = ""
    if basic:
        params = {k: v.to(device).contiguous() for k, v in
                  basic_params(params, dataset, device).items()}
        tag = "basic_"
    out = {}
    for tau in taus:
        spec = replace(rspec, config={**rspec.config, "eval_tau": float(tau)})
        renderer = resolve(spec.factory)(spec)
        sizes, chan = [], []
        orig = renderer.render

        def counted(params, camera, sh_degree=3, _orig=orig):
            o = _orig(params, camera, sh_degree)
            sizes.append(o["info"]["cut_size"])
            chan.append(channel_psnr(o["rgb"].detach(), camera["image"]))
            return o

        renderer.render = counted
        metrics, views = measure(renderer, params, dataset, split, ckpt["sh_degree"],
                                 device)
        for v, c in zip(views, chan, strict=True):
            v["psnr_channel"] = c
        metrics["psnr_channel"] = sum(chan) / len(chan)
        result = {"method": method.name, "scene": dataset["root"].name, "split": split,
                  "tau": float(tau), "metrics": metrics, "per_view": views,
                  "mean_rendered_gaussians": sum(sizes) / len(sizes),
                  "total_gaussians": len(params["means"]), "variant": "basic" if basic else "trained",
                  "steps": ckpt["step"], "hierarchy": ckpt["density_state"].get("hierarchy"),
                  "extent": ckpt["density_state"].get("extent"),
                  "paper_reproduction": "unverified"}
        write_json(run / f"{split}_{tag}tau{tau:g}.json", result)
        out[f"{tau:g}"] = {**metrics, "rendered": result["mean_rendered_gaussians"]}
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--taus", nargs="+", type=float, default=[-0.5, 0.0, 3.0, 6.0, 15.0])
    ap.add_argument("--split", default="test", choices=["test", "validation"])
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--basic", action="store_true",
                    help="rebuild and evaluate the unoptimized hierarchy from the frozen leaves")
    ap.add_argument("--channel-only", action="store_true",
                    help="any method's run: full-image + per-channel PSNR only, to --output")
    ap.add_argument("--output", help="output JSON for --channel-only")
    a = ap.parse_args(argv)
    if a.channel_only:
        if not a.output:
            ap.error("--channel-only needs --output")
        print(json.dumps(channel_only(a.run, a.output, a.split, a.device), indent=1))
        return
    print(json.dumps(evaluate_taus(a.run, a.taus, a.split, a.device, a.basic), indent=1))


if __name__ == "__main__":
    main()
