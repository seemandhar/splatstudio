"""Evaluate one trained run at several target granularities (render_hierarchy.py --taus).

    tools/ssenv python -m ss_hierarchical_3dgs.eval_taus --run RUN [--taus -0.5 3 6 15]

Uses the same checks as `splatstudio.runtime.evaluate` (method specification, source
manifest, protocol) and the runtime's `measure`; only the renderer's `eval_tau` differs.
Writes RUN/test_tau<tau>.json and prints a summary. tau = -0.5 renders all leaves (threshold 0).

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


def basic_params(params, dataset, device):
    """Leaves of a hierarchy -> fresh creator run -> unoptimized hierarchy state."""
    nodes, _, anchors = hier.unpack(params)
    leaf = nodes[nodes[:, 0] == 0, 2].long()
    if not bool(anchors[leaf].all()):
        raise ValueError("leaves were not frozen in this run")
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


def evaluate_taus(run, taus, split="test", device="cuda", basic=False):
    run = Path(run)
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
        sizes = []
        orig = renderer.render

        def counted(params, camera, sh_degree=3, _orig=orig):
            o = _orig(params, camera, sh_degree)
            sizes.append(o["info"]["cut_size"])
            return o

        renderer.render = counted
        metrics, views = measure(renderer, params, dataset, split, ckpt["sh_degree"],
                                 device)
        result = {"method": method.name, "scene": dataset["root"].name, "split": split,
                  "tau": float(tau), "metrics": metrics, "per_view": views,
                  "mean_rendered_gaussians": sum(sizes) / len(sizes),
                  "total_gaussians": len(params["means"]), "variant": "basic" if basic else "trained",
                  "steps": ckpt["step"], "hierarchy": ckpt["density_state"].get("hierarchy"),
                  "paper_reproduction": "unverified"}
        write_json(run / f"{split}_{tag}tau{tau:g}.json", result)
        out[f"{tau:g}"] = {**metrics, "rendered": result["mean_rendered_gaussians"]}
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--taus", nargs="+", type=float, default=[-0.5, 3.0, 6.0, 15.0])
    ap.add_argument("--split", default="test", choices=["test", "validation"])
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--basic", action="store_true",
                    help="rebuild and evaluate the unoptimized hierarchy from the frozen leaves")
    a = ap.parse_args(argv)
    print(json.dumps(evaluate_taus(a.run, a.taus, a.split, a.device, a.basic), indent=1))


if __name__ == "__main__":
    main()
