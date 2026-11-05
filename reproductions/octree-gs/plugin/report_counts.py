"""Octree-GS counts for a finished run: anchors per LOD level, mean rendered neural Gaussians per test view (the
paper's #GS(k): Gaussians with radii > 0 after LOD selection + visible_filter), mean selected/visible anchors, and
render FPS (secondary; shared GPU). Reads only the checkpoint (ss-eval path, no live bank).

Usage: tools/ssenv python reproductions/octree-gs/plugin/report_counts.py <run> [<run> ...]
       -> <run>/octree_gs_counts.json
"""

import json
import sys
import time
from pathlib import Path

import torch
from splatstudio.data import load_dataset, on_device

from ss_octree_gs import model
from ss_octree_gs.renderer import BLOB, OctreeGSRenderer


class _Spec:
    config = {"feat_dim": 32, "n_offsets": 10, "opacity_cap": 0.999}


@torch.no_grad()
def counts(run, device="cuda:0"):
    run = Path(run)
    config = json.loads((run / "config.json").read_text())
    ck = torch.load(run / "checkpoint.pt", map_location=device, weights_only=True)
    params = ck["params"]
    model.ACTIVE["bank"] = None
    renderer = OctreeGSRenderer(_Spec())
    _, oct_ = renderer.decoded(params)
    lev = params["ogs_level"][:, 0]
    per_level = [int((lev == i).sum()) for i in range(oct_["levels"])]
    dataset = load_dataset(config["data"], config["downscale"])
    rendered, selected, visible, times = [], [], [], []
    for name in dataset["splits"]["test"]:
        frame = on_device(dataset["frames"][name], device)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = renderer.render(params, frame)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
        radii = out["info"]["radii"][0]
        ok = (radii > 0).all(-1) if radii.dim() == 2 else radii > 0
        rendered.append(int(ok.sum()))
        selected.append(int(out["info"]["ogs"]["lod"].sum()))
        visible.append(int(out["info"]["ogs"]["visible"].sum()))
    n = len(rendered)
    res = {
        "run": str(run), "anchors": int(len(lev)), "anchors_per_level": per_level, "levels": oct_["levels"],
        "init_level": oct_["init_level"], "mean_extra_level": float(params["ogs_extra_level"].mean()),
        "rendered_gaussians_per_view_k": sum(rendered) / n / 1000, "lod_selected_anchors_per_view": sum(selected) / n,
        "visible_anchors_per_view": sum(visible) / n, "fps": n / sum(times[1:] or times) if n > 1 else None,
        "blob_rows": int(params[BLOB].shape[0]), "test_views": n,
    }
    (run / "octree_gs_counts.json").write_text(json.dumps(res, indent=2))
    return res


if __name__ == "__main__":
    for r in sys.argv[1:]:
        print(json.dumps(counts(r)))
