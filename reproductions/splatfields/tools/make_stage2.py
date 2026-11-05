"""SplatFields stage-2 scene: the stage-1 (`splatfields-3dgs`) cloud as initialisation.

Official run_blender.sh trains SplatFields with `--pts_samples load --max_num_pts 100000
--pc_path <3DGS run>/point_cloud/iteration_40000/point_cloud.ply`; readNerfSyntheticCVInfo
then keeps the points whose official projection lands on mask > 0 in every train view and
draws a random subset of at most max_num_pts (colours U[0,1)/255, unused by SplatFields).

This script reads the stage-1 checkpoint means, applies the same filter (functions from
prepare_blender_fewview.py), and writes <out>/<scene> with hard links to the stage-1 scene's
images / transforms.json / masks.npz plus the new points.npz and prep.json.

    tools/ssenv python -I reproductions/splatfields/tools/make_stage2.py \
        --stage1-data ~/datasets/nerf_synthetic/fewview6/chair --run runs/<exp>/<run> \
        --raw ~/datasets/nerf_synthetic/raw --out ~/datasets/nerf_synthetic/fewview6-sf
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import prepare_blender_fewview as prep  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage1-data", type=Path, required=True)
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--raw", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max-num-pts", type=int, default=100_000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    src = a.stage1_data.expanduser().resolve()
    info = json.loads((src / "prep.json").read_text())
    scene = info["scene"]
    status = json.loads((a.run / "status.json").read_text())
    if status.get("status") != "completed":
        raise SystemExit(f"stage-1 run not completed: {status}")
    config = json.loads((a.run / "config.json").read_text())
    if Path(config["data"]).resolve() != src:
        raise SystemExit(f"stage-1 run trained on {config['data']}, not {src}")
    ckpt = torch.load(a.run / "checkpoint.pt", map_location="cpu", weights_only=True)
    xyz = ckpt["params"]["means"].detach().double().numpy()

    raw = a.raw.expanduser() / scene
    train = prep.read_split(raw, "train", info["world_scale"])
    chosen = [train[i] for i in info["train_indices"]]
    masks, H, W = [], None, None
    for cam in chosen:
        _, _, a01 = prep.composite_8bit(raw / (cam["file"] + ".png"))
        H, W = a01.shape[:2]
        masks.append(a01)
    keep = prep.official_mask_test(xyz, chosen, masks, H, W)
    pts = xyz[keep]
    rng = np.random.RandomState(a.seed)
    n_hull = len(pts)
    if n_hull > a.max_num_pts:
        pts = pts[rng.choice(n_hull, a.max_num_pts, replace=False)]
    colors = rng.random_sample((len(pts), 3)) / 255.0

    target = a.out.expanduser() / scene
    (target / "images").mkdir(parents=True, exist_ok=True)
    for name in ("transforms.json", "masks.npz"):
        if not (target / name).exists():
            os.link(src / name, target / name)
    for img in sorted((src / "images").iterdir()):
        if not (target / "images" / img.name).exists():
            os.link(img, target / "images" / img.name)
    np.savez(target / "points.npz", points=pts.astype(np.float32),
             colors=colors.astype(np.float32))
    out = {**info, "stage": 2, "stage1_run": str(a.run.resolve()),
           "stage1_gaussians": int(len(xyz)), "stage1_in_hull": int(n_hull),
           "init_points": int(len(pts)), "seed": a.seed}
    (target / "prep.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out))


if __name__ == "__main__":
    main()
