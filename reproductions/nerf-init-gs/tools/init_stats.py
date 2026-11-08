"""Init point-cloud diagnostics: distance to nearest training camera, vs the SfM cloud."""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from splatstudio.data import load_dataset

for root in sys.argv[1:]:
    ds = load_dataset(root)
    c = torch.stack([torch.linalg.inv(ds["frames"][n]["viewmat"])[:3, 3]
                     for n in ds["splits"]["train"]]).cuda()
    p = torch.from_numpy(np.load(Path(root) / "points.npz")["points"]).cuda()
    d = torch.cat([torch.cdist(p[i:i + 65536], c).min(1).values for i in range(0, len(p), 65536)])
    cc = c.mean(0)
    r = (c - cc).norm(dim=1).max()
    q = torch.tensor([0.01, 0.05, 0.5], device="cuda")
    print(json.dumps({"root": root, "n": len(p),
                      "dist_nearest_cam_q01_q05_q50": [round(float(x), 3) for x in d.quantile(q)],
                      "frac_within_0.5": float((d < 0.5).float().mean()),
                      "frac_within_1.0": float((d < 1.0).float().mean()),
                      "frac_inside_camera_sphere": float(((p - cc).norm(dim=1) < r).float().mean()),
                      "camera_radius": float(r)}))
