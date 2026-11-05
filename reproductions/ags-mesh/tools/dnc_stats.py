"""Real-data check of the DNC masks (paper Fig. 6 claim: rejects edges and far regions)."""
import json
import sys
from pathlib import Path

import numpy as np

room = sys.argv[1] if len(sys.argv) > 1 else "vr_room"
r = Path.home() / "datasets/mushroom/native" / room
D = np.load(r / "dnsplat/depth_mm.npy", mmap_mode="r")
M = np.load(r / "agsmesh/dnc_mask_u8.npy", mmap_mode="r")
idx = np.linspace(0, len(D) - 1, 40).astype(int)
bins = [0, 1, 2, 3, 4, 100]
rej_bin = {f"{a}-{b}m": [] for a, b in zip(bins[:-1], bins[1:])}
edge, flat, tot = [], [], []
for i in idx:
    d = D[i].astype(np.float64) * 1e-3
    rej = M[i] == 255
    gy, gx = np.gradient(d)
    rel = np.hypot(gx, gy) / np.maximum(d, 1e-3)
    e = rel > 0.02
    edge.append(rej[e].mean() if e.any() else np.nan)
    flat.append(rej[rel < 0.002].mean())
    tot.append(rej.mean())
    for a, b in zip(bins[:-1], bins[1:]):
        s = (d > a) & (d <= b)
        if s.sum() > 100:
            rej_bin[f"{a}-{b}m"].append(rej[s].mean())
out = {"room": room, "frames": len(idx), "rejected": float(np.mean(tot)),
       "rejected_at_depth_edges": float(np.nanmean(edge)),
       "rejected_on_smooth_depth": float(np.mean(flat)),
       "rejected_by_depth": {k: (float(np.mean(v)) if v else None) for k, v in rej_bin.items()}}
print(json.dumps(out, indent=1))
