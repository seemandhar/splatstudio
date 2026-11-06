"""SLS-agg preprocessing (official SemanticParser(cluster=True), Eq. 3-4): per training image,
ward AgglomerativeClustering of the 1280 x 50 x 50 SD features with an 8-NN grid connectivity
into 100 clusters; labels saved as int16 [50, 50] to <scene>/SLS_clusters/<stem>.npy.

    PYTHONPATH=~/models/coherentgs/pylib tools/ssenv python -I make_clusters.py --scene NATIVE
(scikit-learn 1.7.2 there; the official requirements pin no sklearn version.)
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

for p in ("/home/sejain/models/coherentgs/pylib",):
    if p not in sys.path:
        sys.path.append(p)
from sklearn.cluster import AgglomerativeClustering  # noqa: E402
from sklearn.neighbors import kneighbors_graph  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--clusters", type=int, default=100)
    a = ap.parse_args()
    root = Path(a.scene)
    meta = json.loads((root / "transforms.json").read_text())
    out = root / "SLS_clusters"
    out.mkdir(exist_ok=True)
    x = np.linspace(0, 1, 50)
    xv, yv = np.meshgrid(x, x)
    knn = kneighbors_graph(np.reshape(np.stack([xv, yv], axis=-1), (2500, 2)), 8,
                           include_self=False)
    for f in meta["train_filenames"]:
        stem = Path(f).stem
        dst = out / f"{stem}.npy"
        if dst.exists():
            continue
        feat = np.load(root / "SD" / f"{stem}.npy")
        flat = np.transpose(feat.reshape((1280, 2500)), (1, 0))
        model = AgglomerativeClustering(linkage="ward", connectivity=knn,
                                        n_clusters=a.clusters).fit(flat)
        np.save(dst, model.labels_.reshape(50, 50).astype(np.int16))
    print(json.dumps({"scene": str(root), "done": len(list(out.glob("*.npy")))}))


if __name__ == "__main__":
    main()
