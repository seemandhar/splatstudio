"""Official SplatFields few-view train-view selection (scene/dataset_readers.py:679).

`sorted(kmeans_downsample(cam_pos, n_views))` with `KMeans(n, random_state=0).fit(points)` on
the 100 train camera centres in the scaled world. Run with scikit-learn 1.0.2 (the newest
release for the official python 3.7 env; later releases changed k-means++ seeding and the
n_init default, which changes the selected views):
    ~/models/venv-sklearn102/bin/python -I select_views.py --raw ~/datasets/nerf_synthetic/raw \
        --scenes chair hotdog ficus --n-views 6 12 > views.json
"""

import argparse
import json
from pathlib import Path

import numpy as np
import sklearn
from sklearn.cluster import KMeans

MODEL_SCALE = dict(chair=2.1, drums=2.3, ficus=2.3, hotdog=3.0, lego=2.4, materials=2.4,
                   mic=2.5, ship=2.75)
B2O = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]], dtype=np.float32)


def centres(scene_dir, ws):
    frames = json.loads((scene_dir / "transforms_train.json").read_text())["frames"]
    out = []
    for f in frames:
        tm = np.array(f["transform_matrix"]) @ B2O
        tm[:3, :4] *= ws
        out.append(tm[:3, 3])
    return np.stack(out, axis=0)


def kmeans_downsample(points, n):
    km = KMeans(n, random_state=0).fit(points)
    return ((points - km.cluster_centers_[..., None, :]) ** 2).sum(-1).argmin(-1).tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, required=True)
    ap.add_argument("--scenes", nargs="+", required=True)
    ap.add_argument("--n-views", nargs="+", type=int, required=True)
    a = ap.parse_args()
    res = {"sklearn": sklearn.__version__, "numpy": np.__version__, "views": {}}
    for s in a.scenes:
        pts = centres(a.raw.expanduser() / s, 2 / MODEL_SCALE[s])
        res["views"][s] = {str(n): sorted(int(i) for i in kmeans_downsample(pts, n))
                           for n in a.n_views}
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
