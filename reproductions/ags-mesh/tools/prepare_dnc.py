"""AGS-Mesh Depth-Normal-Consistency (DNC) masks for a native MuSHRoom scene (paper §4.1, Eq. 1-2).

Replicates the mask the official MuSHRoom data path builds when ``ns-train ags-mesh ... mushroom
--load-depth-confidence-masks True`` runs (github.com/maturk/dn-splatter @ 97588b4,
data/mushroom_dataparser.py:180-192 -> scripts/depth_to_normal.py:DepthToNormal.main):

* sensor depth (uint16 mm x 1e-3; nearest resize to image size = no-op for MuSHRoom) is
  back-projected with pixel centres (+0.5) into OpenCV world coordinates;
* Open3D ``estimate_normals(KDTreeSearchParamKNN(knn=200))`` on that per-frame cloud (N_d, the
  covariance eigenvector of the 200 nearest points), flipped against the camera ray;
* the Omnidata png (N_p, camera frame, no sign conversion) is rotated to world with R_c2w;
* theta = angle between the [0,1]-ENCODED vectors (N_d+1)/2 and (N_p+1)/2 (official
  DepthToNormal compares the encodings; ``--angle-space signed`` gives the paper's Eq. 2 between
  the unit normals, as in scripts/depth_normal_consistency.py);
* mask = theta > tau_d (10 deg, paper supp. §1 and DepthToNormal), written by cv2.imwrite as a
  JPEG (``frame_xxxxx.jpg``) and read back with IMREAD_ANYDEPTH: the plugin's confidence is
  ``1 - m/255 > 0`` exactly like dn_model.py:646 + regularization_strategy.py:275, so the JPEG
  round trip (which shrinks the rejected regions slightly) is reproduced here.

Output: ``<scene>/agsmesh/dnc_mask_u8.npy`` [N_train, H, W] uint8 (decoded JPEG values, same
order as ``dnsplat/names.json``), ``agsmesh/manifest.json`` (settings, rejected fraction, sha256).

Run (open3d 0.19 + cv2 from ~/models/dtueval-pylib):
  PYTHONPATH=$HOME/models/dtueval-pylib tools/ssenv python \
      reproductions/ags-mesh/tools/prepare_dnc.py --room vr_room [--workers 16]
"""

import argparse
import hashlib
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "plugin"))
from ss_agsmesh.dnc import dnc_frame  # noqa: E402

NATIVE = Path.home() / "datasets/mushroom/native"


def _work(args):
    i, root, intr, knn, tau, space, c2w = args
    side = Path(root) / "dnsplat"
    depth = np.load(side / "depth_mm.npy", mmap_mode="r")[i].astype(np.float64) * 1e-3
    normal = np.array(np.load(side / "normal_omni_u8.npy", mmap_mode="r")[i])
    return i, dnc_frame(depth, normal, c2w, intr, knn, tau, space)


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--room", default="vr_room")
    ap.add_argument("--knn", type=int, default=200)
    ap.add_argument("--tau", type=float, default=10.0)
    ap.add_argument("--angle-space", choices=["encoded", "signed"], default="encoded")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0, help="first N frames only (timing)")
    a = ap.parse_args()
    root = NATIVE / a.room
    tj = json.loads((root / "transforms.json").read_text())
    names = json.loads((root / "dnsplat/names.json").read_text())
    by = {f["file_path"]: f for f in tj["frames"]}
    intr = tuple(float(tj[k]) for k in ("fl_x", "fl_y", "cx", "cy"))
    n = len(names) if not a.limit else a.limit
    H, W = int(tj["h"]), int(tj["w"])
    out = np.zeros((n, H, W), dtype=np.uint8)
    jobs = [(i, str(root), intr, a.knn, a.tau, a.angle_space, by[names[i]]["transform_matrix"])
            for i in range(n)]
    with ProcessPoolExecutor(a.workers) as ex:
        for k, (i, m) in enumerate(ex.map(_work, jobs)):
            out[i] = m
            if k % 25 == 0:
                print(f"dnc {k}/{n} rejected {(m == 255).mean():.3f}", flush=True)
    side = root / "agsmesh"
    side.mkdir(exist_ok=True)
    suffix = "" if not a.limit else f"_first{n}"
    np.save(side / f"dnc_mask_u8{suffix}.npy", out)
    rej = float((out == 255).mean())
    man = {"room": a.room, "knn": a.knn, "tau_deg": a.tau, "angle_space": a.angle_space,
           "jpeg_roundtrip": True, "frames": n, "rejected_fraction": rej,
           "dnsplat_manifest_sha256": sha256(root / "dnsplat/manifest.json"),
           "files": {f"dnc_mask_u8{suffix}.npy": sha256(side / f"dnc_mask_u8{suffix}.npy")}}
    (side / f"manifest{suffix}.json").write_text(json.dumps(man, indent=1))
    print("done", side, "rejected", rej)


if __name__ == "__main__":
    main()
