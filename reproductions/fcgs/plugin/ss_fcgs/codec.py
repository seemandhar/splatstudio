"""FCGS encode -> bitstreams -> decode for SplatStudio parameters.

Mirrors the official `encode_single_scene.py` / `decode_single_scene.py`
(github.com/YihangChen-ee/FCGS @ 31e59a46): Gaussians are processed in independent steps of
1,000,000 (1,100,000 when 1.0 M < N < 1.1 M) in their stored order; every step is compressed
by `FCGS.compress` (determ=1) into real bitstream files and then decoded from those files by
`FCGS.decomprss`. Size is reported exactly as the official code/paper does: the sum of the
bit counts returned by `compress` divided by 8*1024*1024 ("MB" = MiB).

Attribute layout (official, 3DGS ply order): [f_dc(3), f_rest(45, coefficient-major then RGB),
opacity logit(1), log-scale(3), raw quaternion wxyz(4)] = 56. SplatStudio stores sh0 [N,1,3],
shN [N,15,3], opacities [N] (logit), scales [N,3] (log), quats [N,4] (wxyz, unnormalised),
which flattens to exactly this layout.
"""

import os
import time
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch

LAMBDAS = (1e-4, 2e-4, 4e-4, 8e-4, 16e-4)
CHUNK_SIZE_LIST = [200_0000, 100_0000, 100_0000]  # encode_single_scene.py
MIB_BITS = 8 * 1024 * 1024
DEFAULT_WEIGHTS = Path.home() / "models" / "fcgs"
BREAKDOWN = ("total", "xyz", "mask", "fea", "fea_main", "fea_hyp", "feq", "feq_main", "feq_hyp",
             "geo", "geo_main", "geo_hyp")


def per_step_size(n):
    """Official step size (encode_single_scene.py:21-23)."""
    return 110_0000 if 100_0000 < n < 110_0000 else 100_0000


def weights_path(lmd, weights_dir=None):
    root = Path(weights_dir or os.environ.get("FCGS_WEIGHTS", DEFAULT_WEIGHTS))
    return root / f"checkpoint_{lmd}.pkl"  # official names: checkpoint_0.0001.pkl ...


def load_model(lmd, weights_dir=None, device="cuda"):
    from .fcgs_model import FCGS

    if lmd not in LAMBDAS:
        raise ValueError(f"FCGS weights exist only for lambda in {LAMBDAS}, got {lmd}")
    path = weights_path(lmd, weights_dir)
    if not path.exists():
        raise FileNotFoundError(f"FCGS pretrained weights not found: {path}")
    with torch.cuda.device(device):
        model = FCGS(Q=1, resolutions_list=[300, 400, 500], resolutions_list_3D=[70, 80, 90],
                     norm_radius=3).cuda()
    state = torch.load(path, map_location=model.ad_fe.device, weights_only=True)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def params_to_fea(params):
    n = len(params["means"])
    return torch.cat([
        params["sh0"].detach().reshape(n, 3),
        params["shN"].detach().reshape(n, 45),
        params["opacities"].detach().reshape(n, 1),
        params["scales"].detach().reshape(n, 3),
        params["quats"].detach().reshape(n, 4),
    ], dim=-1).float().contiguous()


def fea_to_params(xyz, fea):
    f_dc, f_rst, op, sc, ro = torch.split(fea, [3, 45, 1, 3, 4], dim=-1)
    return {"means": xyz, "sh0": f_dc.reshape(-1, 1, 3), "shN": f_rst.reshape(-1, 15, 3),
            "opacities": op.reshape(-1), "scales": sc.reshape(-1, 3), "quats": ro.reshape(-1, 4)}


def _dir_bytes(path):
    return sum(f.stat().st_size for f in Path(path).rglob("*") if f.is_file())


@torch.no_grad()
def encode_decode(params, lmd, *, weights_dir=None, xyz_coder="auto", bitstream_dir=None,
                  model=None, check_roundtrip=True):
    """Compress params with the pretrained FCGS(lmd), decode the written bitstreams.

    Returns (decoded params dict, report). The decoded model is what evaluation must render.
    """
    from . import xyz_codec

    coder = xyz_codec.resolve_coder(xyz_coder)
    device = params["means"].device
    model = model or load_model(lmd, weights_dir, device)
    xyz = params["means"].detach().float().contiguous()
    fea = params_to_fea(params)
    n = len(xyz)
    step = per_step_size(n)
    steps = int(np.ceil(n / step))
    sums = dict.fromkeys(BREAKDOWN, 0.0)
    mask_ones = 0
    tmp = None
    if bitstream_dir is None:
        tmp = TemporaryDirectory(prefix="fcgs-bits-")
        bitstream_dir = tmp.name
    root = Path(bitstream_dir)
    out_xyz, out_fea = [], []
    max_err = 0.0
    try:
        torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        enc_side = []
        for s in range(steps):
            path = root / f"{lmd}" / f"{s}"
            path.mkdir(parents=True, exist_ok=True)
            res = model.compress(xyz[s * step:(s + 1) * step], fea[s * step:(s + 1) * step],
                                 root_path=str(path), chunk_size_list=CHUNK_SIZE_LIST,
                                 determ_codec=1, xyz_coder=coder)
            for key, value in zip(BREAKDOWN, res[3:]):
                sums[key] += float(value)
            mask_ones += int(res[2].sum().item())
            enc_side.append((res[0], res[1]))
        torch.cuda.synchronize(device)
        t1 = time.perf_counter()
        for s in range(steps):
            g_xyz, g_fea = model.decomprss(root_path=str(root / f"{lmd}" / f"{s}"),
                                           chunk_size_list=CHUNK_SIZE_LIST, xyz_coder=coder)
            if check_roundtrip:
                e_xyz, e_fea = enc_side[s]
                max_err = max(max_err, float((g_xyz - e_xyz).abs().max()),
                              float((g_fea - e_fea).abs().max()))
            out_xyz.append(g_xyz)
            out_fea.append(g_fea)
        torch.cuda.synchronize(device)
        t2 = time.perf_counter()
        disk = _dir_bytes(root / f"{lmd}")
    finally:
        if tmp is not None:
            tmp.cleanup()
    dec_xyz = torch.cat(out_xyz).to(device)
    dec_fea = torch.cat(out_fea).to(device)
    decoded = fea_to_params(dec_xyz, dec_fea)
    report = {
        "lambda": lmd,
        "gaussians": n,
        "steps": steps,
        "xyz_coder_used": coder,
        "size_MB": sums["total"],  # paper/official: sum of bits / (8*1024*1024)
        "size_MB_breakdown": {k: sums[k] for k in BREAKDOWN if k != "total"},
        "size_MB_attributes_only": sums["total"] - sums["xyz"],
        "bitstream_disk_bytes": disk,
        "mask_rate": mask_ones / n,
        "encode_seconds": t1 - t0,
        "decode_seconds": t2 - t1,
        "roundtrip_max_abs_error": max_err if check_roundtrip else None,
        "uncompressed_ply_MB": n * 62 * 4 / 2**20,  # 3DGS ply: 62 float32 / Gaussian
    }
    return decoded, report
