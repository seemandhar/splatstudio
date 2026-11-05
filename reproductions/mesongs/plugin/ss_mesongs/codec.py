"""C7 encoding (official `GaussianModel.save_npz`, per_block_quant=True) and the decoder used
for evaluation (official `GaussianModel.load_npz`).

Bitstream = the official `bins/` directory zipped (`zip -j bins.zip bins/*`): six
`np.savez_compressed` files (DEFLATE) — `ct.npz{i: uint8 log-scale codes, channel-major}`,
`ntk.npz{ntk: int32 codebook indices}`, `oct.npz{points: int64 sorted voxel keys, params: 6
bounds}`, `orgb.npz{f: float32 DC coefficients [7], i: uint8 AC codes, channel-major}`,
`t.npz{t: float64 [depth, n_block, (scale, zero point) per block ...]}`, `um.npz{umap: float32
codebook [K, 45]}` — stored in a DEFLATE zip in memory; size = len(zip bytes) (the official
csv "Size [Bytes]"; the paper's MB = bytes / 2**20). Python's zipfile writes no extra header fields
(Info-ZIP adds ~28 bytes per member).

`decode` rebuilds everything from the bytes alone: voxel positions (lower corners), RAHT tables
from the decoded voxel codes, block dequantisation, inverse RAHT, Euler -> rotation matrix ->
wxyz quaternion for the native parameter layout.
"""

import hashlib
import io
import zipfile

import numpy as np
import torch

from . import euler, octree, raht
from .bank import RF_CHANNELS
from .quant import dequantize_tensor, split_length

MIB = 2**20
PLY_FLOATS_3DGS = 62
MEMBERS = ("ct.npz", "ntk.npz", "oct.npz", "orgb.npz", "t.npz", "um.npz")  # zip -j glob order


def _npz(**arrays):
    buf = io.BytesIO()
    np.savez_compressed(buf, **arrays)
    return buf.getvalue()


@torch.no_grad()
def encode(bank, params, oct_keys, oct_params, depth):
    """Official save_npz + zip; returns (zip bytes, member sizes)."""
    bank.attributes(params, update=True)  # quantiser ranges of the final forward (idempotent)
    C = bank.coefficients(params)
    qci = bank.ac_q.encode(C[1:]).t().reshape(-1)  # channel-major concatenation
    scaling_q = bank.scale_q.encode(params["scales"].detach()).t().reshape(-1)
    trans = np.concatenate(([depth, bank.n_block], bank.ac_q.trans().cpu().numpy(),
                            bank.scale_q.trans().cpu().numpy())).astype(np.float64)
    files = {
        "ct.npz": _npz(i=scaling_q.cpu().numpy().astype(np.uint8)),
        "ntk.npz": _npz(ntk=bank.indices.int().cpu().numpy()),
        "oct.npz": _npz(points=np.asarray(oct_keys), params=np.asarray(oct_params)),
        "orgb.npz": _npz(f=C[0].float().cpu().numpy(), i=qci.cpu().numpy().astype(np.uint8)),
        "t.npz": _npz(t=trans),
        "um.npz": _npz(umap=bank.codebook.detach().contiguous().cpu().numpy()),
    }
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for name in MEMBERS:
            z.writestr(name, files[name])
    return buf.getvalue(), {k: len(v) for k, v in files.items()}


def _dequant_ave(x, split, sz):
    out, start, cnt = [], 0, 0
    for length in split:
        out.append(dequantize_tensor(x[start:start + length], float(sz[cnt]), float(sz[cnt + 1])))
        cnt += 2
        start += length
    return torch.cat(out, 0)


@torch.no_grad()
def decode(data, device, tables=None):
    """Official load_npz -> native base fields (means, scales, quats, opacities, sh0, shN).

    `tables` may pass precomputed RAHT tables of the same geometry (probes compare both paths);
    by default they are rebuilt from the decoded voxel codes."""
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        f = {name: np.load(io.BytesIO(z.read(name))) for name in MEMBERS}
    trans = f["t.npz"]["t"]
    depth, n_block = int(trans[0]), int(trans[1])
    oct_keys, oct_params = f["oct.npz"]["points"], f["oct.npz"]["params"]
    dxyz, V = octree.decode_oct(oct_params, oct_keys, depth)
    n = len(dxyz)
    ntk = torch.from_numpy(f["ntk.npz"]["ntk"].astype(np.int64)).to(device)
    cb = torch.from_numpy(f["um.npz"]["umap"]).float().to(device)
    dc = torch.from_numpy(f["orgb.npz"]["f"]).float().to(device)
    q_ac = torch.from_numpy(f["orgb.npz"]["i"].astype(np.float32)).to(device)
    q_ac = q_ac.reshape(RF_CHANNELS, -1).t()
    q_sc = torch.from_numpy(f["ct.npz"]["i"].astype(np.float32)).to(device).reshape(3, -1).t()
    if len(q_ac) + 1 != n or len(q_sc) != n:
        raise ValueError("MesonGS bitstream: attribute and geometry lengths differ")
    cnt = 2
    split = split_length(n - 1, n_block)
    ac = []
    for i in range(RF_CHANNELS):
        ac.append(_dequant_ave(q_ac[:, i], split, trans[cnt:cnt + 2 * n_block]))
        cnt += 2 * n_block
    split = split_length(n, n_block)
    sc = []
    for i in range(3):
        sc.append(_dequant_ave(q_sc[:, i], split, trans[cnt:cnt + 2 * n_block]))
        cnt += 2 * n_block
    C = torch.cat((dc[None], torch.stack(ac, -1)), 0)
    if tables is None:
        tables = raht.RAHTTables(V, depth, device)
    feats = raht.inverse_transform(C, tables)
    e = feats[:, 1:4].nan_to_num(0.0)
    quats = euler.matrix_to_quaternion(euler.rotation_of(e))
    return {
        "means": torch.from_numpy(dxyz).float().to(device),
        "scales": torch.stack(sc, -1),
        "quats": quats,
        "opacities": feats[:, 0],
        "sh0": feats[:, 4:7].reshape(-1, 1, 3),
        "shN": cb.reshape(-1, 15, 3)[ntk],
    }


def size_report(data, members, n_before_prune, n_after_prune, n_voxels):
    ply_3dgs = n_before_prune * PLY_FLOATS_3DGS * 4
    return {
        "zip_bytes": len(data),
        "size_MB": len(data) / MIB,
        "size_MB_1e6": len(data) / 1e6,
        "members_bytes": members,
        "zip_sha256": hashlib.sha256(data).hexdigest(),
        "gaussians_before_prune": int(n_before_prune),
        "gaussians_after_prune": int(n_after_prune),
        "gaussians": int(n_voxels),
        "3dgs_ply_MB_equivalent": ply_3dgs / MIB,
        "compression_ratio_vs_3dgs_ply": ply_3dgs / len(data),
    }
