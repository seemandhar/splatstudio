"""Official LightGaussian storage (`vectree.py:fully_vq_reformat` / `utils.py:load_vqgaussian`).

`extreme_saving/` holds seven `np.savez_compressed` files (metadata, vq_indexs: 13-bit MSB-first
packed indices, codebook fp16, non_vq_mask packbits, non_vq_feats fp16, other_attribute fp16
[opacity, scale x3, rot x4] raw, xyz fp32); the reported size is the `zip -r` archive in MiB
(`os.path.getsize(...) / 1024 / 1024`). Built in memory here; the zip uses DEFLATE level 6 like
Info-ZIP's default.
"""

import io
import math
import zipfile

import numpy as np
import torch

FILES = ("metadata.npz", "vq_indexs.npz", "codebook.npz", "non_vq_mask.npz",
         "non_vq_feats.npz", "other_attribute.npz", "xyz.npz")


def _npz(**arrays):
    buf = io.BytesIO()
    np.savez_compressed(buf, **arrays)
    return buf.getvalue()


def _load(data, name="arr_0", allow_pickle=False):
    return np.load(io.BytesIO(data), allow_pickle=allow_pickle)[name]


def dec2bin(x, bits):
    mask = 2 ** torch.arange(bits - 1, -1, -1, device=x.device, dtype=x.dtype)
    return x.unsqueeze(-1).bitwise_and(mask).ne(0)


def bin2dec(b, bits):
    mask = 2 ** torch.arange(bits - 1, -1, -1, device=b.device, dtype=torch.long)
    return torch.sum(mask * b.long(), -1)


@torch.no_grad()
def encode(xyz, sh_feats, other, codebook, indices, non_vq_mask, input_pc_dim=41):
    """Returns (zip bytes, {file: npz bytes}).

    xyz [N,3]; sh_feats [N,27] float (PLY order; only non-VQ rows are stored); other [N,8]
    raw opacity/scale/rot; codebook [K,27]; indices [N_vq] (rows of ~non_vq_mask in order).
    `input_pc_dim` is the official PLY width for SH degree 2 (3+3+27+1+3+4 = 41), metadata only.
    """
    k = codebook.shape[0]
    bits = int(math.log2(k))
    meta = {"input_pc_num": int(len(xyz)), "input_pc_dim": int(input_pc_dim),
            "codebook_size": int(k), "codebook_dim": int(codebook.shape[1])}
    meta_buf = io.BytesIO()
    np.savez_compressed(meta_buf, metadata=meta)
    bin_idx = dec2bin(indices.long(), bits).cpu().numpy()
    files = {
        "metadata.npz": meta_buf.getvalue(),
        "vq_indexs.npz": _npz(arr_0=np.packbits(bin_idx.reshape(-1))),
        "codebook.npz": _npz(arr_0=codebook.detach().cpu().half().numpy()),
        "non_vq_mask.npz": _npz(arr_0=np.packbits(non_vq_mask.cpu().numpy().reshape(-1))),
        "non_vq_feats.npz": _npz(arr_0=sh_feats[non_vq_mask].detach().cpu().half().numpy()),
        "other_attribute.npz": _npz(arr_0=other.detach().cpu().half().numpy()),
        "xyz.npz": _npz(arr_0=xyz.detach().cpu().float().numpy()),
    }
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        z.writestr("extreme_saving/", b"")
        for name in FILES:
            z.writestr(f"extreme_saving/{name}", files[name])
    return buf.getvalue(), files


@torch.no_grad()
def decode(zip_bytes, device):
    """`load_vqgaussian` -> dict(xyz [N,3], sh_feats [N,D], other [N,8]) as float32."""
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as z:
        files = {n: z.read(f"extreme_saving/{n}") for n in FILES}
    meta = _load(files["metadata.npz"], "metadata", allow_pickle=True).item()
    n, dim, k = meta["input_pc_num"], meta["codebook_dim"], meta["codebook_size"]
    bits = int(math.log2(k))
    non_vq = torch.from_numpy(np.unpackbits(_load(files["non_vq_mask.npz"]))[:n]).bool()
    non_vq = non_vq.to(device)
    vq_mask = ~non_vq
    m = int(vq_mask.sum())
    codebook = torch.from_numpy(_load(files["codebook.npz"])).float().to(device)
    raw = np.unpackbits(_load(files["vq_indexs.npz"]))[: m * bits].reshape(m, bits)
    idx = bin2dec(torch.from_numpy(raw).to(device), bits)
    sh = torch.zeros(n, dim, device=device)
    sh[vq_mask] = codebook[idx]
    sh[non_vq] = torch.from_numpy(_load(files["non_vq_feats.npz"])).float().to(device)
    other = torch.from_numpy(_load(files["other_attribute.npz"])).float().to(device)
    xyz = torch.from_numpy(_load(files["xyz.npz"])).float().to(device)
    return {"xyz": xyz, "sh_feats": sh, "other": other, "codebook": codebook, "indices": idx,
            "non_vq_mask": non_vq}


def size_report(zip_bytes, files, n_gaussians, n_vq):
    return {
        "zip_bytes": len(zip_bytes),
        "size_MiB": len(zip_bytes) / 1024.0 / 1024.0,
        "size_MB": len(zip_bytes) / 1e6,
        "file_bytes": {k: len(v) for k, v in files.items()},
        "gaussians": int(n_gaussians),
        "vq_gaussians": int(n_vq),
    }
