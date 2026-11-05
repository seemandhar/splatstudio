"""C4 entropy encoding and the decode used for evaluation.

`encode_npz` reproduces the official `GaussianModel.save_npz(path, sort_morton=True)` of a
quantisation-aware indexed model (keys, order, dtypes and shapes) and writes it with
`np.savez_compressed` (DEFLATE) into memory; the paper's size is
`os.path.getsize(point_cloud.npz) / 1024**2` (compress.py), i.e. len(bytes) / 2**20.
`decode_npz` is the official `load_npz` dequantisation followed by the render activations, giving
native base fields (means, log scales, wxyz quats, logit opacities, sh0, shN).
"""

import hashlib
import io

import numpy as np
import torch
import torch.nn.functional as F

from .quant import dequantize, half_ste, morton_order, quantize_int8

MIB = 2**20
FLOAT16_MAX = 65504.0
PLY_FLOATS_3DGS = 62  # x y z, nx ny nz, f_dc 3, f_rest 45, opacity, scale 3, rot 4


def _np(t):
    return t.detach().cpu().numpy()


@torch.no_grad()
def encode_npz(bank, params, sort_morton=True):
    """Official save_npz (quantization=True, colour and shape indexed). Returns (bytes, order)."""
    t, q = bank.tensors, bank.q
    order = (morton_order(params["means"].detach()) if sort_morton
             else torch.arange(bank.n, device=params["means"].device))
    xyz = params["means"].detach()[order]
    opacity = params["opacities"].detach()[order][:, None]
    log_eta = t["log_eta"].detach()[order]
    color_idx = bank.color_idx[order]
    gauss_idx = bank.gauss_idx[order]

    d = {}
    d["quantization"] = True
    d["xyz"] = _np(half_ste(xyz).half())
    d["features_dc"] = _np(quantize_int8(t["cb_dc"], q["features_dc"]))
    d["features_dc_scale"] = _np(q["features_dc"].scale)
    d["features_dc_zero_point"] = _np(q["features_dc"].zero_point)
    d["features_rest"] = _np(quantize_int8(t["cb_rest"], q["features_rest"]))
    d["features_rest_scale"] = _np(q["features_rest"].scale)
    d["features_rest_zero_point"] = _np(q["features_rest"].zero_point)
    d["opacity"] = _np(quantize_int8(torch.sigmoid(opacity), q["opacity"]))
    d["opacity_scale"] = _np(q["opacity"].scale)
    d["opacity_zero_point"] = _np(q["opacity"].zero_point)
    d["feature_indices"] = _np(color_idx.contiguous().int())
    d["gaussian_indices"] = _np(gauss_idx.contiguous().int())
    d["scaling"] = _np(quantize_int8(F.normalize(F.relu(t["cb_scale"].detach())), q["scaling"]))
    d["scaling_scale"] = _np(q["scaling"].scale)
    d["scaling_zero_point"] = _np(q["scaling"].zero_point)
    d["scaling_factor"] = _np(quantize_int8(log_eta, q["scaling_factor"]))
    d["scaling_factor_scale"] = _np(q["scaling_factor"].scale)
    d["scaling_factor_zero_point"] = _np(q["scaling_factor"].zero_point)
    d["rotation"] = _np(quantize_int8(F.normalize(t["cb_rot"].detach()), q["rotation"]))
    d["rotation_scale"] = _np(q["rotation"].scale)
    d["rotation_zero_point"] = _np(q["rotation"].zero_point)
    buf = io.BytesIO()
    np.savez_compressed(buf, **d)
    return buf.getvalue(), order


def _deq(z, key, device):
    return dequantize(torch.from_numpy(z[key]).to(device), torch.from_numpy(z[key + "_scale"])
                      .to(device), torch.from_numpy(z[key + "_zero_point"]).to(device)).float()


@torch.no_grad()
def decode_npz(data, device):
    """Official load_npz + render activations -> native base fields (Morton order).

    Stored values are already the quantised activations: opacity = Q(sigmoid), scaling =
    Q(normalize(relu(s_hat))) (used as stored, which is the value the fake-quantised model
    rendered), rotation = Q(normalize(r)) (normalised again, as render() does), log eta = Q(.).
    """
    z = np.load(io.BytesIO(data))
    if not bool(z["quantization"]):
        raise ValueError("c3dgs npz without quantization is not produced by this plugin")
    means = torch.from_numpy(z["xyz"]).float().to(device)
    dc = _deq(z, "features_dc", device)
    rest = _deq(z, "features_rest", device)
    color_idx = torch.from_numpy(z["feature_indices"]).long().to(device)
    gauss_idx = torch.from_numpy(z["gaussian_indices"]).long().to(device)
    opacity = _deq(z, "opacity", device)[:, 0]
    scale_n = _deq(z, "scaling", device)
    eta = torch.exp(_deq(z, "scaling_factor", device))
    rotation = F.normalize(_deq(z, "rotation", device))
    scales = eta * scale_n[gauss_idx]
    out = {
        "means": means,
        "sh0": dc[color_idx],
        "shN": rest[color_idx],
        # logit(0) / logit(1) / log(0) are +-inf: clamp to the float16 range so the runtime's
        # finiteness check passes (sigmoid / exp give the identical 0, 1 and 0).
        "opacities": torch.logit(opacity.clamp(0, 1).double()).float().clamp(-FLOAT16_MAX, FLOAT16_MAX),
        "scales": scales.log().clamp(-FLOAT16_MAX, FLOAT16_MAX),
        "quats": rotation[gauss_idx],
    }
    return out


def size_report(data, n_before_prune, n_after_prune):
    mb = len(data) / MIB
    ply_3dgs = n_before_prune * PLY_FLOATS_3DGS * 4 / MIB
    return {
        "npz_bytes": len(data),
        "size_MB": mb,
        "npz_sha256": hashlib.sha256(data).hexdigest(),
        "gaussians_before_prune": int(n_before_prune),
        "gaussians": int(n_after_prune),
        "3dgs_ply_MB_equivalent": ply_3dgs,
        "compression_ratio_vs_3dgs_ply": ply_3dgs / mb,
    }
