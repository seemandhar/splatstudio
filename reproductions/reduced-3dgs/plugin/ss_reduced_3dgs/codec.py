"""C3 — codebook quantisation + half floats, with an explicit encode -> bytes -> decode round trip.

Official code (scene/gaussian_model.py): generate_codebook / produce_clusters (K-means
codebooks), save_ply(quantised=True, half_float=True) (the stored `point_cloud_quantised_half.ply`
layout) and load_ply / _parse_vertex_group (decoding, as render.py does for `quantised_half`);
encoded size: generate_results.py:memory_results.
"""

import hashlib
import io
from collections import OrderedDict

import numpy as np
import torch
from plyfile import PlyData, PlyElement

from . import kernels
from .renderer import DEGREE_FIELD

MAX_SH_DEGREE = 3
MAX_COEFFS = (MAX_SH_DEGREE + 1) ** 2 - 1  # 15
MIB = 2**20
FLOAT16_MAX = 65504.0


@torch.no_grad()
def kmeans_1d(values, num_clusters=256, tol=1e-4, max_iterations=500, generator=None,
              assign=None):
    """Official Reduced3DGS::kmeans on flattened values (generate_codebook).

    init: num_clusters random data values; Lloyd iterations; empty cluster centre -> 0 (0/0 nan
    -> 0); stop when sum |old - new| < tol; final reassignment. Returns (ids int64, centres).
    """
    assign = assign or kernels.kmeans_assign
    values = values.flatten().float().contiguous()
    pick = torch.randint(values.shape[0], (num_clusters,), device=values.device,
                         generator=generator)
    centers = values[pick].clone()
    for _ in range(max_iterations):
        ids = assign(values, centers)
        old = centers
        sums = torch.zeros(num_clusters, device=values.device).index_add_(0, ids, values)
        sizes = torch.zeros(num_clusters, device=values.device).index_add_(
            0, ids, torch.ones_like(values)
        )
        centers = torch.nan_to_num(sums / sizes, nan=0.0)
        if float((old - centers).abs().sum()) < tol:
            break
    ids = assign(values, centers)
    return ids, centers


def inverse_sigmoid(x):
    return torch.log(x / (1 - x))


@torch.no_grad()
def produce_codebooks(params, num_clusters=256, seed=0, assign=None):
    """OrderedDict name -> (ids uint8 tensor shaped like the attribute, stored centres float32).

    Spaces as in the official produce_clusters: DC and SH-rest raw (one codebook per SH-rest
    coefficient, shared by RGB), opacity clustered after sigmoid and stored as logit, scale
    clustered after exp and stored as log, rotation on the normalised quaternion (real part,
    imaginary parts).
    """
    g = torch.Generator(device=params["means"].device)
    g.manual_seed(int(seed))
    quats = torch.nn.functional.normalize(params["quats"].detach(), dim=-1)
    jobs = [("features_dc", params["sh0"].detach()[:, 0], None, 0.001)]
    jobs += [(f"features_rest_{i}", params["shN"].detach()[:, i], None, 1e-4)
             for i in range(MAX_COEFFS)]
    jobs += [
        ("opacity", params["opacities"].detach().sigmoid()[:, None], inverse_sigmoid, 1e-4),
        ("scaling", params["scales"].detach().exp(), torch.log, 1e-4),
        ("rotation_re", quats[:, 0:1], None, 1e-4),
        ("rotation_im", quats[:, 1:], None, 1e-4),
    ]
    books = OrderedDict()
    for name, values, inverse, tol in jobs:
        ids, centers = kmeans_1d(values, num_clusters, tol, generator=g, assign=assign)
        stored = inverse(centers) if inverse is not None else centers
        books[name] = (ids.to(torch.uint8).view(values.shape), stored)
    return books


def _attributes(coeffs_num):
    """Official construct_list_of_attributes(rest_coeffs = 3 * coeffs_num)."""
    return (["x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2"]
            + [f"f_rest_{i}" for i in range(3 * coeffs_num)]
            + ["opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"])


@torch.no_grad()
def encode_ply(params, books, half_float=True):
    """Official save_ply(quantised=True, half_float) written to memory; returns bytes."""
    float_type = "int16" if half_float else "f4"
    degree = params[DEGREE_FIELD].detach().round().long()
    xyz_all = params["means"].detach()
    opacity = books["opacity"][0]
    scaling = books["scaling"][0]
    rot = torch.cat((books["rotation_re"][0], books["rotation_im"][0]), dim=1)
    f_dc = books["features_dc"][0]
    f_rest = torch.stack([books[f"features_rest_{i}"][0] for i in range(MAX_COEFFS)], dim=1)
    elements = []
    for d in range(MAX_SH_DEGREE + 1):
        coeffs_num = (d + 1) ** 2 - 1
        m = degree == d
        n = int(m.sum())
        xyz = xyz_all[m].cpu()
        xyz = xyz.half().view(torch.int16).numpy() if half_float else xyz.numpy()
        # rest stored rrr ggg bbb (transpose(1, 2)) as in the official save_ply
        rest = f_rest[m][:, :coeffs_num].transpose(1, 2).reshape(n, 3 * coeffs_num).cpu().numpy()
        cols = np.concatenate(
            [f_dc[m].cpu().numpy(), rest, opacity[m].cpu().numpy(), scaling[m].cpu().numpy(),
             rot[m].cpu().numpy()], axis=1)
        names = _attributes(coeffs_num)
        dtype = [(a, float_type) if a in ("x", "y", "z") else (a, "u1") for a in names]
        el = np.empty(n, dtype=dtype)
        for k, a in enumerate(("x", "y", "z")):
            el[a] = xyz[:, k]
        for k, a in enumerate(names[3:]):
            el[a] = cols[:, k]
        elements.append(PlyElement.describe(el, f"vertex_{d}"))
    centers = [books[k][1].detach().reshape(-1).cpu().numpy() for k in books]
    if half_float:
        centers = [c.astype(np.float16).view(np.int16) for c in centers]
    cb = np.empty(len(centers[0]), dtype=[(k, float_type) for k in books])
    for k, c in zip(books, centers):
        cb[k] = c
    elements.append(PlyElement.describe(cb, "codebook_centers"))
    buf = io.BytesIO()
    PlyData(elements).write(buf)
    return buf.getvalue()


def _f32(a, half_float):
    a = np.asarray(a)
    return a.view(np.float16).astype(np.float32) if half_float else a.astype(np.float32)


@torch.no_grad()
def decode_ply(data, device, half_float=True):
    """Official load_ply(quantised=True, half_float) from bytes -> dict of native params."""
    ply = PlyData.read(io.BytesIO(data))
    cb = ply.elements[-1]

    def book(name):
        return torch.from_numpy(_f32(cb[name], half_float)).to(device)

    rest_book = torch.stack([book(f"features_rest_{i}") for i in range(MAX_COEFFS)], dim=1)
    out = {k: [] for k in ("means", "sh0", "shN", "opacities", "scales", "quats", DEGREE_FIELD)}
    for d in range(MAX_SH_DEGREE + 1):
        el = ply.elements[d]
        n = el.count
        coeffs_num = (d + 1) ** 2 - 1

        def ids(names):
            return torch.from_numpy(
                np.stack([np.asarray(el[a], dtype=np.int64) for a in names], axis=1)
            ).to(device)

        xyz = np.stack([_f32(el[a], half_float) for a in ("x", "y", "z")], axis=1)
        out["means"].append(torch.from_numpy(xyz).to(device))
        out["sh0"].append(book("features_dc")[ids(["f_dc_0", "f_dc_1", "f_dc_2"])][:, None, :])
        rest = torch.zeros((n, MAX_COEFFS, 3), device=device)
        if coeffs_num:
            r = ids([f"f_rest_{i}" for i in range(3 * coeffs_num)]).view(n, 3, coeffs_num)
            cols = torch.arange(coeffs_num, device=device)[None, None, :].expand(n, 3, -1)
            rest[:, :coeffs_num] = rest_book[r, cols].transpose(1, 2)
        out["shN"].append(rest)
        out["opacities"].append(book("opacity")[ids(["opacity"])[:, 0]])
        out["scales"].append(book("scaling")[ids(["scale_0", "scale_1", "scale_2"])])
        rot = ids(["rot_0", "rot_1", "rot_2", "rot_3"])
        out["quats"].append(torch.cat((book("rotation_re")[rot[:, :1]],
                                       book("rotation_im")[rot[:, 1:]]), dim=1))
        out[DEGREE_FIELD].append(torch.full((n,), float(d), device=device))
    out = {k: torch.cat(v, dim=0) for k, v in out.items()}
    # logit(1) / log(0) codebook entries are +-inf; clamp to the float16 range so the runtime's
    # finiteness check passes (sigmoid/exp give the identical 1 / 0).
    for k in ("opacities", "scales"):
        out[k] = out[k].clamp(-FLOAT16_MAX, FLOAT16_MAX)
    return out


def band_counts(degree):
    return [int(c) for c in torch.bincount(degree.detach().round().long(), minlength=4)[:4]]


def paper_memory_bytes(counts, quantised=True, half_float=True):
    """generate_results.py:memory_results (bytes; the paper's 'MB' = bytes / 2**20)."""
    p0, p1, p2, p3 = counts
    p = p0 + p1 + p2 + p3
    colour_values = (p0 + p1 * 4 + p2 * 9 + p3 * 16) * 3
    xyz_values, rest_values = p * 3, p * 8
    if quantised:
        colour = colour_values + 256 * 16 * 4
        rest = rest_values + 256 * 8 * 4
        xyz = xyz_values * (2 if half_float else 4)
    elif half_float:
        colour, rest, xyz = colour_values * 2, rest_values * 2, xyz_values * 2
    else:
        colour, rest, xyz = colour_values * 4, rest_values * 4, xyz_values * 4
    return colour + rest + xyz


def baseline_memory_bytes(num_gaussians):
    """Uncompressed 3DGS model as counted by the paper: P * 59 floats."""
    return num_gaussians * 59 * 4


@torch.no_grad()
def encode_decode(params, num_clusters=256, seed=0, half_float=True):
    """Full codec: codebooks -> official PLY bytes -> decoded params + size report."""
    counts = band_counts(params[DEGREE_FIELD])
    books = produce_codebooks(params, num_clusters, seed)
    data = encode_ply(params, books, half_float)
    decoded = decode_ply(data, params["means"].device, half_float)
    report = {
        "gaussians": int(len(params["means"])),
        "band_counts": counts,
        "band_fractions": [c / max(1, sum(counts)) for c in counts],
        "paper_mem_bytes": paper_memory_bytes(counts, True, half_float),
        "paper_mem_MB": paper_memory_bytes(counts, True, half_float) / MIB,
        "paper_mem_MB_unquantised": paper_memory_bytes(counts, False, False) / MIB,
        "paper_mem_MB_kmeans_only": paper_memory_bytes(counts, True, False) / MIB,
        "ply_bytes": len(data),
        "ply_MB": len(data) / MIB,
        "ply_sha256": hashlib.sha256(data).hexdigest(),
        "half_float": bool(half_float),
        "codebook_size": int(num_clusters),
    }
    return decoded, report
