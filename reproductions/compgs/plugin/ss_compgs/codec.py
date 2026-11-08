"""CompGS storage size (pure functions).

* `paper_size`: paper §3 storage — fp32 position (3) + opacity (1) per Gaussian, fp32
  codebooks, ceil(log2 K)-bit indices for each quantised attribute, except that the Gaussians are
  sorted by one index (rotation) which is stored run-length coded as K 32-bit counts.
* `official_disk_size`: what the released code writes — point_cloud.ply with x,y,z,opacity
  (fp32, header ignored), kmeans_inds.bin with ceil(log2 N) bits per index (`save_kmeans` uses
  len(cls_ids) = N, not K), kmeans_centers.pth fp32 (pickle overhead ignored).
"""

import math


def bits_for(k):
    return max(1, math.ceil(math.log2(k)))


def paper_size(n, codebooks, rle_field="quats"):
    """codebooks: {field: (K, d)}; returns a dict of byte counts and MB (1e6) / MiB."""
    non_quant = n * 4 * 4
    cb = sum(k * d * 4 for k, d in codebooks.values())
    idx_bits = sum(n * bits_for(k) for f, (k, _) in codebooks.items() if f != rle_field)
    rle = codebooks[rle_field][0] * 4 if rle_field in codebooks else 0
    total = non_quant + cb + math.ceil(idx_bits / 8) + rle
    return {"bytes": total, "MB": total / 1e6, "MiB": total / 2**20, "non_quant_bytes": non_quant,
            "codebook_bytes": cb, "index_bytes": math.ceil(idx_bits / 8), "rle_bytes": rle}


def official_disk_size(n, codebooks):
    ply = n * 4 * 4
    inds = math.ceil(len(codebooks) * n * bits_for(n) / 8)
    cb = sum(k * d * 4 for k, d in codebooks.values())
    total = ply + inds + cb
    return {"bytes": total, "MB": total / 1e6, "MiB": total / 2**20}


def uncompressed_3dgs_size(n):
    """59 fp32 values per Gaussian (3DGS ply without normals)."""
    total = n * 59 * 4
    return {"bytes": total, "MB": total / 1e6}
