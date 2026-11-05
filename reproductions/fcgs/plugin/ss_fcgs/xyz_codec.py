"""Gaussian-position coding for FCGS (paper §3.4: "16-bit quantized and encoded losslessly").

Voxelisation, sorting, devoxelisation, the GPCC command line and the container layout are
copied from the official `model/gpcc_utils.py` (github.com/YihangChen-ee/FCGS @ 31e59a46).

Coders (`coder` argument):
- "gpcc":   official path, MPEG G-PCC `tmc3` with the official flags. The binary is taken from
            $FCGS_TMC3 or `tmc3` on PATH.
- "octree": substitute lossless coder used when tmc3 is not installed: Morton-order octree
            occupancy bytes (16 levels) + duplicate counts, each compressed with LZMA. It is
            lossless on the same 16-bit grid, so the decoded model is bit-identical to the GPCC
            path; only the position bitstream size differs (normally larger than GPCC).
- "auto":   "gpcc" if tmc3 is found, otherwise "octree". The coder actually used is returned
            in the report (`xyz_coder_used`).
"""

import lzma
import os
import shutil
from tempfile import TemporaryDirectory

import numpy as np
import torch

VOXELIZE_SCALE_FACTOR = 16
LEVELS = VOXELIZE_SCALE_FACTOR


def tmc3_path():
    path = os.environ.get("FCGS_TMC3") or shutil.which("tmc3")
    return path if path and os.access(path, os.X_OK) else None


def resolve_coder(coder):
    if coder == "auto":
        return "gpcc" if tmc3_path() else "octree"
    if coder == "gpcc" and not tmc3_path():
        raise RuntimeError("xyz_coder='gpcc' needs the MPEG tmc3 binary ($FCGS_TMC3 or PATH)")
    if coder not in ("gpcc", "octree"):
        raise ValueError(f"unknown xyz coder {coder!r}")
    return coder


# ---- official helpers (gpcc_utils.py) -------------------------------------------------------

def gpcc_encode(encoder_path, ply_path, bin_path):
    enc_cmd = (f'{encoder_path} '
               f'--mode=0 --trisoupNodeSizeLog2=0 --mergeDuplicatedPoints=0 --neighbourAvailBoundaryLog2=8 '
               f'--intra_pred_max_node_size_log2=3 --positionQuantizationScale=1 --inferredDirectCodingMode=3 '
               f'--maxNumQtBtBeforeOt=2 --minQtbtSizeLog2=0 --planarEnabled=0 --planarModeIdcmUse=0 --cabac_bypass_stream_enabled_flag=1 '
               f'--uncompressedDataPath={ply_path} --compressedStreamPath={bin_path} > /dev/null 2>&1')
    exit_code = os.system(enc_cmd)
    assert exit_code == 0, f'GPCC encoder failed with exit code {exit_code}.'


def gpcc_decode(decoder_path, bin_path, recon_path):
    dec_cmd = (f'{decoder_path} --mode=1 --outputBinaryPly=1 '
               f'--compressedStreamPath={bin_path} --reconstructedDataPath={recon_path} > /dev/null 2>&1')
    exit_code = os.system(dec_cmd)
    assert exit_code == 0, f'GPCC decoder failed with exit code {exit_code}.'


def write_ply_geo_ascii(geo_data, ply_path):
    geo_data = geo_data.astype(int)
    with open(ply_path, 'w') as f:
        f.writelines(['ply\n', 'format ascii 1.0\n', f'element vertex {geo_data.shape[0]}\n',
                      'property float x\n', 'property float y\n', 'property float z\n', 'end_header\n'])
        np.savetxt(f, geo_data, fmt='%d')


def read_ply_geo_bin(ply_path):
    from plyfile import PlyData

    ply_data = PlyData.read(ply_path).elements[0]
    return np.stack([ply_data.data[name] for name in ['x', 'y', 'z']], axis=1)


def voxelize(means):
    means_min, means_max = means.min(axis=0), means.max(axis=0)
    voxelized_means = (means - means_min) / (means_max - means_min)  # normalize to [0, 1]
    voxelized_means = np.round(voxelized_means * (2 ** VOXELIZE_SCALE_FACTOR - 1))
    return voxelized_means, means_min, means_max


def devoxelize(voxelized_means, means_min, means_max):
    voxelized_means = voxelized_means.astype(np.float32)
    means_min = means_min.astype(np.float32)
    means_max = means_max.astype(np.float32)
    return voxelized_means / (2 ** VOXELIZE_SCALE_FACTOR - 1) * (means_max - means_min) + means_min


def sorted_voxels(voxelized_means, other_params=None):
    """Official ordering (lexicographic key x + y*B + z*B^2 with B = max+1)."""
    indices_sorted = np.argsort(voxelized_means @ np.power(voxelized_means.max() + 1, np.arange(voxelized_means.shape[1])), axis=0)
    voxelized_means = voxelized_means[indices_sorted]
    if other_params is None:
        return voxelized_means
    other_params = other_params[indices_sorted]
    return voxelized_means, other_params


def sorted_orig_voxels(means, other_params=None):
    means = means.detach().cpu().numpy().astype(np.float32)
    voxelized_means, means_min, means_max = voxelize(means=means)
    voxelized_means, other_params = sorted_voxels(voxelized_means=voxelized_means, other_params=other_params)
    means = devoxelize(voxelized_means=voxelized_means, means_min=means_min, means_max=means_max)
    means = torch.from_numpy(means).cuda().to(torch.float32)
    return means, other_params


def write_binary_data(dst_file_handle, src_bin_path):
    with open(src_bin_path, 'rb') as f:
        data = f.read()
        dst_file_handle.write(np.array([len(data), ], dtype=np.uint32).tobytes())  # 4 bytes for length
        dst_file_handle.write(data)


def read_binary_data(dst_bin_path, src_file_handle):
    length = int(np.frombuffer(src_file_handle.read(4), dtype=np.uint32)[0])
    with open(dst_bin_path, 'wb') as f:
        f.write(src_file_handle.read(length))


# ---- substitute lossless octree coder --------------------------------------------------------

def _spread(v):
    """Insert two zero bits between the 16 low bits of v (uint64) for Morton interleaving."""
    v = v.astype(np.uint64) & np.uint64(0xFFFF)
    v = (v | (v << np.uint64(32))) & np.uint64(0x1F00000000FFFF)
    v = (v | (v << np.uint64(16))) & np.uint64(0x1F0000FF0000FF)
    v = (v | (v << np.uint64(8))) & np.uint64(0x100F00F00F00F00F)
    v = (v | (v << np.uint64(4))) & np.uint64(0x10C30C30C30C30C3)
    v = (v | (v << np.uint64(2))) & np.uint64(0x1249249249249249)
    return v


def _compact(v):
    v = v.astype(np.uint64) & np.uint64(0x1249249249249249)
    v = (v ^ (v >> np.uint64(2))) & np.uint64(0x10C30C30C30C30C3)
    v = (v ^ (v >> np.uint64(4))) & np.uint64(0x100F00F00F00F00F)
    v = (v ^ (v >> np.uint64(8))) & np.uint64(0x1F0000FF0000FF)
    v = (v ^ (v >> np.uint64(16))) & np.uint64(0x1F00000000FFFF)
    v = (v ^ (v >> np.uint64(32))) & np.uint64(0xFFFF)
    return v


def morton(vox):
    vox = vox.astype(np.uint64)
    return _spread(vox[:, 0]) | (_spread(vox[:, 1]) << np.uint64(1)) | (_spread(vox[:, 2]) << np.uint64(2))


def unmorton(code):
    return np.stack([_compact(code), _compact(code >> np.uint64(1)), _compact(code >> np.uint64(2))],
                    axis=1).astype(np.float32)


def octree_encode(vox):
    """Lossless bytes for integer voxels in [0, 2^16) (duplicates kept)."""
    codes, counts = np.unique(morton(vox), return_counts=True)  # sorted Morton order
    occupancy = []
    for depth in range(LEVELS):  # depth d: nodes with 3*(16-d) low bits removed
        child = codes >> np.uint64(3 * (LEVELS - depth - 1))
        child = child[np.concatenate([[True], child[1:] != child[:-1]])]
        parent = child >> np.uint64(3)
        starts = np.flatnonzero(np.concatenate([[True], parent[1:] != parent[:-1]]))
        bits = (np.uint8(1) << (child & np.uint64(7)).astype(np.uint8)).astype(np.uint8)
        occupancy.append(np.bitwise_or.reduceat(bits, starts).astype(np.uint8))
    occ = np.concatenate(occupancy).tobytes()
    dup = counts - 1
    dup_bytes = b"" if not dup.any() else _varint(dup)
    head = np.array([len(codes), len(vox), len(occ)], dtype=np.uint64).tobytes()
    occ_z = lzma.compress(occ, preset=9 | lzma.PRESET_EXTREME)
    dup_z = lzma.compress(dup_bytes, preset=9) if dup_bytes else b""
    return head + np.array([len(occ_z)], dtype=np.uint64).tobytes() + occ_z + dup_z


def octree_decode(data):
    n_unique, n_total, n_occ, n_occ_z = np.frombuffer(data[:32], dtype=np.uint64)
    occ = np.frombuffer(lzma.decompress(data[32:32 + int(n_occ_z)]), dtype=np.uint8)
    assert len(occ) == int(n_occ)
    nodes = np.zeros(1, dtype=np.uint64)
    pos = 0
    for _ in range(LEVELS):
        block = occ[pos:pos + len(nodes)]
        pos += len(nodes)
        bits = np.unpackbits(block[:, None], axis=1, bitorder="little")  # [n, 8], bit b = child b
        rows, cols = np.nonzero(bits)
        nodes = (nodes[rows] << np.uint64(3)) | cols.astype(np.uint64)
    assert pos == len(occ) and len(nodes) == int(n_unique)
    rest = data[32 + int(n_occ_z):]
    if rest:
        dup = _unvarint(lzma.decompress(rest), len(nodes))
        nodes = np.repeat(nodes, dup + 1)
    assert len(nodes) == int(n_total)
    return unmorton(nodes)


def _varint(values):
    out = bytearray()
    for v in values.tolist():
        while v >= 0x80:
            out.append((v & 0x7F) | 0x80)
            v >>= 7
        out.append(v)
    return bytes(out)


def _unvarint(data, n):
    out = np.zeros(n, dtype=np.int64)
    k = shift = value = 0
    for b in data:
        value |= (b & 0x7F) << shift
        if b & 0x80:
            shift += 7
        else:
            out[k] = value
            k += 1
            value = shift = 0
    assert k == n
    return out


# ---- official container (compress_gaussian_params / decompress_gaussian_params) ------------

def compress_gaussian_params(gaussian_params, bin_path, coder="gpcc"):
    coder = resolve_coder(coder)
    if isinstance(gaussian_params, torch.Tensor):
        gaussian_params = gaussian_params.detach().cpu().numpy()
    voxelized_means, means_min, means_max = voxelize(means=gaussian_params)
    voxelized_means = sorted_voxels(voxelized_means=voxelized_means, other_params=None)
    with TemporaryDirectory() as temp_dir:
        means_bin_path = os.path.join(temp_dir, 'compressed.bin')
        if coder == "gpcc":
            ply_path = os.path.join(temp_dir, 'voxelized_means.ply')
            write_ply_geo_ascii(geo_data=voxelized_means, ply_path=ply_path)
            gpcc_encode(encoder_path=tmc3_path(), ply_path=ply_path, bin_path=means_bin_path)
        else:
            with open(means_bin_path, 'wb') as f:
                f.write(octree_encode(voxelized_means))
        with open(bin_path, 'wb') as f:
            f.write(np.array([means_min, means_max], dtype=np.float32).tobytes())  # 24 bytes
            f.write(np.array([0 if coder == "gpcc" else 1], dtype=np.uint8).tobytes())
            write_binary_data(dst_file_handle=f, src_bin_path=means_bin_path)
        file_size = {
            'means': os.path.getsize(means_bin_path) / 1024 / 1024,  # MB
            # the 1-byte coder tag is ours, not part of the official container
            'total': (os.path.getsize(bin_path) - 1) / 1024 / 1024,  # MB
        }
    compress_results = {'num_gaussians': voxelized_means.shape[0], 'file_size': file_size,
                        'coder': coder}
    return None, voxelized_means, means_min, means_max, compress_results


def decompress_gaussian_params(bin_path, coder="gpcc"):
    with TemporaryDirectory() as temp_dir:
        with open(bin_path, 'rb') as f:
            head_info = np.frombuffer(f.read(24), dtype=np.float32)
            means_min, means_max = head_info[:3], head_info[3:]
            tag = int(np.frombuffer(f.read(1), dtype=np.uint8)[0])
            means_bin_path = os.path.join(temp_dir, 'compressed.bin')
            read_binary_data(dst_bin_path=means_bin_path, src_file_handle=f)
        if tag == 0:
            ply_path = os.path.join(temp_dir, 'voxelized_means.ply')
            gpcc_decode(decoder_path=tmc3_path(), bin_path=means_bin_path, recon_path=ply_path)
            voxelized_means = read_ply_geo_bin(ply_path=ply_path).astype(np.float32)
        else:
            with open(means_bin_path, 'rb') as f:
                voxelized_means = octree_decode(f.read())
        voxelized_means = sorted_voxels(voxelized_means)  # decoded order differs: sort (official)
        means_dec = devoxelize(voxelized_means=voxelized_means, means_min=means_min, means_max=means_max)
        means_dec = torch.from_numpy(means_dec).cuda()
    return means_dec, voxelized_means, means_min, means_max
