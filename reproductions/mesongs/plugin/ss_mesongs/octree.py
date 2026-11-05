"""C2 octree geometry coding (official `scene/gaussian_model.py`: `octreecodes`,
`create_octree_overall`, `decode_oct`, `d1halfing_fast`).

Keys: per axis `searchsorted(linspace(min, max, 2^d + 1), x, 'right') - 1` (the maximum lands on
2^d, as in the official code), `ki = x * 4^d + y * 2^d + z`, Gaussians with equal keys are merged
by the mean of all attributes (official `merge_type='mean'`; float64 sums here instead of a Python
loop of float32 np.mean), output sorted by key. Decoding returns the lower voxel corner
`linspace[code]` (the official decode, also used for the fine-tune positions).
"""

import numpy as np


def d1halfing_fast(pmin, pmax, pdepth):
    return np.linspace(pmin, pmax, 2 ** int(pdepth) + 1)


def octree_codes(points, depth):
    """Official key computation; returns (ki [N] int64, paramarr [6] (min/max per axis))."""
    mins = [np.amin(points[:, i]) for i in range(3)]
    maxs = [np.amax(points[:, i]) for i in range(3)]
    codes = []
    for i in range(3):
        letra = d1halfing_fast(mins[i], maxs[i], depth)
        codes.append(np.searchsorted(letra, points[:, i], side="right") - 1)
    ki = codes[0] * (2 ** (depth * 2)) + codes[1] * (2**depth) + codes[2]
    paramarr = np.asarray([mins[0], maxs[0], mins[1], maxs[1], mins[2], maxs[2]])
    return ki.astype(np.int64), paramarr


def merge_mean(ki, features):
    """Mean of `features` [N, F] per unique key; returns (sorted unique keys, merged, inverse).

    Sorted segments + np.add.reduceat in float64 (np.add.at is too slow at millions of rows)."""
    order = np.argsort(ki, kind="stable")
    sk = ki[order]
    keys, start, counts = np.unique(sk, return_index=True, return_counts=True)
    sums = np.add.reduceat(features[order].astype(np.float64), start, axis=0)
    inverse = np.empty(len(ki), dtype=np.int64)
    inverse[order] = np.repeat(np.arange(len(keys)), counts)
    return keys, (sums / counts[:, None]).astype(np.float32), inverse


def keys_to_voxels(oct_keys, depth):
    """Official integer decode of keys -> V [N,3] (x, y, z codes)."""
    ox = (oct_keys / (2 ** (depth * 2))).astype(int)
    oy = ((oct_keys - ox * (2 ** (depth * 2))) / (2**depth)).astype(int)
    oz = (oct_keys - ox * (2 ** (depth * 2)) - oy * (2**depth)).astype(int)
    return np.array([ox, oy, oz], dtype=int).T


def decode_oct(paramarr, oct_keys, depth):
    """Official decode_oct: lower voxel corners [N,3] float64 and voxel codes V."""
    V = keys_to_voxels(oct_keys, depth)
    pts = [d1halfing_fast(paramarr[2 * i], paramarr[2 * i + 1], depth)[V[:, i]] for i in range(3)]
    return np.array(pts).T, V


def voxelize(points, features, depth):
    """create_octree_overall + decode: (positions [M,3], features [M,F], V [M,3], keys, params,
    inverse [N] = voxel index of each input Gaussian)."""
    ki, paramarr = octree_codes(points, depth)
    keys, merged, inverse = merge_mean(ki, features)
    xyz, V = decode_oct(paramarr, keys, depth)
    return xyz, merged, V, keys, paramarr, inverse


def octree_codes_reference(points, features, depth):
    """Line-by-line port of the official `octreecodes(..., merge_type='mean')` (tests/probes)."""
    ppoints = np.concatenate([points, features], -1)
    minx, maxx = np.amin(ppoints[:, 0]), np.amax(ppoints[:, 0])
    miny, maxy = np.amin(ppoints[:, 1]), np.amax(ppoints[:, 1])
    minz, maxz = np.amin(ppoints[:, 2]), np.amax(ppoints[:, 2])
    xletra = d1halfing_fast(minx, maxx, depth)
    yletra = d1halfing_fast(miny, maxy, depth)
    zletra = d1halfing_fast(minz, maxz, depth)
    otcodex = np.searchsorted(xletra, ppoints[:, 0], side="right") - 1
    otcodey = np.searchsorted(yletra, ppoints[:, 1], side="right") - 1
    otcodez = np.searchsorted(zletra, ppoints[:, 2], side="right") - 1
    ki = otcodex * (2 ** (depth * 2)) + otcodey * (2**depth) + otcodez
    ki_ranks = np.argsort(ki)
    ppoints = ppoints[ki_ranks]
    ki = ki[ki_ranks]
    dedup = np.split(ppoints[:, 3:], np.unique(ki, return_index=True)[1][1:])
    final = np.concatenate([np.mean(d, 0).reshape(1, -1) for d in dedup], 0)
    return np.unique(ki), final, np.asarray([minx, maxx, miny, maxy, minz, maxz])
