"""C4 region adaptive hierarchical transform (RAHT).

Ported from the official MesonGS `raht_torch.py` (itself derived from 3DAC,
https://fatpeter.github.io/, and https://github.com/digitalivp/RAHT/tree/reorder; MesonGS code
under the Gaussian-Splatting licence). `copy_and_sort`, `haar3d_param`, `get_raht_tree` and
`inv_haar3d_param` keep the official numerics; the `*_reference` functions are line-by-line ports
(np.setdiff1d) used by tests and probes, the default functions replace the set differences by
equivalent boolean masks (both operands are sorted, unique index arrays), which is much faster at
millions of points.

The transform itself (official `ft_render`): forward `C = rf[reorder]` then, for each of the
3*depth levels, `C[l], C[r] = w1 C[l] + w2 C[r], -w2 C[l] + w1 C[r]`; the DC coefficient ends in
C[0]. The inverse runs the official top-down loop with `pos`, `trans` copies and `CT[:S] = OC[:S]`.
Both are orthonormal linear maps with I(F(x)) = x, hence each one's adjoint is the other; the
autograd Functions below use that (no per-level saved tensors).
"""

import numpy as np
import torch

MASK_LSB = np.uint64(0xFFFFFFFFFFFFFFFE)


def morton_val(V):
    """Official copyAsort bit interleave (20 bits per axis; vx = V[:, 2], vz = V[:, 0])."""
    V = V.astype(np.uint64)
    vx, vy, vz = V[:, 2], V[:, 1], V[:, 0]
    val = np.zeros(len(V), dtype=np.uint64)
    for b in range(20):
        m = np.uint64(1 << b)
        val += ((m & vx) << np.uint64(2 * b)) + ((m & vy) << np.uint64(2 * b + 1)) + (
            (m & vz) << np.uint64(2 * b + 2))
    return val


def copy_and_sort(V):
    """Official copyAsort: (W ones, sorted morton val, argsort order)."""
    W = np.ones(V.shape[0])
    val = morton_val(V)
    reord = np.argsort(val)
    val = np.sort(val).astype(np.uint64)
    return W, val, reord


def _level_masks(val, S, fast):
    temp = val.astype(np.uint64) & MASK_LSB
    mask = temp[:-1] == temp[1:]
    mask = np.concatenate((mask, [False]))
    comb = np.where(mask)[0]
    if fast:
        prev = np.concatenate(([False], mask[:-1]))  # i in comb + 1  <=>  mask[i - 1]
        trans = np.where(~mask & ~prev)[0]
        idxT = np.where(~prev[:S])[0]
    else:
        trans = np.where(~mask)[0]
        trans = np.setdiff1d(trans, comb + 1)
        idxT = np.setdiff1d(np.arange(S), comb + 1)
    maskT = mask[idxT]
    return mask, comb, trans, idxT, maskT


def _haar3d_param(depth, w, val, fast):
    N = val.shape[0]
    NN = N
    depth *= 3
    wT = np.zeros((N,)).astype(np.uint64)
    posT = np.zeros((N,)).astype(np.int64)
    pos = np.arange(N)
    iW1, iW2, iLeft, iRight = [], [], [], []
    for _ in range(depth):
        S = N
        mask, comb, trans, idxT, maskT = _level_masks(val, S, fast)
        wT[np.where(~maskT)[0]] = w[trans]
        wT[np.where(maskT)[0]] = w[comb] + w[comb + 1]
        posT[np.where(~maskT)[0]] = pos[trans]
        posT[np.where(maskT)[0]] = pos[comb]
        left, right = comb, comb + 1
        a = np.sqrt((w[left]) + (w[right]))
        iW1.append(np.sqrt((w[left])) / a)
        iW2.append(np.sqrt((w[right])) / a)
        iLeft.append(pos[left] + 0)
        iRight.append(pos[right] + 0)
        valT = (val >> np.uint64(1))[idxT]
        N_T = N
        N = N - comb.shape[0]
        N_idx = np.arange(N_T, N, -1) - NN - 1
        wT[N_idx] = wT[np.where(maskT)[0]]
        posT[N_idx] = pos[comb + 1]
        pos[N:S] = posT[N:S]
        w[N:S] = wT[N:S]
        val, valT = valT, val
        pos, posT = posT, pos
        w, wT = wT, w
    outW = np.zeros(w.shape)
    outW[pos] = w
    return {"w": outW, "iW1": iW1, "iW2": iW2, "iLeft_idx": iLeft, "iRight_idx": iRight}


def haar3d_param(depth, w, val):
    return _haar3d_param(depth, w.copy(), val.copy(), True)


def haar3d_param_reference(depth, w, val):
    return _haar3d_param(depth, w.copy(), val.copy(), False)


def _get_raht_tree(inV, depth, fast, keep_levels=None):
    """Official get_RAHT_tree (3*depth levels). `keep_levels` stores iVAL/iW/iM only for the
    first levels (the official inverse calls this with an already tripled depth, i.e. 9*depth
    levels, but reads only the first 3*depth; `pos` still comes from all levels)."""
    N, _ = inV.shape
    NN = N
    depth *= 3
    keep = depth if keep_levels is None else min(keep_levels, depth)
    wT = np.zeros((N,))
    posT = np.zeros((N,)).astype(np.uint64)
    iVAL = np.zeros((keep, N)).astype(np.uint64)
    iW = np.zeros((keep, N))
    M = N
    iM = np.zeros((keep,)).astype(np.uint64)
    w, val, reord = copy_and_sort(inV)
    pos = np.arange(N).astype(np.uint64)
    for d in range(depth):
        if d < keep:
            iVAL[d, :M] = val[:M]
            iW[d, :M] = w[:M]
            iM[d] = M
        S = N
        mask, comb, trans, idxT, maskT = _level_masks(val, S, fast)
        wT[np.where(~maskT)[0]] = w[trans]
        posT[np.where(~maskT)[0]] = pos[trans]
        wT[np.where(maskT)[0]] = w[comb] + w[comb + 1]
        posT[np.where(maskT)[0]] = pos[comb]
        valT = (val >> np.uint64(1))[idxT]
        N_T = N
        N = N - comb.shape[0]
        M = N
        N_idx = np.arange(N_T, N, -1) - NN - 1
        wT[N_idx] = wT[np.where(maskT)[0]]
        posT[N_idx] = pos[comb + 1]
        pos[N:S] = posT[N:S]
        w[N:S] = wT[N:S]
        val, valT = valT, val
        pos, posT = posT, pos
        w, wT = wT, w
    return {"reord": reord, "pos": pos, "iVAL": iVAL, "iW": iW, "iM": iM}


def _inv_haar3d_param(inV, depth, fast):
    NN = inV.shape[0]
    depth *= 3
    # official: depth *= 3; get_RAHT_tree(inV, depth) -> 9 * depth tree levels
    tree = _get_raht_tree(inV, depth, fast, keep_levels=depth)
    pos, iVAL, iW, iM = tree["pos"], tree["iVAL"], tree["iW"], tree["iM"]
    out = {k: [] for k in ("iW1", "iW2", "iS", "iLeft_idx", "iRight_idx", "iLeft_idx_CT",
                           "iRight_idx_CT", "iTrans_idx", "iTrans_idx_CT")}
    d = depth
    while d:
        d = d - 1
        S = iM[d]
        val, w = iVAL[d, : int(S)], iW[d, : int(S)]
        N = int(S)
        mask, comb, trans, idxT, maskT = _level_masks(val, int(S), fast)
        N_T = N
        N = N - comb.shape[0]
        N_idx = np.arange(N_T, N, -1) - NN - 1
        left, right = comb, comb + 1
        a = np.sqrt((w[left]) + (w[right]))
        out["iW1"].append(np.sqrt((w[left])) / a)
        out["iW2"].append(np.sqrt((w[right])) / a)
        out["iLeft_idx"].append(left.astype(int) + 0)
        out["iRight_idx"].append(right.astype(int) + 0)
        out["iLeft_idx_CT"].append(np.where(maskT)[0].astype(int))
        out["iRight_idx_CT"].append(N_idx.astype(int))
        out["iTrans_idx"].append(trans)
        out["iTrans_idx_CT"].append(np.where(~maskT)[0])
        out["iS"].append(S)
    out["pos"] = pos
    return out


def inv_haar3d_param(inV, depth):
    return _inv_haar3d_param(inV, depth, True)


def inv_haar3d_param_reference(inV, depth):
    return _inv_haar3d_param(inV, depth, False)


class RAHTTables:
    """Official RAHT bookkeeping of a voxelised geometry V [N,3] (int) at `depth`, on `device`."""

    def __init__(self, V, depth, device):
        self.n = len(V)
        self.depth = depth
        w, val, reorder = copy_and_sort(V)
        fwd = haar3d_param(depth, w, val)
        inv = inv_haar3d_param(V, depth)

        def t(x, dtype=torch.long):
            return torch.as_tensor(np.asarray(x), dtype=dtype, device=device)

        self.reorder = t(reorder.astype(np.int64))
        self.fwd = [(t(a, torch.float32)[:, None], t(b, torch.float32)[:, None], t(lf), t(r))
                    for a, b, lf, r in zip(fwd["iW1"], fwd["iW2"], fwd["iLeft_idx"],
                                          fwd["iRight_idx"])]
        self.pos = t(inv["pos"].astype(np.int64))
        self.inv = [
            (t(a, torch.float32)[:, None], t(b, torch.float32)[:, None], int(S), t(lf), t(r),
             t(lc), t(rc), t(tr.astype(np.int64)), t(tc.astype(np.int64)))
            for a, b, S, lf, r, lc, rc, tr, tc in zip(
                inv["iW1"], inv["iW2"], inv["iS"], inv["iLeft_idx"], inv["iRight_idx"],
                inv["iLeft_idx_CT"], inv["iRight_idx_CT"], inv["iTrans_idx"],
                inv["iTrans_idx_CT"])
        ]
        # The official inverse tree has 9*depth levels; with two or more voxel codes on the
        # 2^depth edge (searchsorted maximum) the official forward/inverse pair stops being an
        # exact round trip for those rows. Count them (the adjoint backward assumes I(F(x)) = x).
        g = torch.Generator(device="cpu").manual_seed(0)
        x = torch.randn(self.n, 2, generator=g).to(device)
        err = (inverse_transform(forward_transform(x, self), self) - x).abs().amax(1)
        self.roundtrip_bad_rows = int((err > 1e-3 * (1 + x.abs().amax(1))).sum())


@torch.no_grad()
def forward_transform(rf, tables):
    """Official forward loop: point-order attributes [N,C] -> RAHT coefficients (C[0] = DC)."""
    C = rf[tables.reorder].clone()
    for w1, w2, left, right in tables.fwd:
        cl, cr = C[left], C[right]
        C[left] = w1 * cl + w2 * cr
        C[right] = -w2 * cl + w1 * cr
    return C


@torch.no_grad()
def inverse_transform(coef, tables):
    """Official inverse loop: coefficients [N,C] -> point-order attributes."""
    CT = coef[tables.pos].clone()
    OC = torch.zeros_like(coef)
    for w1, w2, S, left, right, left_ct, right_ct, trans, trans_ct in tables.inv:
        OC[trans] = CT[trans_ct]
        a, b = CT[left_ct], CT[right_ct]
        OC[left] = w1 * a - w2 * b
        OC[right] = w2 * a + w1 * b
        CT[:S] = OC[:S]
    out = torch.zeros_like(coef)
    out[tables.reorder] = OC
    return out


class _Forward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, rf, tables):
        ctx.tables = tables
        return forward_transform(rf, tables)

    @staticmethod
    def backward(ctx, g):
        return inverse_transform(g.contiguous(), ctx.tables), None


class _Inverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, coef, tables):
        ctx.tables = tables
        return inverse_transform(coef, tables)

    @staticmethod
    def backward(ctx, g):
        return forward_transform(g.contiguous(), ctx.tables), None


def raht(rf, tables):
    """Differentiable forward RAHT (adjoint = inverse, the transform is orthonormal)."""
    return _Forward.apply(rf, tables)


def iraht(coef, tables):
    """Differentiable inverse RAHT (adjoint = forward)."""
    return _Inverse.apply(coef, tables)


def forward_transform_functional(rf, tables):
    """Out-of-place version of the official loop for autograd cross-checks (probes/tests)."""
    C = rf[tables.reorder]
    for w1, w2, left, right in tables.fwd:
        cl, cr = C[left], C[right]
        C = C.index_copy(0, left, w1 * cl + w2 * cr).index_copy(0, right, -w2 * cl + w1 * cr)
    return C


def inverse_transform_functional(coef, tables):
    CT = coef[tables.pos]
    OC = torch.zeros_like(coef)
    for w1, w2, S, left, right, left_ct, right_ct, trans, trans_ct in tables.inv:
        OC = OC.index_copy(0, trans, CT[trans_ct])
        a, b = CT[left_ct], CT[right_ct]
        OC = OC.index_copy(0, left, w1 * a - w2 * b).index_copy(0, right, w2 * a + w1 * b)
        CT = torch.cat((OC[:S], CT[S:]), 0)
    return torch.zeros_like(coef).index_copy(0, tables.reorder, OC)
