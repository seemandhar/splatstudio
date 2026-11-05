"""C2 sensitivity-aware vector clustering (official `compression/vq.py`, `utils/splats.py`).

Nearest-centroid search uses the official `weighted_distance` CUDA kernel (csrc/, JIT-built;
plain squared L2 with a first-minimum loop: the per-vector weight S(x) of the paper's distance
does not change the argmin). `torch_scatter.scatter(..., reduce="sum")` is replaced by
`index_add_` (same sums). `matrix_to_quaternion` / `_sqrt_positive_part` are the PyTorch3D
functions the official code vendors.
"""

from pathlib import Path

import torch
import torch.nn.functional as F

CSRC = Path(__file__).parent / "csrc"
_EXT = None
EPS = 1e-5  # official VectorQuantize.eps


def extension():
    global _EXT
    if _EXT is None:
        from torch.utils.cpp_extension import load

        _EXT = load(
            name="ss_c3dgs_weighted_distance",
            sources=[str(CSRC / "weighted_distance.cu"), str(CSRC / "ext.cpp")],
            extra_cuda_cflags=["-O3"],
            extra_cflags=["-O3"],
            verbose=False,
        )
    return _EXT


def nearest(x, codebook):
    """(min squared distance, index) per row of x — official weightedDistance kernel."""
    if not x.is_cuda:
        raise ValueError("the official nearest-centroid kernel needs CUDA tensors")
    return extension().weightedDistance(x.float().contiguous(), codebook.float().contiguous())


def nearest_reference(x, codebook, chunk=4096):
    """Brute-force float64 reference (probes / CPU tests): first minimum on ties."""
    dists, idx = [], []
    c = codebook.double()
    for s in range(0, len(x), chunk):
        d = (x[s : s + chunk].double()[:, None, :] - c[None]).square().sum(-1)
        m, i = d.min(dim=1)
        dists.append(m.float())
        idx.append(i)
    return torch.cat(dists), torch.cat(idx)


def ema_inplace(moving_avg, new, decay):
    moving_avg.mul_(decay).add_(new, alpha=(1 - decay))


@torch.no_grad()
def update(codebook, x, importance, decay, assign=nearest):
    """Official VectorQuantize.update (codebook in place); returns the batch min distances.

    new_k = sum_{x in A(k)} S(x) x / (sum_{x in A(k)} S(x) + eps)  (paper centroid equation)
    c_k <- decay c_k + (1 - decay) new_k  (mini-batch EMA; empty clusters decay towards 0)
    """
    min_dists, idx = assign(x, codebook)
    k = codebook.shape[0]
    acc = torch.zeros(k, device=x.device, dtype=x.dtype).index_add_(0, idx, importance)
    sums = torch.zeros_like(codebook).index_add_(0, idx, x * importance[:, None])
    ema_inplace(codebook, sums / (acc[:, None] + EPS), decay)
    return min_dists


@torch.no_grad()
def vq_features(features, importance, codebook_size, batch_size, steps, decay=0.8,
                scale_normalize=False, generator=None, assign=nearest):
    """Official vq_features: uniform init in [min, max] of all values, `steps` mini-batch
    updates of `batch_size` random vectors (with replacement), optional trace re-normalisation
    of covariance codebooks (entries 0, 3, 5) after every step, final assignment."""
    importance_n = importance / importance.max()
    lo, hi = features.aminmax()
    codebook = torch.rand(codebook_size, features.shape[-1], device=features.device,
                          generator=generator) * (hi - lo) + lo
    for _ in range(steps):
        batch = torch.randint(0, features.shape[0], (batch_size,), device=features.device,
                              generator=generator)
        update(codebook, features[batch], importance_n[batch], decay, assign)
        if scale_normalize:
            tr = codebook[:, [0, 3, 5]].sum(-1)
            codebook /= tr[:, None]
    _, indices = assign(features, codebook)
    return codebook, indices


def join_features(all_features, keep_mask, codebook, codebook_indices):
    """Official join_features: kept (high-sensitivity) vectors are appended to the codebook."""
    keep_features = all_features[keep_mask]
    compressed = torch.cat([codebook, keep_features], 0)
    indices = torch.zeros(len(all_features), dtype=torch.long, device=all_features.device)
    indices[~keep_mask] = codebook_indices
    indices[keep_mask] = torch.arange(len(keep_features), device=indices.device) + len(codebook)
    return compressed, indices


def quat_to_matrix(q):
    """3DGS build_rotation (wxyz, normalised inside)."""
    q = F.normalize(q, dim=-1)
    r, x, y, z = q.unbind(-1)
    return torch.stack(
        (
            1 - 2 * (y * y + z * z), 2 * (x * y - r * z), 2 * (x * z + r * y),
            2 * (x * y + r * z), 1 - 2 * (x * x + z * z), 2 * (y * z - r * x),
            2 * (x * z - r * y), 2 * (y * z + r * x), 1 - 2 * (x * x + y * y),
        ),
        -1,
    ).reshape(q.shape[:-1] + (3, 3))


TRI = ([0, 0, 0, 1, 1, 2], [0, 1, 2, 1, 2, 2])  # 3DGS strip_symmetric order (= gsplat covars)
FULL = [[0, 1, 2], [1, 3, 4], [2, 4, 5]]


def covariance(scales, quats):
    """3DGS build_covariance_from_scaling_rotation(strip_sym=True): (R S)(R S)^T, 6 values."""
    L = quat_to_matrix(quats) * scales[..., None, :]
    cov = L @ L.transpose(-1, -2)
    return cov[..., TRI[0], TRI[1]]


def to_full_cov(cov6):
    """Official to_full_cov (6 -> symmetric 3x3)."""
    return cov6[..., torch.tensor(FULL, device=cov6.device)]


def _sqrt_positive_part(x):
    ret = torch.zeros_like(x)
    positive = x > 0
    ret[positive] = torch.sqrt(x[positive])
    return ret


def matrix_to_quaternion(matrix):
    """PyTorch3D matrix_to_quaternion as vendored by the official utils/splats.py (wxyz)."""
    batch_dim = matrix.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(
        matrix.reshape(batch_dim + (9,)), dim=-1
    )
    q_abs = _sqrt_positive_part(
        torch.stack(
            [
                1.0 + m00 + m11 + m22,
                1.0 + m00 - m11 - m22,
                1.0 - m00 + m11 - m22,
                1.0 - m00 - m11 + m22,
            ],
            dim=-1,
        )
    )
    quat_by_rijk = torch.stack(
        [
            torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
            torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
            torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1),
            torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1),
        ],
        dim=-2,
    )
    flr = torch.tensor(0.1).to(dtype=q_abs.dtype, device=q_abs.device)
    quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].max(flr))
    return quat_candidates[F.one_hot(q_abs.argmax(dim=-1), num_classes=4) > 0.5, :].reshape(
        batch_dim + (4,)
    )


def extract_rot_scale(cov):
    """Official extract_rot_scale: eigh(cov + 1e-8 I, 'U'), scale = sqrt(eigvals) (nan -> 1e-6),
    rotation = normalize(matrix_to_quaternion(R * det R))."""
    S, R = torch.linalg.eigh(cov + torch.eye(3, device=cov.device) * 1e-8, UPLO="U")
    scaling = S.sqrt().nan_to_num(nan=1e-6)
    rot = F.normalize(matrix_to_quaternion(R * R.det()[..., None, None]))
    return rot, scaling
