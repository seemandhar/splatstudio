"""Multi-resolution hash-grid encoding with tiny-cuda-nn semantics (CUDA kernel + torch reference).

Copied unchanged (except the JIT extension name) from the ConeGS reproduction
(reproductions/conegs/plugin/ss_conegs/hashgrid.py, SPLATIFY).

tiny-cuda-nn is not installed in the shared env, so the encoding used by the official ConeGS iNGP
(`tcnn.NetworkWithInputEncoding`, HashGrid, 2 features/level, linear interpolation) is
re-implemented here. Numerics follow `tiny-cuda-nn/include/tiny-cuda-nn/common_device.h`
(`grid_scale`, `grid_resolution`, `grid_index`, `coherent_prime_hash`) and `encodings/grid.h`
(`pos_fract`, per-level parameter counts rounded up to 8, uniform +-1e-4 init):

    scale_l = 2^(l * log2(b)) * N_min - 1,   res_l = ceil(scale_l) + 1
    pos = x * scale_l + 0.5,  cell = floor(pos),  frac = pos - cell
    index = dense (x + y res + z res^2) if res^3 <= T_l, else x ^ y*2654435761 ^ z*805459861
    index %= T_l     (uint32 arithmetic)
"""

import math

import torch

F_PER_LEVEL = 2
PRIME_Y = 2654435761
PRIME_Z = 805459861
MAX_DENSE_RES = 0x659  # tcnn MAX_BASES[3]: larger grids are always hashed (stride overflow)

_CUDA_SRC = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>

__device__ __forceinline__ uint32_t grid_index(uint32_t x, uint32_t y, uint32_t z,
                                               uint32_t res, uint32_t size) {
    uint32_t index = 0;
    uint32_t stride = 0xFFFFFFFFu;
    if (res <= 0x659u) {
        index = x + y * res + z * res * res;
        stride = res * res * res;
    }
    if (size < stride) {
        index = x ^ (y * 2654435761u) ^ (z * 805459861u);
    }
    return index % size;
}

__global__ void hash_fwd_kernel(const float* __restrict__ pos, const float* __restrict__ params,
                                const int32_t* __restrict__ offsets,
                                const float* __restrict__ scales,
                                const int32_t* __restrict__ resolutions,
                                float* __restrict__ out, int64_t n, int n_levels) {
    int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n * n_levels) return;
    int level = (int)(tid / n);
    int64_t i = tid - (int64_t)level * n;
    float scale = scales[level];
    uint32_t res = (uint32_t)resolutions[level];
    uint32_t size = (uint32_t)(offsets[level + 1] - offsets[level]);
    const float* grid = params + (int64_t)offsets[level] * 2;
    float p[3];
    uint32_t g[3];
    for (int d = 0; d < 3; ++d) {
        float v = pos[i * 3 + d] * scale + 0.5f;
        float f = floorf(v);
        g[d] = (uint32_t)(int)f;
        p[d] = v - f;
    }
    float r0 = 0.f, r1 = 0.f;
    for (int c = 0; c < 8; ++c) {
        float w = 1.f;
        uint32_t q[3];
        for (int d = 0; d < 3; ++d) {
            if ((c & (1 << d)) == 0) { w *= 1.f - p[d]; q[d] = g[d]; }
            else { w *= p[d]; q[d] = g[d] + 1; }
        }
        uint32_t idx = grid_index(q[0], q[1], q[2], res, size);
        r0 += w * grid[(int64_t)idx * 2];
        r1 += w * grid[(int64_t)idx * 2 + 1];
    }
    out[i * (n_levels * 2) + level * 2] = r0;
    out[i * (n_levels * 2) + level * 2 + 1] = r1;
}

__global__ void hash_bwd_kernel(const float* __restrict__ pos, const float* __restrict__ grad_out,
                                const int32_t* __restrict__ offsets,
                                const float* __restrict__ scales,
                                const int32_t* __restrict__ resolutions,
                                float* __restrict__ grad_params, int64_t n, int n_levels) {
    int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n * n_levels) return;
    int level = (int)(tid / n);
    int64_t i = tid - (int64_t)level * n;
    float g0 = grad_out[i * (n_levels * 2) + level * 2];
    float g1 = grad_out[i * (n_levels * 2) + level * 2 + 1];
    if (g0 == 0.f && g1 == 0.f) return;
    float scale = scales[level];
    uint32_t res = (uint32_t)resolutions[level];
    uint32_t size = (uint32_t)(offsets[level + 1] - offsets[level]);
    float* grid = grad_params + (int64_t)offsets[level] * 2;
    float p[3];
    uint32_t g[3];
    for (int d = 0; d < 3; ++d) {
        float v = pos[i * 3 + d] * scale + 0.5f;
        float f = floorf(v);
        g[d] = (uint32_t)(int)f;
        p[d] = v - f;
    }
    for (int c = 0; c < 8; ++c) {
        float w = 1.f;
        uint32_t q[3];
        for (int d = 0; d < 3; ++d) {
            if ((c & (1 << d)) == 0) { w *= 1.f - p[d]; q[d] = g[d]; }
            else { w *= p[d]; q[d] = g[d] + 1; }
        }
        uint32_t idx = grid_index(q[0], q[1], q[2], res, size);
        atomicAdd(&grid[(int64_t)idx * 2], w * g0);
        atomicAdd(&grid[(int64_t)idx * 2 + 1], w * g1);
    }
}

torch::Tensor hash_fwd(torch::Tensor pos, torch::Tensor params, torch::Tensor offsets,
                       torch::Tensor scales, torch::Tensor resolutions) {
    int64_t n = pos.size(0);
    int n_levels = scales.size(0);
    auto out = torch::empty({n, n_levels * 2}, pos.options());
    int64_t total = n * n_levels;
    if (total == 0) return out;
    int threads = 256;
    int64_t blocks = (total + threads - 1) / threads;
    hash_fwd_kernel<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        pos.data_ptr<float>(), params.data_ptr<float>(), offsets.data_ptr<int32_t>(),
        scales.data_ptr<float>(), resolutions.data_ptr<int32_t>(), out.data_ptr<float>(),
        n, n_levels);
    return out;
}

torch::Tensor hash_bwd(torch::Tensor pos, torch::Tensor grad_out, torch::Tensor offsets,
                       torch::Tensor scales, torch::Tensor resolutions, int64_t n_params) {
    int64_t n = pos.size(0);
    int n_levels = scales.size(0);
    auto grad = torch::zeros({n_params, 2}, pos.options());
    int64_t total = n * n_levels;
    if (total == 0) return grad;
    int threads = 256;
    int64_t blocks = (total + threads - 1) / threads;
    hash_bwd_kernel<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        pos.data_ptr<float>(), grad_out.data_ptr<float>(), offsets.data_ptr<int32_t>(),
        scales.data_ptr<float>(), resolutions.data_ptr<int32_t>(), grad.data_ptr<float>(),
        n, n_levels);
    return grad;
}
"""

_CPP_SRC = r"""
#include <torch/extension.h>
torch::Tensor hash_fwd(torch::Tensor pos, torch::Tensor params, torch::Tensor offsets,
                       torch::Tensor scales, torch::Tensor resolutions);
torch::Tensor hash_bwd(torch::Tensor pos, torch::Tensor grad_out, torch::Tensor offsets,
                       torch::Tensor scales, torch::Tensor resolutions, int64_t n_params);
"""

_EXT = None


def extension():
    global _EXT
    if _EXT is None:
        from torch.utils.cpp_extension import load_inline

        _EXT = load_inline(
            name="ss_nerf_init_gs_hashgrid_v1",
            cpp_sources=_CPP_SRC,
            cuda_sources=_CUDA_SRC.replace(
                "#include <stdint.h>", "#include <stdint.h>\n#include <ATen/cuda/CUDAContext.h>"
            ),
            functions=["hash_fwd", "hash_bwd"],
            extra_cuda_cflags=["-O3"],
            verbose=False,
        )
    return _EXT


def level_layout(n_levels, log2_hashmap_size, base_resolution, per_level_scale):
    """Per-level (scale, resolution, offset) exactly as tcnn's GridEncodingTemplated."""
    log2_scale = math.log2(per_level_scale)
    scales, resolutions, offsets = [], [], [0]
    max_params = (1 << 32) // 2 - 1
    for level in range(n_levels):
        # float32 like tcnn's exp2f(level * log2_per_level_scale) * base_resolution - 1.0f
        scale = float(
            torch.exp2(torch.tensor(level * log2_scale, dtype=torch.float32)) * base_resolution
            - 1.0
        )
        scale = float(torch.tensor(scale, dtype=torch.float32))
        res = int(math.ceil(scale)) + 1
        params = max_params if float(res) ** 3 > max_params else res**3
        params = (params + 7) // 8 * 8
        params = min(params, 1 << log2_hashmap_size)
        scales.append(scale)
        resolutions.append(res)
        offsets.append(offsets[-1] + params)
    return scales, resolutions, offsets


class _HashEncode(torch.autograd.Function):
    @staticmethod
    def forward(ctx, pos, params, offsets, scales, resolutions):
        ext = extension()
        ctx.save_for_backward(pos, offsets, scales, resolutions)
        ctx.n_params = params.shape[0]
        return ext.hash_fwd(pos, params, offsets, scales, resolutions)

    @staticmethod
    def backward(ctx, grad_out):
        pos, offsets, scales, resolutions = ctx.saved_tensors
        grad = extension().hash_bwd(
            pos, grad_out.contiguous(), offsets, scales, resolutions, ctx.n_params
        )
        return None, grad, None, None, None


def reference_encode(pos, params, offsets, scales, resolutions):
    """Pure-torch reference of the CUDA kernel (int64 emulation of uint32 arithmetic)."""
    mask32 = 0xFFFFFFFF
    outs = []
    for level in range(len(scales)):
        scale = float(scales[level])
        res = int(resolutions[level])
        size = int(offsets[level + 1] - offsets[level])
        grid = params[int(offsets[level]) : int(offsets[level + 1])]
        v = pos * torch.tensor(scale, dtype=torch.float32, device=pos.device) + 0.5
        cell = torch.floor(v)
        frac = v - cell
        cell = cell.long()
        acc = torch.zeros(len(pos), 2, device=pos.device, dtype=pos.dtype)
        for c in range(8):
            w = torch.ones(len(pos), device=pos.device, dtype=pos.dtype)
            q = []
            for d in range(3):
                if c & (1 << d):
                    w = w * frac[:, d]
                    q.append(cell[:, d] + 1)
                else:
                    w = w * (1 - frac[:, d])
                    q.append(cell[:, d])
            if res <= MAX_DENSE_RES and res**3 <= size:
                idx = (q[0] + q[1] * res + q[2] * res * res) & mask32
            else:
                idx = (q[0] ^ ((q[1] * PRIME_Y) & mask32) ^ ((q[2] * PRIME_Z) & mask32)) & mask32
            idx = idx % size
            acc = acc + w[:, None] * grid[idx]
        outs.append(acc)
    return torch.cat(outs, dim=1)


class HashGrid(torch.nn.Module):
    def __init__(self, n_levels, log2_hashmap_size, base_resolution, per_level_scale):
        super().__init__()
        scales, resolutions, offsets = level_layout(
            n_levels, log2_hashmap_size, base_resolution, per_level_scale
        )
        self.n_levels = n_levels
        self.n_output_dims = n_levels * F_PER_LEVEL
        self.register_buffer("offsets", torch.tensor(offsets, dtype=torch.int32))
        self.register_buffer("scales", torch.tensor(scales, dtype=torch.float32))
        self.register_buffer("resolutions", torch.tensor(resolutions, dtype=torch.int32))
        self.params = torch.nn.Parameter(torch.empty(offsets[-1], F_PER_LEVEL).uniform_(-1e-4, 1e-4))

    def forward(self, pos):
        """pos [P,3] in [0,1] -> [P, 2*n_levels]."""
        pos = pos.float().contiguous()
        if pos.is_cuda:
            return _HashEncode.apply(pos, self.params, self.offsets, self.scales, self.resolutions)
        return reference_encode(pos, self.params, self.offsets, self.scales, self.resolutions)
