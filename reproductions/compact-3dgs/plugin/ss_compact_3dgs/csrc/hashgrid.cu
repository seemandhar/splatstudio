// Multiresolution hash-grid encoding (3D input, 2 features per level), fp32, forward + parameter
// gradient. Index, hash and interpolation conventions are adapted from tiny-cuda-nn
// (include/tiny-cuda-nn/common_device.h: pos_fract, grid_index, coherent_prime_hash;
// include/tiny-cuda-nn/encodings/grid.h: kernel_grid / kernel_grid_backward), BSD-3-Clause,
// Copyright (c) NVIDIA CORPORATION, see LICENSE-tcnn.txt. The official Compact-3DGS code uses
// tcnn's "HashGrid" encoding (scene/gaussian_model.py: self.recolor).
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>

namespace {

constexpr int F = 2;  // n_features_per_level

__device__ __forceinline__ uint32_t grid_index(uint32_t hashmap_size, uint32_t resolution,
                                               const uint32_t pos_grid[3]) {
  uint32_t stride = 1;
  uint32_t index = 0;
  // tcnn MAX_BASES[3] = 0x659: larger grids would overflow the dense stride.
  if (resolution <= 0x659u) {
#pragma unroll
    for (int dim = 0; dim < 3; ++dim) {
      index += pos_grid[dim] * stride;
      stride *= resolution;
    }
  } else {
    stride = 0xFFFFFFFFu;
  }
  if (hashmap_size < stride) {
    // coherent_prime_hash: factors {1, 2654435761, 805459861}
    index = pos_grid[0] * 1u;
    index ^= pos_grid[1] * 2654435761u;
    index ^= pos_grid[2] * 805459861u;
  }
  return index % hashmap_size;
}

__device__ __forceinline__ void corners(const float* __restrict__ x, float scale, float frac[3],
                                        uint32_t grid[3]) {
#pragma unroll
  for (int dim = 0; dim < 3; ++dim) {
    float pos = fmaf(scale, x[dim], 0.5f);  // tcnn pos_fract
    float tmp = floorf(pos);
    grid[dim] = (uint32_t)(int)tmp;
    frac[dim] = pos - tmp;
  }
}

template <int LEVELS>
__global__ void hash_forward_kernel(const float* __restrict__ x, const float* __restrict__ params,
                                    const int64_t* __restrict__ offsets,
                                    const float* __restrict__ scales,
                                    const int32_t* __restrict__ resolutions, int64_t n,
                                    float2* __restrict__ out) {
  // Thread per (point, level) with the level in blockIdx.y (level-major traversal keeps each
  // level's table hot in L2, as tcnn does); out is level-major [LEVELS, N] float2 (coalesced).
  const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const float px[3] = {x[i * 3], x[i * 3 + 1], x[i * 3 + 2]};
  {
    const int level = blockIdx.y;
    const uint32_t size = (uint32_t)(offsets[level + 1] - offsets[level]);
    const float2* grid_params = reinterpret_cast<const float2*>(params) + offsets[level];
    float frac[3];
    uint32_t grid[3];
    corners(px, scales[level], frac, grid);
    float r0 = 0.f, r1 = 0.f;
#pragma unroll
    for (int c = 0; c < 8; ++c) {
      float w = 1.f;
      uint32_t local[3];
#pragma unroll
      for (int dim = 0; dim < 3; ++dim) {
        if ((c & (1 << dim)) == 0) {
          w *= 1.f - frac[dim];
          local[dim] = grid[dim];
        } else {
          w *= frac[dim];
          local[dim] = grid[dim] + 1;
        }
      }
      const float2 v = __ldg(grid_params + grid_index(size, (uint32_t)resolutions[level], local));
      r0 += w * v.x;
      r1 += w * v.y;
    }
    out[(int64_t)level * n + i] = make_float2(r0, r1);
  }
}

template <int LEVELS>
__global__ void hash_backward_kernel(const float* __restrict__ x, const float2* __restrict__ grad_out,
                                     const int64_t* __restrict__ offsets,
                                     const float* __restrict__ scales,
                                     const int32_t* __restrict__ resolutions, int64_t n,
                                     float* __restrict__ grad_params) {
  const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const float px[3] = {x[i * 3], x[i * 3 + 1], x[i * 3 + 2]};
  {
    const int level = blockIdx.y;
    const float2 g = grad_out[(int64_t)level * n + i];  // level-major [LEVELS, N]
    if (g.x == 0.f && g.y == 0.f) return;
    const uint32_t size = (uint32_t)(offsets[level + 1] - offsets[level]);
    float* grid_grad = grad_params + offsets[level] * F;
    float frac[3];
    uint32_t grid[3];
    corners(px, scales[level], frac, grid);
#pragma unroll
    for (int c = 0; c < 8; ++c) {
      float w = 1.f;
      uint32_t local[3];
#pragma unroll
      for (int dim = 0; dim < 3; ++dim) {
        if ((c & (1 << dim)) == 0) {
          w *= 1.f - frac[dim];
          local[dim] = grid[dim];
        } else {
          w *= frac[dim];
          local[dim] = grid[dim] + 1;
        }
      }
      const uint32_t idx = grid_index(size, (uint32_t)resolutions[level], local) * F;
      atomicAdd(grid_grad + idx, w * g.x);
      atomicAdd(grid_grad + idx + 1, w * g.y);
    }
  }
}

void check(const torch::Tensor& t, torch::ScalarType type, const char* name) {
  TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
  TORCH_CHECK(t.scalar_type() == type, name, " has the wrong dtype");
}

}  // namespace

constexpr int LEVELS = 16;  // official n_levels

torch::Tensor hash_forward(torch::Tensor x, torch::Tensor params, torch::Tensor offsets,
                           torch::Tensor scales, torch::Tensor resolutions) {
  check(x, torch::kFloat32, "x");
  check(params, torch::kFloat32, "params");
  check(offsets, torch::kInt64, "offsets");
  check(scales, torch::kFloat32, "scales");
  check(resolutions, torch::kInt32, "resolutions");
  TORCH_CHECK(x.dim() == 2 && x.size(1) == 3, "x must be [N,3]");
  TORCH_CHECK(scales.size(0) == LEVELS, "the kernel is compiled for 16 levels");
  const int64_t n = x.size(0);
  auto out = torch::empty({LEVELS, n, F}, x.options());
  if (n == 0) return out.permute({1, 0, 2}).reshape({n, LEVELS * F});
  const int threads = 256;
  const dim3 blocks((unsigned)((n + threads - 1) / threads), (unsigned)LEVELS);
  hash_forward_kernel<LEVELS><<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
      x.data_ptr<float>(), params.data_ptr<float>(), offsets.data_ptr<int64_t>(),
      scales.data_ptr<float>(), resolutions.data_ptr<int32_t>(), n,
      reinterpret_cast<float2*>(out.data_ptr<float>()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  // [N, LEVELS*F] in tcnn feature order (level * F + f)
  return out.permute({1, 0, 2}).reshape({n, LEVELS * F});
}

torch::Tensor hash_backward(torch::Tensor x, torch::Tensor grad_out, int64_t n_params,
                            torch::Tensor offsets, torch::Tensor scales,
                            torch::Tensor resolutions) {
  check(x, torch::kFloat32, "x");
  check(grad_out, torch::kFloat32, "grad_out");
  check(offsets, torch::kInt64, "offsets");
  check(scales, torch::kFloat32, "scales");
  check(resolutions, torch::kInt32, "resolutions");
  TORCH_CHECK(scales.size(0) == LEVELS, "the kernel is compiled for 16 levels");
  const int64_t n = x.size(0);
  TORCH_CHECK(grad_out.dim() == 2 && grad_out.size(0) == n && grad_out.size(1) == LEVELS * F,
              "grad_out must be [N, L*F]");
  auto grad = torch::zeros({n_params}, x.options());
  if (n == 0) return grad;
  auto g = grad_out.view({n, LEVELS, F}).permute({1, 0, 2}).contiguous();
  const int threads = 256;
  const dim3 blocks((unsigned)((n + threads - 1) / threads), (unsigned)LEVELS);
  hash_backward_kernel<LEVELS><<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
      x.data_ptr<float>(), reinterpret_cast<const float2*>(g.data_ptr<float>()),
      offsets.data_ptr<int64_t>(), scales.data_ptr<float>(), resolutions.data_ptr<int32_t>(), n,
      grad.data_ptr<float>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return grad;
}
