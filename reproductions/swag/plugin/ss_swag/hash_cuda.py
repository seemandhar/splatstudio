"""CUDA kernels for the hash encoding (same maths as ``HashGrid.forward_torch``): one thread per
(point, level); forward = trilinear blend of 8 corners; backward = atomicAdd into the tables and
the analytic d/dx of the trilinear weights (gradients reach the Gaussian centres, as with a
tiny-cuda-nn encoding whose input requires grad). JIT-built with torch.utils.cpp_extension
(run through tools/ssenv for nvcc/gcc 11)."""

import torch
from torch.utils.cpp_extension import load_inline

_EXT = None

CPP = r"""
#include <torch/extension.h>
torch::Tensor hash_fwd(torch::Tensor x, torch::Tensor table, torch::Tensor offs, torch::Tensor res,
                       int64_t tsize, int64_t F);
std::vector<torch::Tensor> hash_bwd(torch::Tensor go, torch::Tensor x, torch::Tensor table,
                                    torch::Tensor offs, torch::Tensor res, int64_t tsize, int64_t F,
                                    bool need_x);
"""

CU = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__device__ __forceinline__ long long corner_idx(long long cx, long long cy, long long cz, int r,
                                                long long tsize, long long size) {
  long long R = r + 1;
  if (R * R * R <= tsize) return cx + R * (cy + R * cz);
  unsigned long long h = (unsigned long long)cx * 1ULL;
  h ^= (unsigned long long)cy * 2654435761ULL;
  h ^= (unsigned long long)cz * 805459861ULL;
  return (long long)(h % (unsigned long long)size);
}

template <int F>
__global__ void fwd_k(const float* __restrict__ x, const float* __restrict__ table,
                      const long long* __restrict__ offs, const int* __restrict__ res, int N, int L,
                      long long tsize, float* __restrict__ out) {
  int t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= N * L) return;
  int n = t / L, l = t % L;
  int r = res[l];
  long long size = offs[l + 1] - offs[l];
  const float* tab = table + offs[l] * F;
  float p[3], fr[3]; long long b[3];
  for (int k = 0; k < 3; ++k) {
    float v = fminf(fmaxf(x[n * 3 + k], 0.f), 1.f) * r;
    float fb = fminf(floorf(v), (float)(r - 1));
    b[k] = (long long)fb; fr[k] = v - fb; p[k] = v;
  }
  float acc[F];
  for (int f = 0; f < F; ++f) acc[f] = 0.f;
  for (int c = 0; c < 8; ++c) {
    int dx = c & 1, dy = (c >> 1) & 1, dz = (c >> 2) & 1;
    float w = (dx ? fr[0] : 1 - fr[0]) * (dy ? fr[1] : 1 - fr[1]) * (dz ? fr[2] : 1 - fr[2]);
    long long i = corner_idx(b[0] + dx, b[1] + dy, b[2] + dz, r, tsize, size);
    for (int f = 0; f < F; ++f) acc[f] += w * tab[i * F + f];
  }
  for (int f = 0; f < F; ++f) out[(long long)n * L * F + l * F + f] = acc[f];
}

template <int F>
__global__ void bwd_k(const float* __restrict__ go, const float* __restrict__ x,
                      const float* __restrict__ table, const long long* __restrict__ offs,
                      const int* __restrict__ res, int N, int L, long long tsize,
                      float* __restrict__ gt, float* __restrict__ gx) {
  int t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= N * L) return;
  int n = t / L, l = t % L;
  int r = res[l];
  long long size = offs[l + 1] - offs[l];
  const float* tab = table + offs[l] * F;
  float* gtab = gt + offs[l] * F;
  float fr[3]; long long b[3]; bool inside[3];
  for (int k = 0; k < 3; ++k) {
    float xr = x[n * 3 + k];
    inside[k] = (xr >= 0.f) && (xr <= 1.f);
    float v = fminf(fmaxf(xr, 0.f), 1.f) * r;
    float fb = fminf(floorf(v), (float)(r - 1));
    b[k] = (long long)fb; fr[k] = v - fb;
  }
  float g[F];
  for (int f = 0; f < F; ++f) g[f] = go[(long long)n * L * F + l * F + f];
  float dxs[3] = {0.f, 0.f, 0.f};
  for (int c = 0; c < 8; ++c) {
    int d[3] = {c & 1, (c >> 1) & 1, (c >> 2) & 1};
    float wk[3];
    for (int k = 0; k < 3; ++k) wk[k] = d[k] ? fr[k] : 1 - fr[k];
    float w = wk[0] * wk[1] * wk[2];
    long long i = corner_idx(b[0] + d[0], b[1] + d[1], b[2] + d[2], r, tsize, size);
    float dot = 0.f;
    for (int f = 0; f < F; ++f) {
      atomicAdd(&gtab[i * F + f], w * g[f]);
      dot += tab[i * F + f] * g[f];
    }
    if (gx) {
      for (int k = 0; k < 3; ++k) {
        float dw = (d[k] ? 1.f : -1.f) * wk[(k + 1) % 3] * wk[(k + 2) % 3];
        dxs[k] += dw * dot;
      }
    }
  }
  if (gx) {
    for (int k = 0; k < 3; ++k)
      if (inside[k]) atomicAdd(&gx[n * 3 + k], dxs[k] * r);
  }
}

torch::Tensor hash_fwd(torch::Tensor x, torch::Tensor table, torch::Tensor offs, torch::Tensor res,
                       int64_t tsize, int64_t F) {
  int N = x.size(0), L = res.size(0);
  auto out = torch::empty({N, L * F}, x.options());
  int threads = 256, blocks = (N * L + threads - 1) / threads;
  TORCH_CHECK(F == 2, "F must be 2");
  if (N > 0)
    fwd_k<2><<<blocks, threads>>>(x.data_ptr<float>(), table.data_ptr<float>(),
                                  (const long long*)offs.data_ptr<int64_t>(), res.data_ptr<int>(), N, L, tsize,
                                  out.data_ptr<float>());
  return out;
}

std::vector<torch::Tensor> hash_bwd(torch::Tensor go, torch::Tensor x, torch::Tensor table,
                                    torch::Tensor offs, torch::Tensor res, int64_t tsize, int64_t F,
                                    bool need_x) {
  int N = x.size(0), L = res.size(0);
  auto gt = torch::zeros_like(table);
  auto gx = need_x ? torch::zeros_like(x) : torch::Tensor();
  int threads = 256, blocks = (N * L + threads - 1) / threads;
  if (N > 0)
    bwd_k<2><<<blocks, threads>>>(go.data_ptr<float>(), x.data_ptr<float>(),
                                  table.data_ptr<float>(), (const long long*)offs.data_ptr<int64_t>(),
                                  res.data_ptr<int>(), N, L, tsize, gt.data_ptr<float>(),
                                  need_x ? gx.data_ptr<float>() : nullptr);
  if (need_x) return {gt, gx};
  return {gt};
}
"""


def ext():
    global _EXT
    if _EXT is None:
        _EXT = load_inline("ss_swag_hash", CPP, cuda_sources=CU, functions=["hash_fwd", "hash_bwd"],
                           extra_cuda_cflags=["-O3"], verbose=False)
    return _EXT


class HashEncode(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, table, offs, res, tsize, F):
        x = x.contiguous().float()
        out = ext().hash_fwd(x, table.contiguous(), offs, res, int(tsize), int(F))
        ctx.save_for_backward(x, table, offs, res)
        ctx.meta = (int(tsize), int(F))
        return out

    @staticmethod
    def backward(ctx, go):
        x, table, offs, res = ctx.saved_tensors
        tsize, F = ctx.meta
        need_x = ctx.needs_input_grad[0]
        r = ext().hash_bwd(go.contiguous(), x, table.contiguous(), offs, res, tsize, F, need_x)
        return (r[1] if need_x else None), r[0], None, None, None, None
