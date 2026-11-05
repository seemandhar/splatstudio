/*
 * TrimGS per-Gaussian single-view contribution (paper Eq. 5, official code semantics).
 *
 * Adapted from the official Trim3DGS rasterizer
 *   https://github.com/Abyssaledge/diff-gaussian-rasterization @ 58790196
 *   cuda_rasterizer/forward.cu renderCUDA, `record_transmittance` branch:
 *       float trans = pow(alpha, 2 * GAMMA) * pow(T, 2 - 2 * GAMMA);
 *       atomicAdd(&transmittance[id], trans); atomicAdd(&num_covered_pixels[id], 1);
 * which is derived from the Inria 3DGS rasterizer (Copyright (C) 2023, Inria, GRAPHDECO;
 * free for non-commercial, research and evaluation use, see LICENSE-3DGS.md).
 * The tile traversal skeleton follows the Mini-Splatting statistics kernel of
 * reproductions/mini-splatting (itself adapted from gsplat's rasterize_to_pixels kernel).
 *
 * Inputs are gsplat 1.5.3 projection / tile-intersection tensors (the caller builds the tile
 * lists with the official 3-sigma radius). Numerics follow the official kernel:
 *   power = -0.5 (a dx^2 + c dy^2) - b dx dy ; skip power > 0
 *   alpha = min(0.99, o exp(power))
 *   record alpha^(2g) T^(2-2g) and one covered pixel   <- before the 1/255 skip
 *   skip alpha < 1/255 ; test_T = T (1 - alpha) ; stop the pixel when test_T < 1e-4
 * Pixel centres are at +0.5 in gsplat's (K-based) image coordinates. Forward only.
 */

#include <torch/extension.h>
#include <cooperative_groups.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>

namespace cg = cooperative_groups;

__global__ void trim_contrib_kernel(
    const int n_isects, const int W, const int H, const int tile_size, const int tile_w,
    const int tile_h, const float exp_alpha, const float exp_T,
    const float2 *__restrict__ means2d,    // [N,2]
    const float3 *__restrict__ conics,     // [N,3]  (a, b, c) of the inverse 2D covariance
    const float *__restrict__ opacities,   // [N]
    const int *__restrict__ tile_offsets,  // [tile_h, tile_w]
    const int *__restrict__ flatten_ids,   // [n_isects]
    float *__restrict__ contrib_sum,       // [N]
    int *__restrict__ covered,             // [N]
    float *__restrict__ pix_T              // [H,W] final transmittance
) {
    auto block = cg::this_thread_block();
    const int tile_x = blockIdx.x, tile_y = blockIdx.y;
    const int tile_id = tile_y * tile_w + tile_x;
    const int i = tile_y * tile_size + threadIdx.y;
    const int j = tile_x * tile_size + threadIdx.x;
    const float px = (float)j + 0.5f;
    const float py = (float)i + 0.5f;
    const bool inside = (i < H && j < W);
    bool done = !inside;

    const int range_start = tile_offsets[tile_id];
    const int range_end = (tile_id == tile_w * tile_h - 1) ? n_isects : tile_offsets[tile_id + 1];
    const int block_size = blockDim.x * blockDim.y;
    const int num_batches = (range_end - range_start + block_size - 1) / block_size;

    extern __shared__ int s[];
    int *id_batch = (int *)s;
    float3 *xy_opacity_batch = reinterpret_cast<float3 *>(&id_batch[block_size]);
    float3 *conic_batch = reinterpret_cast<float3 *>(&xy_opacity_batch[block_size]);

    float T = 1.0f;
    const int tr = block.thread_rank();
    for (int b = 0; b < num_batches; ++b) {
        if (__syncthreads_count(done) >= block_size) break;
        const int batch_start = range_start + block_size * b;
        const int idx = batch_start + tr;
        if (idx < range_end) {
            const int g = flatten_ids[idx];
            id_batch[tr] = g;
            const float2 xy = means2d[g];
            xy_opacity_batch[tr] = {xy.x, xy.y, opacities[g]};
            conic_batch[tr] = conics[g];
        }
        block.sync();
        const int batch_size = min(block_size, range_end - batch_start);
        for (int t = 0; (t < batch_size) && !done; ++t) {
            const float3 con = conic_batch[t];
            const float3 xyo = xy_opacity_batch[t];
            const float dx = xyo.x - px, dy = xyo.y - py;
            const float power = -0.5f * (con.x * dx * dx + con.z * dy * dy) - con.y * dx * dy;
            if (power > 0.0f) continue;
            const float alpha = fminf(0.99f, xyo.z * __expf(power));
            const int g = id_batch[t];
            // official: recorded for every covered pixel, before the 1/255 skip (Eq. 5)
            atomicAdd(&contrib_sum[g], powf(alpha, exp_alpha) * powf(T, exp_T));
            atomicAdd(&covered[g], 1);
            if (alpha < 1.0f / 255.0f) continue;
            const float test_T = T * (1.0f - alpha);
            if (test_T < 0.0001f) {
                done = true;
                continue;
            }
            T = test_T;
        }
    }
    if (inside) pix_T[i * W + j] = T;
}

static void check_input(const torch::Tensor &t, const char *name, torch::ScalarType type) {
    TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
    TORCH_CHECK(t.scalar_type() == type, name, " has the wrong dtype");
}

// returns contrib_sum [N] (float), covered [N] (int32), T [H,W]
std::vector<torch::Tensor> trim_contrib(
    torch::Tensor means2d, torch::Tensor conics, torch::Tensor opacities,
    torch::Tensor tile_offsets, torch::Tensor flatten_ids, int64_t width, int64_t height,
    int64_t tile_size, double gamma) {
    check_input(means2d, "means2d", torch::kFloat32);
    check_input(conics, "conics", torch::kFloat32);
    check_input(opacities, "opacities", torch::kFloat32);
    check_input(tile_offsets, "tile_offsets", torch::kInt32);
    check_input(flatten_ids, "flatten_ids", torch::kInt32);
    const int N = means2d.size(0);
    const int tile_h = tile_offsets.size(0), tile_w = tile_offsets.size(1);
    auto fopt = means2d.options();
    auto contrib = torch::zeros({N}, fopt);
    auto covered = torch::zeros({N}, fopt.dtype(torch::kInt32));
    auto T = torch::ones({height, width}, fopt);
    const int n_isects = flatten_ids.numel();
    if (n_isects == 0) return {contrib, covered, T};
    dim3 grid(tile_w, tile_h);
    dim3 block(tile_size, tile_size);
    const size_t shmem = tile_size * tile_size * (sizeof(int) + 2 * sizeof(float3));
    auto stream = at::cuda::getCurrentCUDAStream();
    trim_contrib_kernel<<<grid, block, shmem, stream>>>(
        n_isects, width, height, tile_size, tile_w, tile_h, (float)(2.0 * gamma),
        (float)(2.0 - 2.0 * gamma), (const float2 *)means2d.data_ptr<float>(),
        (const float3 *)conics.data_ptr<float>(), opacities.data_ptr<float>(),
        tile_offsets.data_ptr<int>(), flatten_ids.data_ptr<int>(), contrib.data_ptr<float>(),
        covered.data_ptr<int>(), T.data_ptr<float>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {contrib, covered, T};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("trim_contrib", &trim_contrib, "TrimGS per-Gaussian single-view contribution");
}
