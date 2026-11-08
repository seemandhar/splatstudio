/*
 * BOGausS per-Gaussian statistics on gsplat 1.5.3 tile lists (forward only, no gradients).
 *
 * For every (pixel p, Gaussian i) pair that contributes to the blend of one camera:
 *   blend[i] += w_i(p) = T * alpha         (numerator of the effective opacity P_i^(c))
 *   se[i]    += w_i(p) * E(p)              (SE_{c,i} = sum_p w_i(p) (R(p) - I(p))^2)
 *   count[i] += 1                          (S_i^(c): footprint in pixels)
 * BOGausS, arXiv 2504.01844, "Effective opacity-based Gaussian pruning" and
 * "Distortion-based prioritization of Gaussians splitting".
 *
 * Tile traversal adapted from the SplatStudio Taming 3DGS port
 * (reproductions/taming-3dgs/plugin/ss_taming_3dgs/csrc/taming_stats.cu), which follows gsplat's
 * rasterize_to_pixels_3dgs_fwd_kernel numerics (Apache-2.0): alpha = min(0.999, o exp(-sigma)),
 * skip sigma < 0 or alpha < 1/255, stop when the next transmittance <= 1e-4, pixel centres at +0.5.
 */

#include <torch/extension.h>
#include <cooperative_groups.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>

namespace cg = cooperative_groups;

#define BOGAUSS_ALPHA_THRESHOLD (1.f / 255.f)

__global__ void bogauss_stats_kernel(
    const int n_isects, const int W, const int H, const int tile_size, const int tile_w,
    const int tile_h,
    const float2 *__restrict__ means2d,       // [N,2]
    const float3 *__restrict__ conics,        // [N,3]
    const float *__restrict__ opacities,      // [N]
    const int *__restrict__ tile_offsets,     // [tile_h, tile_w]
    const int *__restrict__ flatten_ids,      // [n_isects]
    const float *__restrict__ pixel_error,    // [H,W]
    float *__restrict__ accum_blend,          // [N]
    float *__restrict__ accum_se,             // [N]
    int *__restrict__ accum_count             // [N]
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
    const float err = inside ? pixel_error[i * W + j] : 0.f;

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
            const float3 conic = conic_batch[t];
            const float3 xy_opac = xy_opacity_batch[t];
            const float ddx = xy_opac.x - px, ddy = xy_opac.y - py;
            const float sigma =
                0.5f * (conic.x * ddx * ddx + conic.z * ddy * ddy) + conic.y * ddx * ddy;
            const float alpha = min(0.999f, xy_opac.z * __expf(-sigma));
            if (sigma < 0.f || alpha < BOGAUSS_ALPHA_THRESHOLD) continue;
            const float next_T = T * (1.0f - alpha);
            if (next_T <= 1e-4f) {
                done = true;
                break;
            }
            const int g = id_batch[t];
            const float w = T * alpha;
            atomicAdd(&accum_blend[g], w);
            atomicAdd(&accum_se[g], w * err);
            atomicAdd(&accum_count[g], 1);
            T = next_T;
        }
    }
}

static void check_input(const torch::Tensor &t, const char *name, torch::ScalarType type) {
    TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
    TORCH_CHECK(t.scalar_type() == type, name, " has the wrong dtype");
}

// returns blend [N] f32, se [N] f32, count [N] i32
std::vector<torch::Tensor> bogauss_stats(
    torch::Tensor means2d, torch::Tensor conics, torch::Tensor opacities,
    torch::Tensor tile_offsets, torch::Tensor flatten_ids, torch::Tensor pixel_error,
    int64_t width, int64_t height, int64_t tile_size) {
    check_input(means2d, "means2d", torch::kFloat32);
    check_input(conics, "conics", torch::kFloat32);
    check_input(opacities, "opacities", torch::kFloat32);
    check_input(tile_offsets, "tile_offsets", torch::kInt32);
    check_input(flatten_ids, "flatten_ids", torch::kInt32);
    check_input(pixel_error, "pixel_error", torch::kFloat32);
    TORCH_CHECK(pixel_error.numel() == width * height, "pixel_error must be [H,W]");
    const int N = means2d.size(0);
    const int tile_h = tile_offsets.size(0), tile_w = tile_offsets.size(1);
    auto fopt = means2d.options();
    auto blend = torch::zeros({N}, fopt);
    auto se = torch::zeros({N}, fopt);
    auto cnt = torch::zeros({N}, fopt.dtype(torch::kInt32));
    const int n_isects = flatten_ids.numel();
    if (n_isects == 0) return {blend, se, cnt};
    dim3 grid(tile_w, tile_h);
    dim3 block(tile_size, tile_size);
    const size_t shmem = tile_size * tile_size * (sizeof(int) + 2 * sizeof(float3));
    auto stream = at::cuda::getCurrentCUDAStream();
    bogauss_stats_kernel<<<grid, block, shmem, stream>>>(
        n_isects, width, height, tile_size, tile_w, tile_h,
        (const float2 *)means2d.data_ptr<float>(), (const float3 *)conics.data_ptr<float>(),
        opacities.data_ptr<float>(), tile_offsets.data_ptr<int>(), flatten_ids.data_ptr<int>(),
        pixel_error.data_ptr<float>(), blend.data_ptr<float>(), se.data_ptr<float>(),
        cnt.data_ptr<int>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {blend, se, cnt};
}
