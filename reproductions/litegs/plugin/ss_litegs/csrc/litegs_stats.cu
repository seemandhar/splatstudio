/*
 * LiteGS opacity-gradient statistics on gsplat 1.5.3 tile lists (arXiv 2503.01199 Sec. 3.3).
 *
 * For every (pixel, Gaussian) fragment that contributes to gsplat's blend this kernel computes the
 * per-pixel derivative of the loss w.r.t. the *activated* opacity o_i,
 *     g = dL/do_i (pixel) = G_i * dL/dalpha_i,
 *     dL/dalpha_i = v . ( T_i c_i - (C - P_i) / (1 - alpha_i) ),
 * where v = dL/dC is the incoming gradient of the rendered pixel colour, C the final pixel colour
 * (including the background term T_final * bg), T_i the transmittance before Gaussian i and
 * P_i = sum_{j<=i} T_j alpha_j c_j the front-to-back prefix. C - P_i is everything blended behind i
 * (incl. background), which is the quantity 3DGS's backward pass accumulates back-to-front.
 * Accumulated per Gaussian:  S += g^2,  M += g,  count += 1,  weight += T_i alpha_i.
 * These are the moments the official LiteGS rasteriser aggregates
 * (litegs/submodules/gaussian_raster/raster.cu, rasterize_backward `err_square` / forward
 * `fragment_count`, `fragment_weight_sum`; https://github.com/MooreThreads/LiteGS, Gaussian-
 * Splatting licence, see LICENSE-3DGS.md). Like the official kernel, the alpha clamp does not zero g.
 *
 * Tile traversal follows gsplat's rasterize_to_pixels_3dgs_fwd_kernel numerics
 * (alpha = min(0.999, o exp(-sigma)), skip sigma < 0 or alpha < 1/255, stop when the next
 * transmittance <= 1e-4, pixel centres at +0.5) and is adapted from the SplatStudio Taming-3DGS
 * port (reproductions/taming-3dgs/plugin/ss_taming_3dgs/csrc/taming_stats.cu).
 */

#include <torch/extension.h>
#include <cooperative_groups.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>

namespace cg = cooperative_groups;

#define LITEGS_ALPHA_THRESHOLD (1.f / 255.f)

__global__ void litegs_stats_kernel(
    const int n_isects, const int W, const int H, const int tile_size, const int tile_w,
    const int tile_h,
    const float2 *__restrict__ means2d,       // [N,2]
    const float3 *__restrict__ conics,        // [N,3]
    const float *__restrict__ opacities,      // [N]
    const float3 *__restrict__ colors,        // [N,3]
    const int *__restrict__ tile_offsets,     // [tile_h, tile_w]
    const int *__restrict__ flatten_ids,      // [n_isects]
    const float3 *__restrict__ final_rgb,     // [H,W,3]  C (with background)
    const float3 *__restrict__ v_rgb,         // [H,W,3]  dL/dC
    float *__restrict__ sq_sum,               // [N]
    float *__restrict__ sum,                  // [N]
    int *__restrict__ count,                  // [N]
    float *__restrict__ weight                // [N]
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
    float3 C = {0.f, 0.f, 0.f}, v = {0.f, 0.f, 0.f};
    if (inside) {
        C = final_rgb[i * W + j];
        v = v_rgb[i * W + j];
    }

    const int range_start = tile_offsets[tile_id];
    const int range_end = (tile_id == tile_w * tile_h - 1) ? n_isects : tile_offsets[tile_id + 1];
    const int block_size = blockDim.x * blockDim.y;
    const int num_batches = (range_end - range_start + block_size - 1) / block_size;

    extern __shared__ int s[];
    int *id_batch = (int *)s;
    float3 *xy_opacity_batch = reinterpret_cast<float3 *>(&id_batch[block_size]);
    float3 *conic_batch = reinterpret_cast<float3 *>(&xy_opacity_batch[block_size]);
    float3 *color_batch = reinterpret_cast<float3 *>(&conic_batch[block_size]);

    float T = 1.0f;
    float3 P = {0.f, 0.f, 0.f};
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
            color_batch[tr] = colors[g];
        }
        block.sync();
        const int batch_size = min(block_size, range_end - batch_start);
        for (int t = 0; (t < batch_size) && !done; ++t) {
            const float3 conic = conic_batch[t];
            const float3 xy_opac = xy_opacity_batch[t];
            const float ddx = xy_opac.x - px, ddy = xy_opac.y - py;
            const float sigma =
                0.5f * (conic.x * ddx * ddx + conic.z * ddy * ddy) + conic.y * ddx * ddy;
            const float G = __expf(-sigma);
            const float alpha = min(0.999f, xy_opac.z * G);
            if (sigma < 0.f || alpha < LITEGS_ALPHA_THRESHOLD) continue;
            const float next_T = T * (1.0f - alpha);
            if (next_T <= 1e-4f) {
                done = true;
                break;
            }
            const float3 c = color_batch[t];
            const float w = T * alpha;
            P.x += w * c.x;
            P.y += w * c.y;
            P.z += w * c.z;
            const float ra = 1.0f / (1.0f - alpha);
            const float v_alpha = v.x * (T * c.x - (C.x - P.x) * ra) +
                                  v.y * (T * c.y - (C.y - P.y) * ra) +
                                  v.z * (T * c.z - (C.z - P.z) * ra);
            const float gval = G * v_alpha;
            const int g = id_batch[t];
            atomicAdd(&sq_sum[g], gval * gval);
            atomicAdd(&sum[g], gval);
            atomicAdd(&count[g], 1);
            atomicAdd(&weight[g], w);
            T = next_T;
        }
    }
}

static void check_input(const torch::Tensor &t, const char *name, torch::ScalarType type) {
    TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
    TORCH_CHECK(t.scalar_type() == type, name, " has the wrong dtype");
}

// returns sq_sum [N] f32, sum [N] f32, count [N] i32, weight [N] f32
std::vector<torch::Tensor> litegs_stats(
    torch::Tensor means2d, torch::Tensor conics, torch::Tensor opacities, torch::Tensor colors,
    torch::Tensor tile_offsets, torch::Tensor flatten_ids, torch::Tensor final_rgb,
    torch::Tensor v_rgb, int64_t width, int64_t height, int64_t tile_size) {
    check_input(means2d, "means2d", torch::kFloat32);
    check_input(conics, "conics", torch::kFloat32);
    check_input(opacities, "opacities", torch::kFloat32);
    check_input(colors, "colors", torch::kFloat32);
    check_input(tile_offsets, "tile_offsets", torch::kInt32);
    check_input(flatten_ids, "flatten_ids", torch::kInt32);
    check_input(final_rgb, "final_rgb", torch::kFloat32);
    check_input(v_rgb, "v_rgb", torch::kFloat32);
    TORCH_CHECK(final_rgb.numel() == width * height * 3, "final_rgb must be [H,W,3]");
    TORCH_CHECK(v_rgb.numel() == width * height * 3, "v_rgb must be [H,W,3]");
    const int N = means2d.size(0);
    TORCH_CHECK(colors.size(0) == N && colors.size(1) == 3, "colors must be [N,3]");
    const int tile_h = tile_offsets.size(0), tile_w = tile_offsets.size(1);
    auto fopt = means2d.options();
    auto iopt = means2d.options().dtype(torch::kInt32);
    auto sq = torch::zeros({N}, fopt);
    auto sm = torch::zeros({N}, fopt);
    auto cnt = torch::zeros({N}, iopt);
    auto wt = torch::zeros({N}, fopt);
    const int n_isects = flatten_ids.numel();
    if (n_isects == 0) return {sq, sm, cnt, wt};
    dim3 grid(tile_w, tile_h);
    dim3 block(tile_size, tile_size);
    const size_t shmem = tile_size * tile_size * (sizeof(int) + 3 * sizeof(float3));
    auto stream = at::cuda::getCurrentCUDAStream();
    litegs_stats_kernel<<<grid, block, shmem, stream>>>(
        n_isects, width, height, tile_size, tile_w, tile_h,
        (const float2 *)means2d.data_ptr<float>(), (const float3 *)conics.data_ptr<float>(),
        opacities.data_ptr<float>(), (const float3 *)colors.data_ptr<float>(),
        tile_offsets.data_ptr<int>(), flatten_ids.data_ptr<int>(),
        (const float3 *)final_rgb.data_ptr<float>(), (const float3 *)v_rgb.data_ptr<float>(),
        sq.data_ptr<float>(), sm.data_ptr<float>(), cnt.data_ptr<int>(), wt.data_ptr<float>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {sq, sm, cnt, wt};
}
