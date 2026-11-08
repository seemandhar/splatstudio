/*
 * Forward-only blending statistics for AuGS on gsplat 1.5.3 tile lists.
 *
 * Adapted from gsplat v1.5.3 (Apache-2.0, https://github.com/nerfstudio-project/gsplat)
 *   gsplat/cuda/csrc/RasterizeToPixels3DGSFwd.cu (rasterize_to_pixels_3dgs_fwd_kernel)
 * via reproductions/linear-splatting/plugin/ss_linear_splatting/csrc/linear_raster.cu.
 * Tile traversal, shared-memory batching, the 1/255 alpha threshold and front-to-back order
 * are unchanged. Instead of colours it records, as the AuGS official modified gsplat 1.4.0
 * (github.com/XiaoXinyyx/AuGS_Code submodules/gsplat/gsplat/cuda/csrc/
 * rasterize_to_pixels_fwd.cu:173-181) does:
 *   depth_idx[pix] = last contributing Gaussian g with T_before(g) >= t_threshold, or the first
 *                    contributing one (`if (cur == -1 || T >= thr) cur = g;`), -1 if none;
 *   counts[g]     += 1 for every pixel g contributes to (`is_used` / 2D `pix_covered`).
 * alpha = min(alpha_max, o exp(-sigma)); termination is exclusive (gsplat: next_T <= 1e-4 stops
 * before blending) or inclusive (Inria 2D rasterizer: blend, then stop when T < 1e-4).
 * Single camera, non-packed.
 */

#include <torch/extension.h>
#include <cooperative_groups.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>

namespace cg = cooperative_groups;

__global__ void contrib_stats_kernel(
    const uint32_t n_isects, const float2 *__restrict__ means2d,
    const float3 *__restrict__ conics, const float *__restrict__ opacities, const uint32_t W,
    const uint32_t H, const uint32_t tile_size, const uint32_t tile_w, const uint32_t tile_h,
    const int32_t *__restrict__ tile_offsets, const int32_t *__restrict__ flatten_ids,
    const float t_threshold, const float alpha_max, const bool inclusive_stop,
    int32_t *__restrict__ depth_idx, int32_t *__restrict__ counts) {
    auto block = cg::this_thread_block();
    const int32_t tile_id = block.group_index().y * tile_w + block.group_index().z;
    const uint32_t i = block.group_index().y * tile_size + block.thread_index().y;
    const uint32_t j = block.group_index().z * tile_size + block.thread_index().x;
    const float px = (float)j + 0.5f;
    const float py = (float)i + 0.5f;
    const int32_t pix_id = i * W + j;
    const bool inside = (i < H && j < W);
    bool done = !inside;

    const int32_t range_start = tile_offsets[tile_id];
    const int32_t range_end =
        (tile_id == tile_w * tile_h - 1) ? n_isects : tile_offsets[tile_id + 1];
    const uint32_t block_size = block.size();
    const uint32_t num_batches = (range_end - range_start + block_size - 1) / block_size;

    extern __shared__ int s[];
    int32_t *id_batch = (int32_t *)s;
    float3 *xy_opacity_batch = reinterpret_cast<float3 *>(&id_batch[block_size]);
    float3 *conic_batch = reinterpret_cast<float3 *>(&xy_opacity_batch[block_size]);

    float T = 1.0f;
    int32_t cur_depth_idx = -1;
    const uint32_t tr = block.thread_rank();
    for (uint32_t b = 0; b < num_batches; ++b) {
        if (__syncthreads_count(done) >= block_size) break;
        const uint32_t batch_start = range_start + block_size * b;
        const uint32_t idx = batch_start + tr;
        if (idx < range_end) {
            const int32_t g = flatten_ids[idx];
            id_batch[tr] = g;
            const float2 xy = means2d[g];
            xy_opacity_batch[tr] = {xy.x, xy.y, opacities[g]};
            conic_batch[tr] = conics[g];
        }
        block.sync();
        const uint32_t batch_size = min(block_size, range_end - batch_start);
        for (uint32_t t = 0; (t < batch_size) && !done; ++t) {
            const float3 conic = conic_batch[t];
            const float3 xyo = xy_opacity_batch[t];
            const float2 delta = {xyo.x - px, xyo.y - py};
            const float sigma = 0.5f * (conic.x * delta.x * delta.x + conic.z * delta.y * delta.y) +
                                conic.y * delta.x * delta.y;
            const float alpha = fminf(alpha_max, xyo.z * __expf(-sigma));
            if (sigma < 0.f || alpha < 1.f / 255.f) continue;
            const float next_T = T * (1.0f - alpha);
            if (!inclusive_stop && next_T <= 1e-4f) {  // gsplat: exclusive
                done = true;
                break;
            }
            const int32_t g = id_batch[t];
            if (cur_depth_idx == -1 || T >= t_threshold) cur_depth_idx = g;
            atomicAdd(&counts[g], 1);
            T = next_T;
            if (inclusive_stop && T < 1e-4f) {  // Inria: blend, then stop
                done = true;
                break;
            }
        }
    }
    if (inside) depth_idx[pix_id] = cur_depth_idx;
}

std::vector<at::Tensor> contrib_stats(const at::Tensor &means2d, const at::Tensor &conics,
                                      const at::Tensor &opacities, const int64_t W,
                                      const int64_t H, const int64_t tile_size,
                                      const at::Tensor &tile_offsets,
                                      const at::Tensor &flatten_ids, const double t_threshold,
                                      const double alpha_max, const bool inclusive_stop) {
    for (auto *t : {&means2d, &conics, &opacities}) {
        TORCH_CHECK(t->is_cuda() && t->is_contiguous() && t->scalar_type() == at::kFloat,
                    "inputs must be contiguous float32 CUDA tensors");
    }
    TORCH_CHECK(means2d.dim() == 2 && means2d.size(1) == 2, "means2d must be [N,2]");
    TORCH_CHECK(conics.dim() == 2 && conics.size(1) == 3, "conics must be [N,3]");
    TORCH_CHECK(opacities.dim() == 1 && opacities.size(0) == means2d.size(0), "opacities [N]");
    TORCH_CHECK(tile_offsets.is_cuda() && tile_offsets.scalar_type() == at::kInt &&
                    tile_offsets.is_contiguous() && tile_offsets.dim() == 2,
                "tile_offsets must be contiguous int32 [th,tw]");
    TORCH_CHECK(flatten_ids.is_cuda() && flatten_ids.scalar_type() == at::kInt &&
                    flatten_ids.is_contiguous(),
                "flatten_ids must be contiguous int32");
    const uint32_t tile_h = tile_offsets.size(0), tile_w = tile_offsets.size(1);
    auto depth_idx = at::full({H, W}, -1, means2d.options().dtype(at::kInt));
    auto counts = at::zeros({means2d.size(0)}, means2d.options().dtype(at::kInt));
    const uint32_t n_isects = flatten_ids.size(0);
    if (n_isects == 0) return {depth_idx, counts};
    dim3 threads = {(uint32_t)tile_size, (uint32_t)tile_size, 1};
    dim3 grid = {1, tile_h, tile_w};
    const int64_t shmem = tile_size * tile_size * (sizeof(int32_t) + 2 * sizeof(float3));
    auto stream = at::cuda::getCurrentCUDAStream();
    contrib_stats_kernel<<<grid, threads, shmem, stream>>>(
        n_isects, (const float2 *)means2d.data_ptr<float>(),
        (const float3 *)conics.data_ptr<float>(), opacities.data_ptr<float>(), W, H, tile_size,
        tile_w, tile_h, tile_offsets.data_ptr<int32_t>(), flatten_ids.data_ptr<int32_t>(),
        (float)t_threshold, (float)alpha_max, inclusive_stop, depth_idx.data_ptr<int32_t>(),
        counts.data_ptr<int32_t>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {depth_idx, counts};
}
