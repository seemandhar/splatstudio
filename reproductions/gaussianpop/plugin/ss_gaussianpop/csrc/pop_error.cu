/*
 * GaussianPOP (arXiv 2602.06830) Algorithm 1, "render once, compute locally", on gsplat 1.5.3
 * tile lists.
 *
 * Adapted from gsplat v1.5.3 (Apache-2.0, https://github.com/nerfstudio-project/gsplat)
 *   gsplat/cuda/csrc/RasterizeToPixels3DGSFwd.cu (rasterize_to_pixels_3dgs_fwd_kernel)
 * via reproductions/augs/plugin/ss_augs/csrc/contrib_stats.cu. Tile traversal, shared-memory
 * batching, the 1/255 alpha threshold, the 0.999 alpha clamp, front-to-back order and the
 * exclusive termination (next_T <= 1e-4 stops before blending) are unchanged, so the
 * contributing list H of every pixel is exactly gsplat's.
 *
 * Per pixel the tile list is traversed twice:
 *   pass 0 (Alg. 1 phase 1/2): C_render = sum_k T_k a_k c_k + T_final * bg
 *   pass 1 (Alg. 1 phase 3):  recompute P_k = sum_{i<=k} T_i a_i c_i, T_k, T_{k+1};
 *                             b_{k+1} = (C_render - P_k) / (T_{k+1} + eps)
 *                             dSE_k   = || T_k a_k (c_k - b_{k+1}) ||^2         (Eq. 4)
 *                             delta_se[ID_k] += dSE_k   (first n_max contributors only)
 * Recomputing P_k/T_k in pass 1 equals reading the cached arrays P, T of Alg. 1 (same float
 * operations in the same order) without a per-pixel local buffer.
 * Single camera, non-packed.
 */

#include <torch/extension.h>
#include <cooperative_groups.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>

namespace cg = cooperative_groups;

__global__ void pop_error_kernel(
    const uint32_t n_isects, const float2 *__restrict__ means2d,
    const float3 *__restrict__ conics, const float *__restrict__ opacities,
    const float3 *__restrict__ colors, const float3 bg, const uint32_t W, const uint32_t H,
    const uint32_t tile_size, const uint32_t tile_w, const uint32_t tile_h,
    const int32_t *__restrict__ tile_offsets, const int32_t *__restrict__ flatten_ids,
    const int32_t n_max, const float eps, float *__restrict__ delta_se,
    float *__restrict__ render) {
    auto block = cg::this_thread_block();
    const int32_t tile_id = block.group_index().y * tile_w + block.group_index().z;
    const uint32_t i = block.group_index().y * tile_size + block.thread_index().y;
    const uint32_t j = block.group_index().z * tile_size + block.thread_index().x;
    const float px = (float)j + 0.5f;
    const float py = (float)i + 0.5f;
    const int32_t pix_id = i * W + j;
    const bool inside = (i < H && j < W);

    const int32_t range_start = tile_offsets[tile_id];
    const int32_t range_end =
        (tile_id == tile_w * tile_h - 1) ? n_isects : tile_offsets[tile_id + 1];
    const uint32_t block_size = block.size();
    const uint32_t num_batches = (range_end - range_start + block_size - 1) / block_size;

    extern __shared__ int s[];
    int32_t *id_batch = (int32_t *)s;
    float3 *xy_opacity_batch = reinterpret_cast<float3 *>(&id_batch[block_size]);
    float3 *conic_batch = reinterpret_cast<float3 *>(&xy_opacity_batch[block_size]);
    float3 *color_batch = reinterpret_cast<float3 *>(&conic_batch[block_size]);
    const uint32_t tr = block.thread_rank();

    float3 C = {0.f, 0.f, 0.f};  // C_render (pass 0 result)
    for (int pass = 0; pass < 2; ++pass) {
        bool done = !inside;
        float T = 1.0f;
        float3 P = {0.f, 0.f, 0.f};
        int32_t k = 0;  // number of contributors seen so far
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
                color_batch[tr] = colors[g];
            }
            block.sync();
            const uint32_t batch_size = min(block_size, range_end - batch_start);
            for (uint32_t t = 0; (t < batch_size) && !done; ++t) {
                const float3 conic = conic_batch[t];
                const float3 xyo = xy_opacity_batch[t];
                const float2 delta = {xyo.x - px, xyo.y - py};
                const float sigma =
                    0.5f * (conic.x * delta.x * delta.x + conic.z * delta.y * delta.y) +
                    conic.y * delta.x * delta.y;
                const float alpha = fminf(0.999f, xyo.z * __expf(-sigma));
                if (sigma < 0.f || alpha < 1.f / 255.f) continue;
                const float next_T = T * (1.0f - alpha);
                if (next_T <= 1e-4f) {  // gsplat: exclusive termination
                    done = true;
                    break;
                }
                const float3 c = color_batch[t];
                const float w = alpha * T;  // T_k a_k
                P.x += w * c.x;
                P.y += w * c.y;
                P.z += w * c.z;
                if (pass == 1 && k < n_max) {
                    // b_{k+1} = (C_render - P_k) / (T_{k+1} + eps)   (Alg. 1)
                    const float inv = 1.0f / (next_T + eps);
                    const float dx = c.x - (C.x - P.x) * inv;
                    const float dy = c.y - (C.y - P.y) * inv;
                    const float dz = c.z - (C.z - P.z) * inv;
                    const float se = w * w * (dx * dx + dy * dy + dz * dz);  // Eq. 4
                    atomicAdd(&delta_se[id_batch[t]], se);
                }
                ++k;
                T = next_T;
            }
            block.sync();  // shared batch is overwritten next iteration
        }
        if (pass == 0) {
            C = {P.x + T * bg.x, P.y + T * bg.y, P.z + T * bg.z};
            if (inside) {
                render[pix_id * 3 + 0] = C.x;
                render[pix_id * 3 + 1] = C.y;
                render[pix_id * 3 + 2] = C.z;
            }
        }
        block.sync();
    }
}

std::vector<at::Tensor> pop_error(const at::Tensor &means2d, const at::Tensor &conics,
                                  const at::Tensor &opacities, const at::Tensor &colors,
                                  const at::Tensor &background, const int64_t W, const int64_t H,
                                  const int64_t tile_size, const at::Tensor &tile_offsets,
                                  const at::Tensor &flatten_ids, const int64_t n_max,
                                  const double eps, at::Tensor delta_se) {
    for (const at::Tensor *t : {&means2d, &conics, &opacities, &colors,
                                static_cast<const at::Tensor *>(&delta_se)}) {
        TORCH_CHECK(t->is_cuda() && t->is_contiguous() && t->scalar_type() == at::kFloat,
                    "inputs must be contiguous float32 CUDA tensors");
    }
    const int64_t N = means2d.size(0);
    TORCH_CHECK(means2d.dim() == 2 && means2d.size(1) == 2, "means2d must be [N,2]");
    TORCH_CHECK(conics.dim() == 2 && conics.size(1) == 3 && conics.size(0) == N, "conics [N,3]");
    TORCH_CHECK(colors.dim() == 2 && colors.size(1) == 3 && colors.size(0) == N, "colors [N,3]");
    TORCH_CHECK(opacities.dim() == 1 && opacities.size(0) == N, "opacities [N]");
    TORCH_CHECK(delta_se.dim() == 1 && delta_se.size(0) == N, "delta_se [N]");
    TORCH_CHECK(background.numel() == 3, "background must have 3 values");
    TORCH_CHECK(tile_offsets.is_cuda() && tile_offsets.scalar_type() == at::kInt &&
                    tile_offsets.is_contiguous() && tile_offsets.dim() == 2,
                "tile_offsets must be contiguous int32 [th,tw]");
    TORCH_CHECK(flatten_ids.is_cuda() && flatten_ids.scalar_type() == at::kInt &&
                    flatten_ids.is_contiguous(),
                "flatten_ids must be contiguous int32");
    auto bgc = background.to(at::kCPU, at::kFloat).contiguous();
    const float *bgp = bgc.data_ptr<float>();
    const float3 bg = {bgp[0], bgp[1], bgp[2]};
    const uint32_t tile_h = tile_offsets.size(0), tile_w = tile_offsets.size(1);
    auto render = at::empty({H, W, 3}, means2d.options());
    const uint32_t n_isects = flatten_ids.size(0);
    if (n_isects == 0) {
        render.select(2, 0).fill_(bg.x);
        render.select(2, 1).fill_(bg.y);
        render.select(2, 2).fill_(bg.z);
        return {render};
    }
    dim3 threads = {(uint32_t)tile_size, (uint32_t)tile_size, 1};
    dim3 grid = {1, tile_h, tile_w};
    const int64_t shmem = tile_size * tile_size * (sizeof(int32_t) + 3 * sizeof(float3));
    auto stream = at::cuda::getCurrentCUDAStream();
    pop_error_kernel<<<grid, threads, shmem, stream>>>(
        n_isects, (const float2 *)means2d.data_ptr<float>(),
        (const float3 *)conics.data_ptr<float>(), opacities.data_ptr<float>(),
        (const float3 *)colors.data_ptr<float>(), bg, W, H, tile_size, tile_w, tile_h,
        tile_offsets.data_ptr<int32_t>(), flatten_ids.data_ptr<int32_t>(), (int32_t)n_max,
        (float)eps, delta_se.data_ptr<float>(), render.data_ptr<float>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {render};
}
