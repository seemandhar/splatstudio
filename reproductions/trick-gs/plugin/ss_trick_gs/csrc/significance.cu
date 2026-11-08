/*
 * Copied for Trick-GS (arXiv 2501.14534, SG trick) from reproductions/lightgaussian.
 * LightGaussian (arXiv 2311.17245) global-significance accumulation on gsplat 1.5.3 tile lists.
 *
 * Per-pixel rules ported from the official modified rasterizer
 *   github.com/Kevin-2017/compress-diff-gaussian-rasterization @ 9893d8c
 *   cuda_rasterizer/forward.cu : renderCUDA_count   (Inria Gaussian-Splatting license,
 *   non-commercial research use)
 * Tile traversal / shared-memory batching adapted from gsplat v1.5.3 (Apache-2.0)
 *   gsplat/cuda/csrc/RasterizeToPixels3DGSFwd.cu
 * via reproductions/gaussianpop/plugin/ss_gaussianpop/csrc/pop_error.cu.
 *
 * Front to back over the tile list of each pixel (official renderCUDA_count):
 *   power = -0.5 (a dx^2 + c dy^2) - b dx dy;  skip if power > 0
 *   alpha = min(0.99, opacity * exp(power));   skip if alpha < 1/255
 *   test_T = T (1 - alpha);  if test_T < 1e-4: pixel done, this Gaussian NOT counted
 *   count[g] += 1;  score[g] += opacity * T;  T = test_T
 * (`opacity` is the Gaussian's activated opacity sigma_j, T the transmittance before it.)
 * The official kernel uses a non-atomic `+=`; atomics are used here.
 */

#include <torch/extension.h>
#include <cooperative_groups.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>

namespace cg = cooperative_groups;

__global__ void significance_kernel(
    const uint32_t n_isects, const float2 *__restrict__ means2d,
    const float3 *__restrict__ conics, const float *__restrict__ opacities, const uint32_t W,
    const uint32_t H, const uint32_t tile_size, const uint32_t tile_w, const uint32_t tile_h,
    const int32_t *__restrict__ tile_offsets, const int32_t *__restrict__ flatten_ids,
    int32_t *__restrict__ count, float *__restrict__ score) {
    auto block = cg::this_thread_block();
    const int32_t tile_id = block.group_index().y * tile_w + block.group_index().z;
    const uint32_t i = block.group_index().y * tile_size + block.thread_index().y;
    const uint32_t j = block.group_index().z * tile_size + block.thread_index().x;
    const float px = (float)j + 0.5f;
    const float py = (float)i + 0.5f;
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
    const uint32_t tr = block.thread_rank();

    bool done = !inside;
    float T = 1.0f;
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
            const float dx = xyo.x - px;
            const float dy = xyo.y - py;
            const float power =
                -0.5f * (conic.x * dx * dx + conic.z * dy * dy) - conic.y * dx * dy;
            if (power > 0.0f) continue;
            const float alpha = fminf(0.99f, xyo.z * __expf(power));
            if (alpha < 1.0f / 255.0f) continue;
            const float test_T = T * (1.0f - alpha);
            if (test_T < 0.0001f) {
                done = true;
                continue;
            }
            atomicAdd(&count[id_batch[t]], 1);
            atomicAdd(&score[id_batch[t]], xyo.z * T);
            T = test_T;
        }
        block.sync();  // shared batch is overwritten next iteration
    }
}

void significance(const at::Tensor &means2d, const at::Tensor &conics,
                  const at::Tensor &opacities, const int64_t W, const int64_t H,
                  const int64_t tile_size, const at::Tensor &tile_offsets,
                  const at::Tensor &flatten_ids, at::Tensor count, at::Tensor score) {
    for (const at::Tensor *t : {&means2d, &conics, &opacities,
                                static_cast<const at::Tensor *>(&score)}) {
        TORCH_CHECK(t->is_cuda() && t->is_contiguous() && t->scalar_type() == at::kFloat,
                    "inputs must be contiguous float32 CUDA tensors");
    }
    const int64_t N = means2d.size(0);
    TORCH_CHECK(means2d.dim() == 2 && means2d.size(1) == 2, "means2d must be [N,2]");
    TORCH_CHECK(conics.dim() == 2 && conics.size(1) == 3 && conics.size(0) == N, "conics [N,3]");
    TORCH_CHECK(opacities.dim() == 1 && opacities.size(0) == N, "opacities [N]");
    TORCH_CHECK(score.dim() == 1 && score.size(0) == N, "score [N]");
    TORCH_CHECK(count.is_cuda() && count.is_contiguous() && count.scalar_type() == at::kInt &&
                    count.dim() == 1 && count.size(0) == N,
                "count must be contiguous int32 [N]");
    TORCH_CHECK(tile_offsets.is_cuda() && tile_offsets.scalar_type() == at::kInt &&
                    tile_offsets.is_contiguous() && tile_offsets.dim() == 2,
                "tile_offsets must be contiguous int32 [th,tw]");
    TORCH_CHECK(flatten_ids.is_cuda() && flatten_ids.scalar_type() == at::kInt &&
                    flatten_ids.is_contiguous(),
                "flatten_ids must be contiguous int32");
    const uint32_t tile_h = tile_offsets.size(0), tile_w = tile_offsets.size(1);
    const uint32_t n_isects = flatten_ids.size(0);
    if (n_isects == 0) return;
    dim3 threads = {(uint32_t)tile_size, (uint32_t)tile_size, 1};
    dim3 grid = {1, tile_h, tile_w};
    const int64_t shmem = tile_size * tile_size * (sizeof(int32_t) + 2 * sizeof(float3));
    auto stream = at::cuda::getCurrentCUDAStream();
    significance_kernel<<<grid, threads, shmem, stream>>>(
        n_isects, (const float2 *)means2d.data_ptr<float>(),
        (const float3 *)conics.data_ptr<float>(), opacities.data_ptr<float>(), W, H, tile_size,
        tile_w, tile_h, tile_offsets.data_ptr<int32_t>(), flatten_ids.data_ptr<int32_t>(),
        count.data_ptr<int32_t>(), score.data_ptr<float>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
