/*
 * Weighted Sum Rendering (WSR) rasterization, Sort-free Gaussian Splatting (arXiv 2410.18931,
 * Eq. 7). Per pixel, order independent:
 *   S_f = sum_i f_i * alpha_i * w_i      (f = colour channels [+ depth])
 *   S_w = sum_i alpha_i * w_i
 *   alpha_i = u_i * G_i,  G_i = exp(-1/2 d^T conic d)    (Eq. 2 with t_i -> u_i, Eq. 11)
 * The background term and the normalisation (S_f + c_B w_B) / (S_w + w_B) are done in torch.
 * No transmittance, no early termination, no alpha clamp (alpha may exceed 1). A Gaussian is
 * skipped for a pixel when |alpha| < 1/255 or its Mahalanobis distance exceeds 3.33 (gsplat's
 * projected footprint), so the result does not depend on tile boxes or on Gaussian order.
 *
 * Adapted from gsplat v1.5.3 (Apache-2.0, https://github.com/nerfstudio-project/gsplat)
 *   gsplat/cuda/csrc/RasterizeToPixels3DGSFwd.cu / RasterizeToPixels3DGSBwd.cu
 * via this repo's reproductions/linear-splatting/plugin/ss_linear_splatting/csrc/linear_raster.cu
 * (tile traversal, shared-memory batching, warp reductions and atomics kept; blending replaced).
 * Single camera, non-packed.
 */

#include <torch/extension.h>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <ATen/cuda/Atomic.cuh>

namespace cg = cooperative_groups;

#define WSR_ALPHA_THRESHOLD (1.f / 255.f)
#define WSR_MAX_SIGMA (0.5f * 3.33f * 3.33f)

// returns false when the Gaussian does not contribute to this pixel
__device__ __forceinline__ bool wsr_alpha(const float3 conic, const float2 xy, const float u,
                                          const float px, const float py, float &alpha,
                                          float &G, float2 &delta) {
    delta = {xy.x - px, xy.y - py};
    const float sigma = 0.5f * (conic.x * delta.x * delta.x + conic.z * delta.y * delta.y) +
                        conic.y * delta.x * delta.y;
    if (sigma < 0.f || sigma > WSR_MAX_SIGMA) return false;
    G = __expf(-sigma);
    alpha = u * G;
    return fabsf(alpha) >= WSR_ALPHA_THRESHOLD;
}

template <uint32_t CDIM>
__global__ void wsr_fwd_kernel(
    const uint32_t n_isects, const float2 *__restrict__ means2d,
    const float3 *__restrict__ conics, const float *__restrict__ colors,
    const float *__restrict__ opac, const float *__restrict__ weights, const uint32_t W,
    const uint32_t H, const uint32_t tile_size, const uint32_t tile_w, const uint32_t tile_h,
    const int32_t *__restrict__ tile_offsets, const int32_t *__restrict__ flatten_ids,
    float *__restrict__ render_feats, float *__restrict__ render_wsum) {
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
    float4 *xy_uw_batch = reinterpret_cast<float4 *>(&id_batch[block_size]);
    float3 *conic_batch = reinterpret_cast<float3 *>(&xy_uw_batch[block_size]);

    const uint32_t tr = block.thread_rank();
    float acc[CDIM] = {0.f};
    float wsum = 0.f;
    for (uint32_t b = 0; b < num_batches; ++b) {
        block.sync();
        const uint32_t batch_start = range_start + block_size * b;
        const uint32_t idx = batch_start + tr;
        if (idx < range_end) {
            const int32_t g = flatten_ids[idx];
            id_batch[tr] = g;
            const float2 xy = means2d[g];
            xy_uw_batch[tr] = {xy.x, xy.y, opac[g], weights[g]};
            conic_batch[tr] = conics[g];
        }
        block.sync();
        const uint32_t batch_size = min(block_size, range_end - batch_start);
        if (!inside) continue;
        for (uint32_t t = 0; t < batch_size; ++t) {
            const float4 xyuw = xy_uw_batch[t];
            float alpha, G;
            float2 delta;
            if (!wsr_alpha(conic_batch[t], {xyuw.x, xyuw.y}, xyuw.z, px, py, alpha, G, delta))
                continue;
            const float aw = alpha * xyuw.w;  // alpha_i w(d_i)
            const float *c_ptr = colors + id_batch[t] * CDIM;
#pragma unroll
            for (uint32_t c = 0; c < CDIM; ++c) acc[c] += c_ptr[c] * aw;
            wsum += aw;
        }
    }
    if (inside) {
#pragma unroll
        for (uint32_t c = 0; c < CDIM; ++c) render_feats[pix_id * CDIM + c] = acc[c];
        render_wsum[pix_id] = wsum;
    }
}

template <uint32_t CDIM>
__global__ void wsr_bwd_kernel(
    const uint32_t n_isects, const float2 *__restrict__ means2d,
    const float3 *__restrict__ conics, const float *__restrict__ colors,
    const float *__restrict__ opac, const float *__restrict__ weights, const uint32_t W,
    const uint32_t H, const uint32_t tile_size, const uint32_t tile_w, const uint32_t tile_h,
    const int32_t *__restrict__ tile_offsets, const int32_t *__restrict__ flatten_ids,
    const float *__restrict__ v_feats, const float *__restrict__ v_wsum,
    float2 *__restrict__ v_means2d, float3 *__restrict__ v_conics, float *__restrict__ v_colors,
    float *__restrict__ v_opac, float *__restrict__ v_weights) {
    auto block = cg::this_thread_block();
    const uint32_t tile_id = block.group_index().y * tile_w + block.group_index().z;
    const uint32_t i = block.group_index().y * tile_size + block.thread_index().y;
    const uint32_t j = block.group_index().z * tile_size + block.thread_index().x;
    const float px = (float)j + 0.5f;
    const float py = (float)i + 0.5f;
    const int32_t pix_id = min(i * W + j, W * H - 1);
    const bool inside = (i < H && j < W);

    const int32_t range_start = tile_offsets[tile_id];
    const int32_t range_end =
        (tile_id == tile_w * tile_h - 1) ? n_isects : tile_offsets[tile_id + 1];
    const uint32_t block_size = block.size();
    const uint32_t num_batches = (range_end - range_start + block_size - 1) / block_size;

    extern __shared__ int s[];
    int32_t *id_batch = (int32_t *)s;
    float4 *xy_uw_batch = reinterpret_cast<float4 *>(&id_batch[block_size]);
    float3 *conic_batch = reinterpret_cast<float3 *>(&xy_uw_batch[block_size]);
    float *feat_batch = (float *)&conic_batch[block_size];

    float v_f[CDIM];
#pragma unroll
    for (uint32_t c = 0; c < CDIM; ++c) v_f[c] = inside ? v_feats[pix_id * CDIM + c] : 0.f;
    const float v_ws = inside ? v_wsum[pix_id] : 0.f;

    const uint32_t tr = block.thread_rank();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);
    for (uint32_t b = 0; b < num_batches; ++b) {
        block.sync();
        const uint32_t batch_start = range_start + block_size * b;
        const uint32_t idx = batch_start + tr;
        if (idx < range_end) {
            const int32_t g = flatten_ids[idx];
            id_batch[tr] = g;
            const float2 xy = means2d[g];
            xy_uw_batch[tr] = {xy.x, xy.y, opac[g], weights[g]};
            conic_batch[tr] = conics[g];
#pragma unroll
            for (uint32_t c = 0; c < CDIM; ++c) feat_batch[tr * CDIM + c] = colors[g * CDIM + c];
        }
        block.sync();
        const uint32_t batch_size = min(block_size, range_end - batch_start);
        for (uint32_t t = 0; t < batch_size; ++t) {
            bool valid = inside;
            float alpha = 0.f, G = 0.f;
            float2 delta = {0.f, 0.f};
            const float4 xyuw = xy_uw_batch[t];
            const float3 conic = conic_batch[t];
            if (valid) valid = wsr_alpha(conic, {xyuw.x, xyuw.y}, xyuw.z, px, py, alpha, G, delta);
            if (!warp.any(valid)) continue;
            float v_rgb_local[CDIM] = {0.f};
            float3 v_conic_local = {0.f, 0.f, 0.f};
            float2 v_xy_local = {0.f, 0.f};
            float v_u_local = 0.f, v_w_local = 0.f;
            if (valid) {
                const float w = xyuw.w;
                const float aw = alpha * w;
                float sdot = v_ws;  // d(loss)/d(alpha_i w_i) = f_i . v_S_f + v_S_w
#pragma unroll
                for (uint32_t c = 0; c < CDIM; ++c) {
                    v_rgb_local[c] = aw * v_f[c];
                    sdot += feat_batch[t * CDIM + c] * v_f[c];
                }
                v_u_local = G * w * sdot;
                v_w_local = alpha * sdot;
                // d(alpha w)/d sigma = -u G w ; sigma = 1/2 d^T conic d (gsplat convention)
                const float v_sigma = -alpha * w * sdot;
                v_conic_local = {0.5f * v_sigma * delta.x * delta.x,
                                 v_sigma * delta.x * delta.y,
                                 0.5f * v_sigma * delta.y * delta.y};
                v_xy_local = {v_sigma * (conic.x * delta.x + conic.y * delta.y),
                              v_sigma * (conic.y * delta.x + conic.z * delta.y)};
            }
#pragma unroll
            for (uint32_t c = 0; c < CDIM; ++c)
                v_rgb_local[c] = cg::reduce(warp, v_rgb_local[c], cg::plus<float>());
            v_conic_local.x = cg::reduce(warp, v_conic_local.x, cg::plus<float>());
            v_conic_local.y = cg::reduce(warp, v_conic_local.y, cg::plus<float>());
            v_conic_local.z = cg::reduce(warp, v_conic_local.z, cg::plus<float>());
            v_xy_local.x = cg::reduce(warp, v_xy_local.x, cg::plus<float>());
            v_xy_local.y = cg::reduce(warp, v_xy_local.y, cg::plus<float>());
            v_u_local = cg::reduce(warp, v_u_local, cg::plus<float>());
            v_w_local = cg::reduce(warp, v_w_local, cg::plus<float>());
            if (warp.thread_rank() == 0) {
                const int32_t g = id_batch[t];
                float *v_rgb_ptr = v_colors + CDIM * g;
#pragma unroll
                for (uint32_t c = 0; c < CDIM; ++c) gpuAtomicAdd(v_rgb_ptr + c, v_rgb_local[c]);
                float *v_conic_ptr = (float *)(v_conics) + 3 * g;
                gpuAtomicAdd(v_conic_ptr, v_conic_local.x);
                gpuAtomicAdd(v_conic_ptr + 1, v_conic_local.y);
                gpuAtomicAdd(v_conic_ptr + 2, v_conic_local.z);
                float *v_xy_ptr = (float *)(v_means2d) + 2 * g;
                gpuAtomicAdd(v_xy_ptr, v_xy_local.x);
                gpuAtomicAdd(v_xy_ptr + 1, v_xy_local.y);
                gpuAtomicAdd(v_opac + g, v_u_local);
                gpuAtomicAdd(v_weights + g, v_w_local);
            }
        }
    }
}

static void check_f32(const at::Tensor &t, const char *name) {
    TORCH_CHECK(t.is_cuda() && t.is_contiguous() && t.scalar_type() == at::kFloat, name,
                " must be a contiguous float32 CUDA tensor");
}

static void check_common(const at::Tensor &means2d, const at::Tensor &conics,
                         const at::Tensor &colors, const at::Tensor &opac,
                         const at::Tensor &weights, const at::Tensor &tile_offsets,
                         const at::Tensor &flatten_ids) {
    check_f32(means2d, "means2d");
    check_f32(conics, "conics");
    check_f32(colors, "colors");
    check_f32(opac, "opac");
    check_f32(weights, "weights");
    const int64_t N = means2d.size(0);
    TORCH_CHECK(means2d.dim() == 2 && means2d.size(1) == 2, "means2d must be [N,2]");
    TORCH_CHECK(conics.dim() == 2 && conics.size(0) == N && conics.size(1) == 3,
                "conics must be [N,3]");
    TORCH_CHECK(colors.dim() == 2 && colors.size(0) == N, "colors must be [N,C]");
    TORCH_CHECK(opac.numel() == N && weights.numel() == N, "opac/weights must be [N]");
    TORCH_CHECK(tile_offsets.is_cuda() && tile_offsets.is_contiguous() &&
                    tile_offsets.scalar_type() == at::kInt && tile_offsets.dim() == 2,
                "tile_offsets must be contiguous int32 [th,tw]");
    TORCH_CHECK(flatten_ids.is_cuda() && flatten_ids.is_contiguous() &&
                    flatten_ids.scalar_type() == at::kInt,
                "flatten_ids must be contiguous int32");
}

#define WSR_DISPATCH(CD, ...)                                                                   \
    switch (CD) {                                                                               \
    case 1: { constexpr uint32_t CDIM = 1; __VA_ARGS__; break; }                                \
    case 3: { constexpr uint32_t CDIM = 3; __VA_ARGS__; break; }                                \
    case 4: { constexpr uint32_t CDIM = 4; __VA_ARGS__; break; }                                \
    default: TORCH_CHECK(false, "unsupported channel count ", CD, " (1, 3 or 4)");              \
    }

std::vector<at::Tensor> wsr_forward(const at::Tensor &means2d, const at::Tensor &conics,
                                    const at::Tensor &colors, const at::Tensor &opac,
                                    const at::Tensor &weights, const int64_t W, const int64_t H,
                                    const int64_t tile_size, const at::Tensor &tile_offsets,
                                    const at::Tensor &flatten_ids) {
    check_common(means2d, conics, colors, opac, weights, tile_offsets, flatten_ids);
    const int64_t CD = colors.size(1);
    const uint32_t tile_h = tile_offsets.size(0), tile_w = tile_offsets.size(1);
    auto feats = at::zeros({H, W, CD}, means2d.options());
    auto wsum = at::zeros({H, W, 1}, means2d.options());
    const uint32_t n_isects = flatten_ids.size(0);
    if (n_isects == 0) return {feats, wsum};
    dim3 threads = {(uint32_t)tile_size, (uint32_t)tile_size, 1};
    dim3 grid = {1, tile_h, tile_w};
    const int64_t shmem = tile_size * tile_size * (sizeof(int32_t) + sizeof(float4) + sizeof(float3));
    auto stream = at::cuda::getCurrentCUDAStream();
    WSR_DISPATCH(CD, {
        TORCH_CHECK(cudaFuncSetAttribute(wsr_fwd_kernel<CDIM>,
                                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                                         shmem) == cudaSuccess,
                    "cannot set shared memory size");
        wsr_fwd_kernel<CDIM><<<grid, threads, shmem, stream>>>(
            n_isects, (const float2 *)means2d.data_ptr<float>(),
            (const float3 *)conics.data_ptr<float>(), colors.data_ptr<float>(),
            opac.data_ptr<float>(), weights.data_ptr<float>(), W, H, tile_size, tile_w, tile_h,
            tile_offsets.data_ptr<int32_t>(), flatten_ids.data_ptr<int32_t>(),
            feats.data_ptr<float>(), wsum.data_ptr<float>());
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {feats, wsum};
}

std::vector<at::Tensor> wsr_backward(const at::Tensor &means2d, const at::Tensor &conics,
                                     const at::Tensor &colors, const at::Tensor &opac,
                                     const at::Tensor &weights, const int64_t W, const int64_t H,
                                     const int64_t tile_size, const at::Tensor &tile_offsets,
                                     const at::Tensor &flatten_ids, const at::Tensor &v_feats,
                                     const at::Tensor &v_wsum) {
    check_common(means2d, conics, colors, opac, weights, tile_offsets, flatten_ids);
    check_f32(v_feats, "v_feats");
    check_f32(v_wsum, "v_wsum");
    const int64_t CD = colors.size(1);
    TORCH_CHECK(v_feats.numel() == H * W * CD, "v_feats must be [H,W,C]");
    TORCH_CHECK(v_wsum.numel() == H * W, "v_wsum must be [H,W,1]");
    auto v_means2d = at::zeros_like(means2d);
    auto v_conics = at::zeros_like(conics);
    auto v_colors = at::zeros_like(colors);
    auto v_opac = at::zeros_like(opac);
    auto v_weights = at::zeros_like(weights);
    const uint32_t n_isects = flatten_ids.size(0);
    if (n_isects == 0) return {v_means2d, v_conics, v_colors, v_opac, v_weights};
    const uint32_t tile_h = tile_offsets.size(0), tile_w = tile_offsets.size(1);
    dim3 threads = {(uint32_t)tile_size, (uint32_t)tile_size, 1};
    dim3 grid = {1, tile_h, tile_w};
    TORCH_CHECK((tile_size * tile_size) % 32 == 0, "tile must hold whole warps");
    auto stream = at::cuda::getCurrentCUDAStream();
    WSR_DISPATCH(CD, {
        const int64_t shmem = tile_size * tile_size *
                              (sizeof(int32_t) + sizeof(float4) + sizeof(float3) + sizeof(float) * CDIM);
        TORCH_CHECK(cudaFuncSetAttribute(wsr_bwd_kernel<CDIM>,
                                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                                         shmem) == cudaSuccess,
                    "cannot set shared memory size");
        wsr_bwd_kernel<CDIM><<<grid, threads, shmem, stream>>>(
            n_isects, (const float2 *)means2d.data_ptr<float>(),
            (const float3 *)conics.data_ptr<float>(), colors.data_ptr<float>(),
            opac.data_ptr<float>(), weights.data_ptr<float>(), W, H, tile_size, tile_w, tile_h,
            tile_offsets.data_ptr<int32_t>(), flatten_ids.data_ptr<int32_t>(),
            v_feats.data_ptr<float>(), v_wsum.data_ptr<float>(),
            (float2 *)v_means2d.data_ptr<float>(), (float3 *)v_conics.data_ptr<float>(),
            v_colors.data_ptr<float>(), v_opac.data_ptr<float>(), v_weights.data_ptr<float>());
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {v_means2d, v_conics, v_colors, v_opac, v_weights};
}
