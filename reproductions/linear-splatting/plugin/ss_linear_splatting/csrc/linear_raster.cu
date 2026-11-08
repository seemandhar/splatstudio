/*
 * 3DLS linear-kernel rasterization (arXiv 2411.12440) on gsplat 1.5.3 tile lists.
 *
 * Adapted from gsplat v1.5.3 (Apache-2.0, https://github.com/nerfstudio-project/gsplat)
 *   gsplat/cuda/csrc/RasterizeToPixels3DGSFwd.cu  (rasterize_to_pixels_3dgs_fwd_kernel)
 *   gsplat/cuda/csrc/RasterizeToPixels3DGSBwd.cu  (rasterize_to_pixels_3dgs_bwd_kernel)
 * Tile traversal, batching through shared memory, blending, early termination, the 0.999 alpha
 * clamp, the 1/255 alpha threshold and the back-to-front backward with `buffer` are unchanged.
 * Only the falloff and its derivative differ:
 *   sigma = 1/2 d^T conic d,  D = sqrt(2 sigma)            (paper Eq. 11, Mahalanobis distance)
 *   k     = 1 - D / lambda,   alpha = min(0.999, o k)       (Eq. 12, DA with lambda)
 *   dk/dD = -1/lambda,        dD/dsigma = 1/D
 *   AGS:  dk/dD *= exp(-c D^2) (Eq. 13-14: c = 1, D unscaled; Fig. 3b) on the geometry path
 *         only; c = 0 disables AGS.
 * Single camera, non-packed.
 */

#include <torch/extension.h>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <ATen/cuda/Atomic.cuh>

namespace cg = cooperative_groups;

#define LS_ALPHA_THRESHOLD (1.f / 255.f)
#define LS_MIN_D 1e-6f

// linear falloff; returns false when the Gaussian does not touch this pixel (gsplat's `continue`)
__device__ __forceinline__ bool linear_alpha(const float3 conic, const float3 xyo, const float px,
                                             const float py, const float inv_lam, float &alpha,
                                             float &k, float &D, float2 &delta) {
    delta = {xyo.x - px, xyo.y - py};
    const float sigma = 0.5f * (conic.x * delta.x * delta.x + conic.z * delta.y * delta.y) +
                        conic.y * delta.x * delta.y;
    if (sigma < 0.f) return false;
    D = sqrtf(2.f * sigma);
    k = 1.f - D * inv_lam;
    if (k <= 0.f) return false;
    alpha = fminf(0.999f, xyo.z * k);
    return alpha >= LS_ALPHA_THRESHOLD;
}

template <uint32_t CDIM>
__global__ void linear_fwd_kernel(
    const uint32_t n_isects, const float2 *__restrict__ means2d,
    const float3 *__restrict__ conics, const float *__restrict__ colors,
    const float *__restrict__ opacities, const float *__restrict__ backgrounds,
    const uint32_t W, const uint32_t H, const uint32_t tile_size, const uint32_t tile_w,
    const uint32_t tile_h, const int32_t *__restrict__ tile_offsets,
    const int32_t *__restrict__ flatten_ids, const float inv_lam,
    float *__restrict__ render_colors, float *__restrict__ render_alphas,
    int32_t *__restrict__ last_ids) {
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
    uint32_t cur_idx = 0;
    const uint32_t tr = block.thread_rank();
    float pix_out[CDIM] = {0.f};
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
            float alpha, k, D;
            float2 delta;
            if (!linear_alpha(conic_batch[t], xy_opacity_batch[t], px, py, inv_lam, alpha, k, D,
                              delta))
                continue;
            const float next_T = T * (1.0f - alpha);
            if (next_T <= 1e-4f) {  // exclusive, as gsplat
                done = true;
                break;
            }
            const int32_t g = id_batch[t];
            const float vis = alpha * T;
            const float *c_ptr = colors + g * CDIM;
#pragma unroll
            for (uint32_t c = 0; c < CDIM; ++c) pix_out[c] += c_ptr[c] * vis;
            cur_idx = batch_start + t;
            T = next_T;
        }
    }
    if (inside) {
        render_alphas[pix_id] = 1.0f - T;
#pragma unroll
        for (uint32_t c = 0; c < CDIM; ++c)
            render_colors[pix_id * CDIM + c] =
                backgrounds == nullptr ? pix_out[c] : pix_out[c] + T * backgrounds[c];
        last_ids[pix_id] = static_cast<int32_t>(cur_idx);
    }
}

template <uint32_t CDIM>
__global__ void linear_bwd_kernel(
    const uint32_t n_isects, const float2 *__restrict__ means2d,
    const float3 *__restrict__ conics, const float *__restrict__ colors,
    const float *__restrict__ opacities, const float *__restrict__ backgrounds,
    const uint32_t W, const uint32_t H, const uint32_t tile_size, const uint32_t tile_w,
    const uint32_t tile_h, const int32_t *__restrict__ tile_offsets,
    const int32_t *__restrict__ flatten_ids, const float inv_lam, const float ags_c,
    const float *__restrict__ render_alphas, const int32_t *__restrict__ last_ids,
    const float *__restrict__ v_render_colors, const float *__restrict__ v_render_alphas,
    float2 *__restrict__ v_means2d, float3 *__restrict__ v_conics,
    float *__restrict__ v_colors, float *__restrict__ v_opacities) {
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
    float3 *xy_opacity_batch = reinterpret_cast<float3 *>(&id_batch[block_size]);
    float3 *conic_batch = reinterpret_cast<float3 *>(&xy_opacity_batch[block_size]);
    float *rgbs_batch = (float *)&conic_batch[block_size];

    const float T_final = 1.0f - render_alphas[pix_id];
    float T = T_final;
    float buffer[CDIM] = {0.f};
    const int32_t bin_final = inside ? last_ids[pix_id] : 0;
    float v_render_c[CDIM];
#pragma unroll
    for (uint32_t c = 0; c < CDIM; ++c) v_render_c[c] = v_render_colors[pix_id * CDIM + c];
    const float v_render_a = v_render_alphas[pix_id];

    const uint32_t tr = block.thread_rank();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);
    const int32_t warp_bin_final = cg::reduce(warp, bin_final, cg::greater<int>());
    for (uint32_t b = 0; b < num_batches; ++b) {
        block.sync();
        const int32_t batch_end = range_end - 1 - block_size * b;
        const int32_t batch_size = min((int32_t)block_size, batch_end + 1 - range_start);
        const int32_t idx = batch_end - tr;
        if (idx >= range_start) {
            const int32_t g = flatten_ids[idx];
            id_batch[tr] = g;
            const float2 xy = means2d[g];
            xy_opacity_batch[tr] = {xy.x, xy.y, opacities[g]};
            conic_batch[tr] = conics[g];
#pragma unroll
            for (uint32_t c = 0; c < CDIM; ++c) rgbs_batch[tr * CDIM + c] = colors[g * CDIM + c];
        }
        block.sync();
        for (uint32_t t = max(0, batch_end - warp_bin_final); t < batch_size; ++t) {
            bool valid = inside;
            if (batch_end - (int32_t)t > bin_final) valid = false;
            float alpha = 0.f, k = 0.f, D = 0.f, opac = 0.f;
            float2 delta = {0.f, 0.f};
            float3 conic = {0.f, 0.f, 0.f};
            if (valid) {
                conic = conic_batch[t];
                const float3 xyo = xy_opacity_batch[t];
                opac = xyo.z;
                valid = linear_alpha(conic, xyo, px, py, inv_lam, alpha, k, D, delta);
            }
            if (!warp.any(valid)) continue;
            float v_rgb_local[CDIM] = {0.f};
            float3 v_conic_local = {0.f, 0.f, 0.f};
            float2 v_xy_local = {0.f, 0.f};
            float v_opacity_local = 0.f;
            if (valid) {
                const float ra = 1.0f / (1.0f - alpha);
                T *= ra;
                const float fac = alpha * T;
#pragma unroll
                for (uint32_t c = 0; c < CDIM; ++c) v_rgb_local[c] = fac * v_render_c[c];
                float v_alpha = 0.f;
#pragma unroll
                for (uint32_t c = 0; c < CDIM; ++c)
                    v_alpha += (rgbs_batch[t * CDIM + c] * T - buffer[c] * ra) * v_render_c[c];
                v_alpha += T_final * ra * v_render_a;
                if (backgrounds != nullptr) {
                    float accum = 0.f;
#pragma unroll
                    for (uint32_t c = 0; c < CDIM; ++c) accum += backgrounds[c] * v_render_c[c];
                    v_alpha += -T_final * ra * accum;
                }
                if (opac * k <= 0.999f) {  // gsplat's saturation gate
                    // dalpha/dsigma = o * dk/dD * dD/dsigma = -o / (lambda D)  [x exp(-D^2), AGS]
                    float dk_dD = -inv_lam;
                    if (ags_c > 0.f) dk_dD *= __expf(-ags_c * D * D);  // Eq. 13-14 (paper: ags_c = 1)
                    const float v_sigma = opac * dk_dD / fmaxf(D, LS_MIN_D) * v_alpha;
                    v_conic_local = {0.5f * v_sigma * delta.x * delta.x,
                                     v_sigma * delta.x * delta.y,
                                     0.5f * v_sigma * delta.y * delta.y};
                    v_xy_local = {v_sigma * (conic.x * delta.x + conic.y * delta.y),
                                  v_sigma * (conic.y * delta.x + conic.z * delta.y)};
                    v_opacity_local = k * v_alpha;
                }
#pragma unroll
                for (uint32_t c = 0; c < CDIM; ++c) buffer[c] += rgbs_batch[t * CDIM + c] * fac;
            }
#pragma unroll
            for (uint32_t c = 0; c < CDIM; ++c)
                v_rgb_local[c] = cg::reduce(warp, v_rgb_local[c], cg::plus<float>());
            v_conic_local.x = cg::reduce(warp, v_conic_local.x, cg::plus<float>());
            v_conic_local.y = cg::reduce(warp, v_conic_local.y, cg::plus<float>());
            v_conic_local.z = cg::reduce(warp, v_conic_local.z, cg::plus<float>());
            v_xy_local.x = cg::reduce(warp, v_xy_local.x, cg::plus<float>());
            v_xy_local.y = cg::reduce(warp, v_xy_local.y, cg::plus<float>());
            v_opacity_local = cg::reduce(warp, v_opacity_local, cg::plus<float>());
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
                gpuAtomicAdd(v_opacities + g, v_opacity_local);
            }
        }
    }
}

static void check_f32(const at::Tensor &t, const char *name) {
    TORCH_CHECK(t.is_cuda() && t.is_contiguous() && t.scalar_type() == at::kFloat, name,
                " must be a contiguous float32 CUDA tensor");
}

static void check_common(const at::Tensor &means2d, const at::Tensor &conics,
                         const at::Tensor &colors, const at::Tensor &opacities,
                         const at::Tensor &tile_offsets, const at::Tensor &flatten_ids) {
    check_f32(means2d, "means2d");
    check_f32(conics, "conics");
    check_f32(colors, "colors");
    check_f32(opacities, "opacities");
    TORCH_CHECK(means2d.dim() == 2 && means2d.size(1) == 2, "means2d must be [N,2]");
    TORCH_CHECK(conics.dim() == 2 && conics.size(1) == 3, "conics must be [N,3]");
    TORCH_CHECK(colors.dim() == 2 && colors.size(0) == means2d.size(0), "colors must be [N,C]");
    TORCH_CHECK(opacities.numel() == means2d.size(0), "opacities must be [N]");
    TORCH_CHECK(tile_offsets.is_cuda() && tile_offsets.is_contiguous() &&
                    tile_offsets.scalar_type() == at::kInt && tile_offsets.dim() == 2,
                "tile_offsets must be contiguous int32 [th,tw]");
    TORCH_CHECK(flatten_ids.is_cuda() && flatten_ids.is_contiguous() &&
                    flatten_ids.scalar_type() == at::kInt,
                "flatten_ids must be contiguous int32");
}

#define LS_DISPATCH(CD, ...)                                                                    \
    switch (CD) {                                                                               \
    case 1: { constexpr uint32_t CDIM = 1; __VA_ARGS__; break; }                                \
    case 3: { constexpr uint32_t CDIM = 3; __VA_ARGS__; break; }                                \
    case 4: { constexpr uint32_t CDIM = 4; __VA_ARGS__; break; }                                \
    default: TORCH_CHECK(false, "unsupported channel count ", CD, " (1, 3 or 4)");              \
    }

std::vector<at::Tensor> linear_forward(const at::Tensor &means2d, const at::Tensor &conics,
                                       const at::Tensor &colors, const at::Tensor &opacities,
                                       const c10::optional<at::Tensor> &backgrounds,
                                       const int64_t W, const int64_t H, const int64_t tile_size,
                                       const at::Tensor &tile_offsets,
                                       const at::Tensor &flatten_ids, const double lam) {
    check_common(means2d, conics, colors, opacities, tile_offsets, flatten_ids);
    TORCH_CHECK(lam > 0, "lambda must be positive");
    const int64_t CD = colors.size(1);
    const float *bg = nullptr;
    if (backgrounds.has_value()) {
        check_f32(*backgrounds, "backgrounds");
        TORCH_CHECK(backgrounds->numel() == CD, "backgrounds must be [C]");
        bg = backgrounds->data_ptr<float>();
    }
    const uint32_t tile_h = tile_offsets.size(0), tile_w = tile_offsets.size(1);
    auto renders = at::empty({H, W, CD}, means2d.options());
    auto alphas = at::empty({H, W, 1}, means2d.options());
    auto last_ids = at::empty({H, W}, means2d.options().dtype(at::kInt));
    const uint32_t n_isects = flatten_ids.size(0);
    dim3 threads = {(uint32_t)tile_size, (uint32_t)tile_size, 1};
    dim3 grid = {1, tile_h, tile_w};
    const int64_t shmem = tile_size * tile_size * (sizeof(int32_t) + 2 * sizeof(float3));
    auto stream = at::cuda::getCurrentCUDAStream();
    LS_DISPATCH(CD, {
        TORCH_CHECK(cudaFuncSetAttribute(linear_fwd_kernel<CDIM>,
                                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                                         shmem) == cudaSuccess,
                    "cannot set shared memory size");
        linear_fwd_kernel<CDIM><<<grid, threads, shmem, stream>>>(
            n_isects, (const float2 *)means2d.data_ptr<float>(),
            (const float3 *)conics.data_ptr<float>(), colors.data_ptr<float>(),
            opacities.data_ptr<float>(), bg, W, H, tile_size, tile_w, tile_h,
            tile_offsets.data_ptr<int32_t>(), flatten_ids.data_ptr<int32_t>(),
            (float)(1.0 / lam), renders.data_ptr<float>(), alphas.data_ptr<float>(),
            last_ids.data_ptr<int32_t>());
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {renders, alphas, last_ids};
}

std::vector<at::Tensor> linear_backward(
    const at::Tensor &means2d, const at::Tensor &conics, const at::Tensor &colors,
    const at::Tensor &opacities, const c10::optional<at::Tensor> &backgrounds, const int64_t W,
    const int64_t H, const int64_t tile_size, const at::Tensor &tile_offsets,
    const at::Tensor &flatten_ids, const double lam, const double ags_c,
    const at::Tensor &render_alphas, const at::Tensor &last_ids,
    const at::Tensor &v_render_colors, const at::Tensor &v_render_alphas) {
    check_common(means2d, conics, colors, opacities, tile_offsets, flatten_ids);
    check_f32(render_alphas, "render_alphas");
    check_f32(v_render_colors, "v_render_colors");
    check_f32(v_render_alphas, "v_render_alphas");
    const int64_t CD = colors.size(1);
    TORCH_CHECK(v_render_colors.numel() == H * W * CD, "v_render_colors must be [H,W,C]");
    TORCH_CHECK(v_render_alphas.numel() == H * W, "v_render_alphas must be [H,W,1]");
    const float *bg = nullptr;
    if (backgrounds.has_value()) {
        check_f32(*backgrounds, "backgrounds");
        bg = backgrounds->data_ptr<float>();
    }
    auto v_means2d = at::zeros_like(means2d);
    auto v_conics = at::zeros_like(conics);
    auto v_colors = at::zeros_like(colors);
    auto v_opacities = at::zeros_like(opacities);
    const uint32_t n_isects = flatten_ids.size(0);
    if (n_isects == 0) return {v_means2d, v_conics, v_colors, v_opacities};
    const uint32_t tile_h = tile_offsets.size(0), tile_w = tile_offsets.size(1);
    dim3 threads = {(uint32_t)tile_size, (uint32_t)tile_size, 1};
    dim3 grid = {1, tile_h, tile_w};
    TORCH_CHECK((tile_size * tile_size) % 32 == 0, "tile must hold whole warps");
    auto stream = at::cuda::getCurrentCUDAStream();
    LS_DISPATCH(CD, {
        const int64_t shmem =
            tile_size * tile_size * (sizeof(int32_t) + 2 * sizeof(float3) + sizeof(float) * CDIM);
        TORCH_CHECK(cudaFuncSetAttribute(linear_bwd_kernel<CDIM>,
                                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                                         shmem) == cudaSuccess,
                    "cannot set shared memory size");
        linear_bwd_kernel<CDIM><<<grid, threads, shmem, stream>>>(
            n_isects, (const float2 *)means2d.data_ptr<float>(),
            (const float3 *)conics.data_ptr<float>(), colors.data_ptr<float>(),
            opacities.data_ptr<float>(), bg, W, H, tile_size, tile_w, tile_h,
            tile_offsets.data_ptr<int32_t>(), flatten_ids.data_ptr<int32_t>(),
            (float)(1.0 / lam), (float)ags_c, render_alphas.data_ptr<float>(),
            last_ids.data_ptr<int32_t>(), v_render_colors.data_ptr<float>(),
            v_render_alphas.data_ptr<float>(), (float2 *)v_means2d.data_ptr<float>(),
            (float3 *)v_conics.data_ptr<float>(), v_colors.data_ptr<float>(),
            v_opacities.data_ptr<float>());
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {v_means2d, v_conics, v_colors, v_opacities};
}
