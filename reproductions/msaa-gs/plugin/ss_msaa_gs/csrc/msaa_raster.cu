/*
 * MSAA-GS (arXiv 2508.10507) multi-sample 3DGS rasterization on gsplat 1.5.3 tile lists.
 *
 * Adapted from gsplat v1.5.3 (Apache-2.0, https://github.com/nerfstudio-project/gsplat)
 *   gsplat/cuda/csrc/RasterizeToPixels3DGSFwd.cu  (rasterize_to_pixels_3dgs_fwd_kernel)
 *   gsplat/cuda/csrc/RasterizeToPixels3DGSBwd.cu  (rasterize_to_pixels_3dgs_bwd_kernel)
 * via this repository's validated port reproductions/linear-splatting/.../linear_raster.cu.
 *
 * Change: every pixel carries S independent alpha-blending states, one per sub-sample position
 * (j + ox[k], i + oy[k]) inside the pixel square (paper Eq. 4: u_k = u + delta_k). Each state
 * keeps gsplat's numerics exactly (exp(-sigma) falloff, 0.999 alpha clamp, 1/255 threshold,
 * exclusive 1e-4 early termination per sample). The pixel output is the mean of the S sample
 * colours / alphas (Eq. 5). Backward runs gsplat's back-to-front recurrence for each sample with
 * dL/dC_k = dL/dC / S and sums the S per-Gaussian contributions before the warp reduction, so
 * v_means2d is the gradient of a single fused MSAA rasterizer (sum over sample paths).
 * Sub-samples stay inside the pixel square, which lies inside the tile used by gsplat's tile
 * intersection, so the tile lists are exact. Single camera, non-packed, no background
 * (the caller composites over white with the averaged alpha, which equals averaging composites).
 */

#include <torch/extension.h>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <ATen/cuda/Atomic.cuh>

namespace cg = cooperative_groups;

#define MS_ALPHA_THRESHOLD (1.f / 255.f)
#define MS_MAX_SAMPLES 8

struct SampleOffsets {
    float x[MS_MAX_SAMPLES];
    float y[MS_MAX_SAMPLES];
};

// gsplat's Gaussian falloff; false when the Gaussian does not touch this sample (`continue`)
__device__ __forceinline__ bool gauss_alpha(const float3 conic, const float3 xyo, const float px,
                                            const float py, float &alpha, float &vis,
                                            float2 &delta) {
    delta = {xyo.x - px, xyo.y - py};
    const float sigma = 0.5f * (conic.x * delta.x * delta.x + conic.z * delta.y * delta.y) +
                        conic.y * delta.x * delta.y;
    if (sigma < 0.f) return false;
    vis = __expf(-sigma);
    alpha = fminf(0.999f, xyo.z * vis);
    return alpha >= MS_ALPHA_THRESHOLD;
}

template <uint32_t CDIM, uint32_t S>
__global__ void msaa_fwd_kernel(
    const uint32_t n_isects, const float2 *__restrict__ means2d,
    const float3 *__restrict__ conics, const float *__restrict__ colors,
    const float *__restrict__ opacities, const SampleOffsets offs, const uint32_t W,
    const uint32_t H, const uint32_t tile_size, const uint32_t tile_w, const uint32_t tile_h,
    const int32_t *__restrict__ tile_offsets, const int32_t *__restrict__ flatten_ids,
    float *__restrict__ render_colors, float *__restrict__ render_alphas,
    float *__restrict__ sample_T, int32_t *__restrict__ last_ids) {
    auto block = cg::this_thread_block();
    const int32_t tile_id = block.group_index().y * tile_w + block.group_index().z;
    const uint32_t i = block.group_index().y * tile_size + block.thread_index().y;
    const uint32_t j = block.group_index().z * tile_size + block.thread_index().x;
    const int32_t pix_id = i * W + j;
    const bool inside = (i < H && j < W);
    bool done = !inside;

    float spx[S], spy[S];
#pragma unroll
    for (uint32_t k = 0; k < S; ++k) {
        spx[k] = (float)j + offs.x[k];
        spy[k] = (float)i + offs.y[k];
    }

    const int32_t range_start = tile_offsets[tile_id];
    const int32_t range_end =
        (tile_id == tile_w * tile_h - 1) ? n_isects : tile_offsets[tile_id + 1];
    const uint32_t block_size = block.size();
    const uint32_t num_batches = (range_end - range_start + block_size - 1) / block_size;

    extern __shared__ int s[];
    int32_t *id_batch = (int32_t *)s;
    float3 *xy_opacity_batch = reinterpret_cast<float3 *>(&id_batch[block_size]);
    float3 *conic_batch = reinterpret_cast<float3 *>(&xy_opacity_batch[block_size]);

    float T[S];
    bool sdone[S];
    uint32_t cur_idx[S];
    float pix_out[S][CDIM];
#pragma unroll
    for (uint32_t k = 0; k < S; ++k) {
        T[k] = 1.0f;
        sdone[k] = false;
        cur_idx[k] = 0;
#pragma unroll
        for (uint32_t c = 0; c < CDIM; ++c) pix_out[k][c] = 0.f;
    }
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
            const float *c_ptr = colors + id_batch[t] * CDIM;
            bool all_done = true;
#pragma unroll
            for (uint32_t k = 0; k < S; ++k) {
                if (sdone[k]) continue;
                float alpha, vis;
                float2 delta;
                if (gauss_alpha(conic, xyo, spx[k], spy[k], alpha, vis, delta)) {
                    const float next_T = T[k] * (1.0f - alpha);
                    if (next_T <= 1e-4f) {  // exclusive, as gsplat
                        sdone[k] = true;
                        continue;
                    }
                    const float w = alpha * T[k];
#pragma unroll
                    for (uint32_t c = 0; c < CDIM; ++c) pix_out[k][c] += c_ptr[c] * w;
                    cur_idx[k] = batch_start + t;
                    T[k] = next_T;
                }
                all_done = false;
            }
            done = all_done;
        }
    }
    if (inside) {
        float a = 0.f;
        float col[CDIM] = {0.f};
#pragma unroll
        for (uint32_t k = 0; k < S; ++k) {
            a += 1.0f - T[k];
#pragma unroll
            for (uint32_t c = 0; c < CDIM; ++c) col[c] += pix_out[k][c];
            sample_T[pix_id * S + k] = T[k];
            last_ids[pix_id * S + k] = static_cast<int32_t>(cur_idx[k]);
        }
        const float inv_s = 1.0f / (float)S;
        render_alphas[pix_id] = a * inv_s;
#pragma unroll
        for (uint32_t c = 0; c < CDIM; ++c) render_colors[pix_id * CDIM + c] = col[c] * inv_s;
    }
}

template <uint32_t CDIM, uint32_t S>
__global__ void msaa_bwd_kernel(
    const uint32_t n_isects, const float2 *__restrict__ means2d,
    const float3 *__restrict__ conics, const float *__restrict__ colors,
    const float *__restrict__ opacities, const SampleOffsets offs, const uint32_t W,
    const uint32_t H, const uint32_t tile_size, const uint32_t tile_w, const uint32_t tile_h,
    const int32_t *__restrict__ tile_offsets, const int32_t *__restrict__ flatten_ids,
    const float *__restrict__ sample_T, const int32_t *__restrict__ last_ids,
    const float *__restrict__ v_render_colors, const float *__restrict__ v_render_alphas,
    float2 *__restrict__ v_means2d, float3 *__restrict__ v_conics,
    float *__restrict__ v_colors, float *__restrict__ v_opacities) {
    auto block = cg::this_thread_block();
    const uint32_t tile_id = block.group_index().y * tile_w + block.group_index().z;
    const uint32_t i = block.group_index().y * tile_size + block.thread_index().y;
    const uint32_t j = block.group_index().z * tile_size + block.thread_index().x;
    const int32_t pix_id = min(i * W + j, W * H - 1);
    const bool inside = (i < H && j < W);

    float spx[S], spy[S];
#pragma unroll
    for (uint32_t k = 0; k < S; ++k) {
        spx[k] = (float)j + offs.x[k];
        spy[k] = (float)i + offs.y[k];
    }

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

    const float inv_s = 1.0f / (float)S;
    float T_final[S], T[S];
    int32_t bin_final[S];
    float buffer[S][CDIM];
    int32_t bin_max = 0;
#pragma unroll
    for (uint32_t k = 0; k < S; ++k) {
        T_final[k] = sample_T[pix_id * S + k];
        T[k] = T_final[k];
        bin_final[k] = inside ? last_ids[pix_id * S + k] : 0;
        bin_max = max(bin_max, bin_final[k]);
#pragma unroll
        for (uint32_t c = 0; c < CDIM; ++c) buffer[k][c] = 0.f;
    }
    // Eq. 5: dL/dC_k = dL/dC^MSAA / S (same for the averaged alpha)
    float v_render_c[CDIM];
#pragma unroll
    for (uint32_t c = 0; c < CDIM; ++c) v_render_c[c] = v_render_colors[pix_id * CDIM + c] * inv_s;
    const float v_render_a = v_render_alphas[pix_id] * inv_s;

    const uint32_t tr = block.thread_rank();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);
    const int32_t warp_bin_final = cg::reduce(warp, bin_max, cg::greater<int>());
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
            const int32_t gidx = batch_end - (int32_t)t;
            float v_rgb_local[CDIM] = {0.f};
            float3 v_conic_local = {0.f, 0.f, 0.f};
            float2 v_xy_local = {0.f, 0.f};
            float v_opacity_local = 0.f;
            bool any_valid = false;
            if (inside && gidx <= bin_max) {
                const float3 conic = conic_batch[t];
                const float3 xyo = xy_opacity_batch[t];
                const float opac = xyo.z;
#pragma unroll
                for (uint32_t k = 0; k < S; ++k) {
                    if (gidx > bin_final[k]) continue;
                    float alpha, vis;
                    float2 delta;
                    if (!gauss_alpha(conic, xyo, spx[k], spy[k], alpha, vis, delta)) continue;
                    any_valid = true;
                    const float ra = 1.0f / (1.0f - alpha);
                    T[k] *= ra;
                    const float fac = alpha * T[k];
#pragma unroll
                    for (uint32_t c = 0; c < CDIM; ++c) v_rgb_local[c] += fac * v_render_c[c];
                    float v_alpha = 0.f;
#pragma unroll
                    for (uint32_t c = 0; c < CDIM; ++c)
                        v_alpha +=
                            (rgbs_batch[t * CDIM + c] * T[k] - buffer[k][c] * ra) * v_render_c[c];
                    v_alpha += T_final[k] * ra * v_render_a;
                    if (opac * vis <= 0.999f) {  // gsplat's saturation gate
                        const float v_sigma = -opac * vis * v_alpha;
                        v_conic_local.x += 0.5f * v_sigma * delta.x * delta.x;
                        v_conic_local.y += v_sigma * delta.x * delta.y;
                        v_conic_local.z += 0.5f * v_sigma * delta.y * delta.y;
                        v_xy_local.x += v_sigma * (conic.x * delta.x + conic.y * delta.y);
                        v_xy_local.y += v_sigma * (conic.y * delta.x + conic.z * delta.y);
                        v_opacity_local += vis * v_alpha;
                    }
#pragma unroll
                    for (uint32_t c = 0; c < CDIM; ++c)
                        buffer[k][c] += rgbs_batch[t * CDIM + c] * fac;
                }
            }
            if (!warp.any(any_valid)) continue;
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

static SampleOffsets make_offsets(const at::Tensor &offsets) {
    TORCH_CHECK(offsets.device().is_cpu() && offsets.scalar_type() == at::kFloat &&
                    offsets.dim() == 2 && offsets.size(1) == 2,
                "offsets must be a float32 CPU tensor [S,2]");
    const int64_t S = offsets.size(0);
    TORCH_CHECK(S >= 1 && S <= MS_MAX_SAMPLES, "1..8 samples supported");
    auto acc = offsets.contiguous();
    const float *p = acc.data_ptr<float>();
    SampleOffsets o;
    for (int k = 0; k < MS_MAX_SAMPLES; ++k) {
        o.x[k] = k < S ? p[2 * k] : 0.5f;
        o.y[k] = k < S ? p[2 * k + 1] : 0.5f;
        TORCH_CHECK(o.x[k] >= 0.f && o.x[k] <= 1.f && o.y[k] >= 0.f && o.y[k] <= 1.f,
                    "offsets must lie in the pixel square [0,1]^2");
    }
    return o;
}

#define MS_DISPATCH_C(CD, ...)                                                                  \
    switch (CD) {                                                                               \
    case 1: { constexpr uint32_t CDIM = 1; __VA_ARGS__; break; }                                \
    case 3: { constexpr uint32_t CDIM = 3; __VA_ARGS__; break; }                                \
    case 4: { constexpr uint32_t CDIM = 4; __VA_ARGS__; break; }                                \
    default: TORCH_CHECK(false, "unsupported channel count ", CD, " (1, 3 or 4)");              \
    }

#define MS_DISPATCH_S(NS, ...)                                                                  \
    switch (NS) {                                                                               \
    case 1: { constexpr uint32_t SN = 1; __VA_ARGS__; break; }                                  \
    case 2: { constexpr uint32_t SN = 2; __VA_ARGS__; break; }                                  \
    case 4: { constexpr uint32_t SN = 4; __VA_ARGS__; break; }                                  \
    case 8: { constexpr uint32_t SN = 8; __VA_ARGS__; break; }                                  \
    default: TORCH_CHECK(false, "unsupported sample count ", NS, " (1, 2, 4 or 8)");            \
    }

std::vector<at::Tensor> msaa_forward(const at::Tensor &means2d, const at::Tensor &conics,
                                     const at::Tensor &colors, const at::Tensor &opacities,
                                     const at::Tensor &offsets, const int64_t W, const int64_t H,
                                     const int64_t tile_size, const at::Tensor &tile_offsets,
                                     const at::Tensor &flatten_ids) {
    check_common(means2d, conics, colors, opacities, tile_offsets, flatten_ids);
    const SampleOffsets offs = make_offsets(offsets);
    const int64_t NS = offsets.size(0);
    const int64_t CD = colors.size(1);
    const uint32_t tile_h = tile_offsets.size(0), tile_w = tile_offsets.size(1);
    auto renders = at::empty({H, W, CD}, means2d.options());
    auto alphas = at::empty({H, W, 1}, means2d.options());
    auto sample_T = at::empty({H, W, NS}, means2d.options());
    auto last_ids = at::empty({H, W, NS}, means2d.options().dtype(at::kInt));
    const uint32_t n_isects = flatten_ids.size(0);
    dim3 threads = {(uint32_t)tile_size, (uint32_t)tile_size, 1};
    dim3 grid = {1, tile_h, tile_w};
    const int64_t shmem = tile_size * tile_size * (sizeof(int32_t) + 2 * sizeof(float3));
    auto stream = at::cuda::getCurrentCUDAStream();
    MS_DISPATCH_C(CD, MS_DISPATCH_S(NS, {
        TORCH_CHECK(cudaFuncSetAttribute(msaa_fwd_kernel<CDIM, SN>,
                                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                                         shmem) == cudaSuccess,
                    "cannot set shared memory size");
        msaa_fwd_kernel<CDIM, SN><<<grid, threads, shmem, stream>>>(
            n_isects, (const float2 *)means2d.data_ptr<float>(),
            (const float3 *)conics.data_ptr<float>(), colors.data_ptr<float>(),
            opacities.data_ptr<float>(), offs, W, H, tile_size, tile_w, tile_h,
            tile_offsets.data_ptr<int32_t>(), flatten_ids.data_ptr<int32_t>(),
            renders.data_ptr<float>(), alphas.data_ptr<float>(), sample_T.data_ptr<float>(),
            last_ids.data_ptr<int32_t>());
    }));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {renders, alphas, sample_T, last_ids};
}

std::vector<at::Tensor> msaa_backward(
    const at::Tensor &means2d, const at::Tensor &conics, const at::Tensor &colors,
    const at::Tensor &opacities, const at::Tensor &offsets, const int64_t W, const int64_t H,
    const int64_t tile_size, const at::Tensor &tile_offsets, const at::Tensor &flatten_ids,
    const at::Tensor &sample_T, const at::Tensor &last_ids, const at::Tensor &v_render_colors,
    const at::Tensor &v_render_alphas) {
    check_common(means2d, conics, colors, opacities, tile_offsets, flatten_ids);
    const SampleOffsets offs = make_offsets(offsets);
    const int64_t NS = offsets.size(0);
    check_f32(sample_T, "sample_T");
    check_f32(v_render_colors, "v_render_colors");
    check_f32(v_render_alphas, "v_render_alphas");
    const int64_t CD = colors.size(1);
    TORCH_CHECK(sample_T.numel() == H * W * NS && last_ids.numel() == H * W * NS,
                "per-sample state must be [H,W,S]");
    TORCH_CHECK(v_render_colors.numel() == H * W * CD, "v_render_colors must be [H,W,C]");
    TORCH_CHECK(v_render_alphas.numel() == H * W, "v_render_alphas must be [H,W,1]");
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
    MS_DISPATCH_C(CD, MS_DISPATCH_S(NS, {
        const int64_t shmem =
            tile_size * tile_size * (sizeof(int32_t) + 2 * sizeof(float3) + sizeof(float) * CDIM);
        TORCH_CHECK(cudaFuncSetAttribute(msaa_bwd_kernel<CDIM, SN>,
                                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                                         shmem) == cudaSuccess,
                    "cannot set shared memory size");
        msaa_bwd_kernel<CDIM, SN><<<grid, threads, shmem, stream>>>(
            n_isects, (const float2 *)means2d.data_ptr<float>(),
            (const float3 *)conics.data_ptr<float>(), colors.data_ptr<float>(),
            opacities.data_ptr<float>(), offs, W, H, tile_size, tile_w, tile_h,
            tile_offsets.data_ptr<int32_t>(), flatten_ids.data_ptr<int32_t>(),
            sample_T.data_ptr<float>(), last_ids.data_ptr<int32_t>(),
            v_render_colors.data_ptr<float>(), v_render_alphas.data_ptr<float>(),
            (float2 *)v_means2d.data_ptr<float>(), (float3 *)v_conics.data_ptr<float>(),
            v_colors.data_ptr<float>(), v_opacities.data_ptr<float>());
    }));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {v_means2d, v_conics, v_colors, v_opacities};
}
