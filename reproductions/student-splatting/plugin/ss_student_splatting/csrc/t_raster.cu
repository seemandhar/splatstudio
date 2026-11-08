/*
 * 3D Student Splatting and Scooping (arXiv 2503.10148) rasterizer on gsplat 1.5.3 tile lists.
 *
 * Numerics ported from the official code, github.com/realcrane/3D-student-splating-and-scooping,
 * submodules/diff-t-rasterization/cuda_rasterizer/{forward,backward}.cu `renderCUDA`
 * (Inria Gaussian-Splatting licence, research use). The tile/batch scaffold (single camera,
 * non-packed, float32, warp reductions) follows
 * reproductions/beta-splatting/plugin/ss_beta_splatting/csrc/beta_raster.cu, itself adapted from
 * gsplat v1.5.3 RasterizeToPixels3DGS{Fwd,Bwd}.cu (Apache-2.0).
 *
 * Forward (official forward.cu renderCUDA):
 *   base  = conic.x dx^2 + conic.z dy^2 + 2 conic.y dx dy;  skip if base <= 0
 *   G     = (1 + base / nu)^(-(nu + 2) / 2)                               (paper Eq. t_2D)
 *   alpha = clamp(o G, -0.99, 0.99), o signed (tanh, Sec. 3.3: splatting and scooping)
 *   skip if |alpha| < 1/255; the pixel is done when T (1 - alpha) < 1e-4 (that splat excluded)
 * Backward (official backward.cu renderCUDA):
 *   gradients ignore the clamp (straight-through), dL/dG = o dL/dalpha,
 *   dG/dbase = p / nu (1 + base/nu)^(p - 1), p = -(nu+2)/2
 *   dL/dnu   = dL/dG (nu+2)/2 base / nu^2 (1 + base/nu)^(p-1)      (official: partial derivative,
 *              exponent's nu held fixed); exact_nu_grad adds -1/2 ln(1 + base/nu) G.
 * Conic gradient convention: full derivative w.r.t. (a, b, c) of base = a dx^2 + 2 b dx dy + c dy^2.
 */

#include <torch/extension.h>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <ATen/cuda/Atomic.cuh>

namespace cg = cooperative_groups;

template <uint32_t CDIM>
__global__ void t_fwd_kernel(
    const uint32_t n_isects, const float2 *__restrict__ means2d,
    const float3 *__restrict__ conics, const float *__restrict__ colors,
    const float *__restrict__ opacities, const float *__restrict__ nus,
    const float *__restrict__ backgrounds, const uint32_t W, const uint32_t H,
    const uint32_t tile_size, const uint32_t tile_w, const uint32_t tile_h,
    const int32_t *__restrict__ tile_offsets, const int32_t *__restrict__ flatten_ids,
    const float alpha_clip, float *__restrict__ render_colors,
    float *__restrict__ render_alphas, int32_t *__restrict__ last_ids) {
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
    float4 *xy_opacity_batch = reinterpret_cast<float4 *>(&id_batch[block_size]);
    float3 *conic_batch = reinterpret_cast<float3 *>(&xy_opacity_batch[block_size]);

    float T = 1.0f;
    int32_t cur_idx = -1;
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
            xy_opacity_batch[tr] = {xy.x, xy.y, opacities[g], nus[g]};
            conic_batch[tr] = conics[g];
        }
        block.sync();
        const uint32_t batch_size = min(block_size, range_end - batch_start);
        for (uint32_t t = 0; (t < batch_size) && !done; ++t) {
            const float3 conic = conic_batch[t];
            const float4 xyon = xy_opacity_batch[t];
            const float nu = xyon.w;
            if (nu < 1.f) continue;
            const float2 delta = {xyon.x - px, xyon.y - py};
            const float base = (conic.x * delta.x * delta.x + conic.z * delta.y * delta.y) +
                               2.f * conic.y * delta.x * delta.y;
            if (base <= 0.f) continue;
            const float power = -(nu + 2.0f) / 2.0f;
            const float G = powf(1.0f + (1.0f / nu) * base, power);
            float alpha = xyon.z * G;
            alpha = alpha >= 0.f ? fminf(alpha_clip, alpha) : fmaxf(-alpha_clip, alpha);
            if (alpha > -1.0f / 255.0f && alpha < 1.0f / 255.0f) continue;
            const float next_T = T * (1.0f - alpha);
            if (next_T < 1e-4f) {  // official: done, this splat not composited
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
        last_ids[pix_id] = cur_idx;
    }
}

template <uint32_t CDIM>
__global__ void t_bwd_kernel(
    const uint32_t n_isects, const float2 *__restrict__ means2d,
    const float3 *__restrict__ conics, const float *__restrict__ colors,
    const float *__restrict__ opacities, const float *__restrict__ nus,
    const float *__restrict__ backgrounds, const uint32_t W, const uint32_t H,
    const uint32_t tile_size, const uint32_t tile_w, const uint32_t tile_h,
    const int32_t *__restrict__ tile_offsets, const int32_t *__restrict__ flatten_ids,
    const float alpha_clip, const bool exact_nu_grad,
    const float *__restrict__ render_alphas, const int32_t *__restrict__ last_ids,
    const float *__restrict__ v_render_colors, const float *__restrict__ v_render_alphas,
    float2 *__restrict__ v_means2d, float3 *__restrict__ v_conics,
    float *__restrict__ v_colors, float *__restrict__ v_opacities, float *__restrict__ v_nus) {
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
    float4 *xy_opacity_batch = reinterpret_cast<float4 *>(&id_batch[block_size]);
    float3 *conic_batch = reinterpret_cast<float3 *>(&xy_opacity_batch[block_size]);
    float *rgbs_batch = (float *)&conic_batch[block_size];

    const float T_final = 1.0f - render_alphas[pix_id];
    float T = T_final;
    float buffer[CDIM] = {0.f};
    const int32_t bin_final = inside ? last_ids[pix_id] : -1;
    float v_render_c[CDIM];
#pragma unroll
    for (uint32_t c = 0; c < CDIM; ++c) v_render_c[c] = v_render_colors[pix_id * CDIM + c];
    const float v_render_a = v_render_alphas[pix_id];
    float bg_dot = 0.f;
    if (backgrounds != nullptr) {
#pragma unroll
        for (uint32_t c = 0; c < CDIM; ++c) bg_dot += backgrounds[c] * v_render_c[c];
    }

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
            xy_opacity_batch[tr] = {xy.x, xy.y, opacities[g], nus[g]};
            conic_batch[tr] = conics[g];
#pragma unroll
            for (uint32_t c = 0; c < CDIM; ++c) rgbs_batch[tr * CDIM + c] = colors[g * CDIM + c];
        }
        block.sync();
        for (uint32_t t = max(0, batch_end - warp_bin_final); t < batch_size; ++t) {
            bool valid = inside;
            if (batch_end - (int32_t)t > bin_final) valid = false;
            float alpha = 0.f, opac = 0.f, nu = 0.f, G = 0.f, base = 0.f, q = 0.f;
            float2 delta = {0.f, 0.f};
            float3 conic = {0.f, 0.f, 0.f};
            if (valid) {
                conic = conic_batch[t];
                const float4 xyon = xy_opacity_batch[t];
                opac = xyon.z;
                nu = xyon.w;
                delta = {xyon.x - px, xyon.y - py};
                base = (conic.x * delta.x * delta.x + conic.z * delta.y * delta.y) +
                       2.f * conic.y * delta.x * delta.y;
                if (nu < 1.f || base <= 0.f) {
                    valid = false;
                } else {
                    q = 1.0f + (1.0f / nu) * base;
                    G = powf(q, -(nu + 2.0f) / 2.0f);
                    alpha = opac * G;
                    alpha = alpha >= 0.f ? fminf(alpha_clip, alpha) : fmaxf(-alpha_clip, alpha);
                    if (alpha > -1.0f / 255.0f && alpha < 1.0f / 255.0f) valid = false;
                }
            }
            if (!warp.any(valid)) continue;
            float v_rgb_local[CDIM] = {0.f};
            float3 v_conic_local = {0.f, 0.f, 0.f};
            float2 v_xy_local = {0.f, 0.f};
            float v_opacity_local = 0.f;
            float v_nu_local = 0.f;
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
                v_alpha += -T_final * ra * bg_dot;
                // official: no clamp gate (straight-through)
                const float power = -(nu + 2.0f) / 2.0f;
                const float qp1 = powf(q, power - 1.0f);
                const float dG_dbase = power * (1.0f / nu) * qp1;
                const float v_G = opac * v_alpha;
                const float v_base = v_G * dG_dbase;
                v_conic_local = {delta.x * delta.x * v_base, 2.f * delta.x * delta.y * v_base,
                                 delta.y * delta.y * v_base};
                v_xy_local = {2.f * v_base * (conic.x * delta.x + conic.y * delta.y),
                              2.f * v_base * (conic.y * delta.x + conic.z * delta.y)};
                v_opacity_local = G * v_alpha;
                float dG_dnu = (nu + 2.0f) / 2.0f * (base / (nu * nu)) * qp1;
                if (exact_nu_grad) dG_dnu += -0.5f * logf(q) * G;
                v_nu_local = dG_dnu * v_G;
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
            v_nu_local = cg::reduce(warp, v_nu_local, cg::plus<float>());
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
                gpuAtomicAdd(v_nus + g, v_nu_local);
            }
        }
    }
}

// ------------------------------------------------------------------ host wrappers
static void check_f32(const at::Tensor &t, const char *name) {
    TORCH_CHECK(t.is_cuda() && t.is_contiguous() && t.scalar_type() == at::kFloat, name,
                " must be a contiguous float32 CUDA tensor");
}

static void check_common(const at::Tensor &means2d, const at::Tensor &conics,
                         const at::Tensor &colors, const at::Tensor &opacities,
                         const at::Tensor &nus, const at::Tensor &tile_offsets,
                         const at::Tensor &flatten_ids) {
    check_f32(means2d, "means2d");
    check_f32(conics, "conics");
    check_f32(colors, "colors");
    check_f32(opacities, "opacities");
    check_f32(nus, "nus");
    TORCH_CHECK(means2d.dim() == 2 && means2d.size(1) == 2, "means2d must be [N,2]");
    TORCH_CHECK(conics.dim() == 2 && conics.size(1) == 3, "conics must be [N,3]");
    TORCH_CHECK(colors.dim() == 2 && colors.size(0) == means2d.size(0), "colors must be [N,C]");
    TORCH_CHECK(opacities.numel() == means2d.size(0), "opacities must be [N]");
    TORCH_CHECK(nus.numel() == means2d.size(0), "nus must be [N]");
    TORCH_CHECK(tile_offsets.is_cuda() && tile_offsets.is_contiguous() &&
                    tile_offsets.scalar_type() == at::kInt && tile_offsets.dim() == 2,
                "tile_offsets must be contiguous int32 [th,tw]");
    TORCH_CHECK(flatten_ids.is_cuda() && flatten_ids.is_contiguous() &&
                    flatten_ids.scalar_type() == at::kInt,
                "flatten_ids must be contiguous int32");
}

#define T_DISPATCH(CD, ...)                                                                     \
    switch (CD) {                                                                               \
    case 1: { constexpr uint32_t CDIM = 1; __VA_ARGS__; break; }                                \
    case 3: { constexpr uint32_t CDIM = 3; __VA_ARGS__; break; }                                \
    case 4: { constexpr uint32_t CDIM = 4; __VA_ARGS__; break; }                                \
    default: TORCH_CHECK(false, "unsupported channel count ", CD, " (1, 3 or 4)");              \
    }

std::vector<at::Tensor> t_forward(const at::Tensor &means2d, const at::Tensor &conics,
                                  const at::Tensor &colors, const at::Tensor &opacities,
                                  const at::Tensor &nus,
                                  const c10::optional<at::Tensor> &backgrounds,
                                  const int64_t W, const int64_t H, const int64_t tile_size,
                                  const at::Tensor &tile_offsets, const at::Tensor &flatten_ids,
                                  const double alpha_clip) {
    check_common(means2d, conics, colors, opacities, nus, tile_offsets, flatten_ids);
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
    const int64_t shmem =
        tile_size * tile_size * (sizeof(int32_t) + sizeof(float4) + sizeof(float3));
    auto stream = at::cuda::getCurrentCUDAStream();
    T_DISPATCH(CD, {
        TORCH_CHECK(cudaFuncSetAttribute(t_fwd_kernel<CDIM>,
                                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                                         shmem) == cudaSuccess,
                    "cannot set shared memory size");
        t_fwd_kernel<CDIM><<<grid, threads, shmem, stream>>>(
            n_isects, (const float2 *)means2d.data_ptr<float>(),
            (const float3 *)conics.data_ptr<float>(), colors.data_ptr<float>(),
            opacities.data_ptr<float>(), nus.data_ptr<float>(), bg, W, H, tile_size, tile_w,
            tile_h, tile_offsets.data_ptr<int32_t>(), flatten_ids.data_ptr<int32_t>(),
            (float)alpha_clip, renders.data_ptr<float>(), alphas.data_ptr<float>(),
            last_ids.data_ptr<int32_t>());
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {renders, alphas, last_ids};
}

std::vector<at::Tensor> t_backward(
    const at::Tensor &means2d, const at::Tensor &conics, const at::Tensor &colors,
    const at::Tensor &opacities, const at::Tensor &nus,
    const c10::optional<at::Tensor> &backgrounds, const int64_t W, const int64_t H,
    const int64_t tile_size, const at::Tensor &tile_offsets, const at::Tensor &flatten_ids,
    const at::Tensor &render_alphas, const at::Tensor &last_ids,
    const at::Tensor &v_render_colors, const at::Tensor &v_render_alphas,
    const double alpha_clip, const bool exact_nu_grad) {
    check_common(means2d, conics, colors, opacities, nus, tile_offsets, flatten_ids);
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
    auto v_nus = at::zeros_like(nus);
    const uint32_t n_isects = flatten_ids.size(0);
    if (n_isects == 0) return {v_means2d, v_conics, v_colors, v_opacities, v_nus};
    const uint32_t tile_h = tile_offsets.size(0), tile_w = tile_offsets.size(1);
    dim3 threads = {(uint32_t)tile_size, (uint32_t)tile_size, 1};
    dim3 grid = {1, tile_h, tile_w};
    TORCH_CHECK((tile_size * tile_size) % 32 == 0, "tile must hold whole warps");
    auto stream = at::cuda::getCurrentCUDAStream();
    T_DISPATCH(CD, {
        const int64_t shmem = tile_size * tile_size *
                              (sizeof(int32_t) + sizeof(float4) + sizeof(float3) +
                               sizeof(float) * CDIM);
        TORCH_CHECK(cudaFuncSetAttribute(t_bwd_kernel<CDIM>,
                                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                                         shmem) == cudaSuccess,
                    "cannot set shared memory size");
        t_bwd_kernel<CDIM><<<grid, threads, shmem, stream>>>(
            n_isects, (const float2 *)means2d.data_ptr<float>(),
            (const float3 *)conics.data_ptr<float>(), colors.data_ptr<float>(),
            opacities.data_ptr<float>(), nus.data_ptr<float>(), bg, W, H, tile_size, tile_w,
            tile_h, tile_offsets.data_ptr<int32_t>(), flatten_ids.data_ptr<int32_t>(),
            (float)alpha_clip, exact_nu_grad, render_alphas.data_ptr<float>(),
            last_ids.data_ptr<int32_t>(), v_render_colors.data_ptr<float>(),
            v_render_alphas.data_ptr<float>(), (float2 *)v_means2d.data_ptr<float>(),
            (float3 *)v_conics.data_ptr<float>(), v_colors.data_ptr<float>(),
            v_opacities.data_ptr<float>(), v_nus.data_ptr<float>());
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {v_means2d, v_conics, v_colors, v_opacities, v_nus};
}
