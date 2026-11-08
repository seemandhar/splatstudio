/*
 * Deformable Beta Splatting (arXiv 2501.18630) CUDA kernels on gsplat 1.5.3 tile lists.
 *
 * Ported from the official code (Apache-2.0), github.com/RongLiu-Leo/beta-splatting @ 431d575,
 * submodules/gsplat/cuda/csrc/:
 *   rasterize_to_pixels_fwd.cu / rasterize_to_pixels_bwd.cu   (Beta-kernel rasterization)
 *   spherical_beta.cuh, compute_sb_fwd.cu, compute_sb_bwd.cu  (Spherical Beta colour)
 * The scaffold (single camera, non-packed, float32, cooperative-groups warp reductions) follows
 * reproductions/linear-splatting/plugin/ss_linear_splatting/csrc/linear_raster.cu, itself adapted
 * from gsplat v1.5.3 RasterizeToPixels3DGS{Fwd,Bwd}.cu (Apache-2.0).
 *
 * Beta kernel (paper Eq. beta + compositing eq.; official fwd lines 147-155):
 *   sigma = conic.x dx^2 + conic.z dy^2 + 2 conic.y dx dy      (= r^2, no 1/2 factor)
 *   skip if sigma < 0 or sigma >= 1                            (bounded support)
 *   alpha = min(0.999, o (1 - sigma)^beta),  beta = 4 e^b (activated in Python)
 *   no 1/255 alpha threshold; stop when T (1 - alpha) <= 1e-4 (exclusive), as official.
 * Backward (official bwd lines 186-247): gradients only where o (1-sigma)^beta <= 0.999,
 *   v_sigma = -v_alpha o beta (1-sigma)^(beta-1),  v_beta = v_alpha o (1-sigma)^beta ln(1-sigma).
 *
 * Spherical Beta (paper Sec. 4.3; official spherical_beta.cuh):
 *   c = c0 + sum_m [dot > 0] dot^(4 e^{b_m}) c_m,  dot = <normalize(dir), (st cp, st sp, ct)>
 *   gradients for c0 and the six lobe parameters; no gradient w.r.t. the view direction
 *   (the official kernel never writes v_dirs).
 */

#include <torch/extension.h>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <ATen/cuda/Atomic.cuh>

namespace cg = cooperative_groups;

// ------------------------------------------------------------------ Beta-kernel rasterizer
template <uint32_t CDIM>
__global__ void beta_fwd_kernel(
    const uint32_t n_isects, const float2 *__restrict__ means2d,
    const float3 *__restrict__ conics, const float *__restrict__ colors,
    const float *__restrict__ opacities, const float *__restrict__ betas,
    const float *__restrict__ backgrounds, const uint32_t W, const uint32_t H,
    const uint32_t tile_size, const uint32_t tile_w, const uint32_t tile_h,
    const int32_t *__restrict__ tile_offsets, const int32_t *__restrict__ flatten_ids,
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
    float4 *xy_opacity_batch = reinterpret_cast<float4 *>(&id_batch[block_size]);
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
            xy_opacity_batch[tr] = {xy.x, xy.y, opacities[g], betas[g]};
            conic_batch[tr] = conics[g];
        }
        block.sync();
        const uint32_t batch_size = min(block_size, range_end - batch_start);
        for (uint32_t t = 0; (t < batch_size) && !done; ++t) {
            const float3 conic = conic_batch[t];
            const float4 xyob = xy_opacity_batch[t];
            const float2 delta = {xyob.x - px, xyob.y - py};
            const float sigma = (conic.x * delta.x * delta.x + conic.z * delta.y * delta.y) +
                                2.f * conic.y * delta.x * delta.y;
            if (sigma < 0.f || sigma >= 1.f) continue;
            const float alpha = fminf(0.999f, xyob.z * __powf(1.f - sigma, xyob.w));
            const float next_T = T * (1.0f - alpha);
            if (next_T <= 1e-4f) {  // this pixel is done: exclusive
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
__global__ void beta_bwd_kernel(
    const uint32_t n_isects, const float2 *__restrict__ means2d,
    const float3 *__restrict__ conics, const float *__restrict__ colors,
    const float *__restrict__ opacities, const float *__restrict__ betas,
    const float *__restrict__ backgrounds, const uint32_t W, const uint32_t H,
    const uint32_t tile_size, const uint32_t tile_w, const uint32_t tile_h,
    const int32_t *__restrict__ tile_offsets, const int32_t *__restrict__ flatten_ids,
    const float *__restrict__ render_alphas, const int32_t *__restrict__ last_ids,
    const float *__restrict__ v_render_colors, const float *__restrict__ v_render_alphas,
    float2 *__restrict__ v_means2d, float3 *__restrict__ v_conics,
    float *__restrict__ v_colors, float *__restrict__ v_opacities,
    float *__restrict__ v_betas) {
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
            xy_opacity_batch[tr] = {xy.x, xy.y, opacities[g], betas[g]};
            conic_batch[tr] = conics[g];
#pragma unroll
            for (uint32_t c = 0; c < CDIM; ++c) rgbs_batch[tr * CDIM + c] = colors[g * CDIM + c];
        }
        block.sync();
        for (uint32_t t = max(0, batch_end - warp_bin_final); t < batch_size; ++t) {
            bool valid = inside;
            if (batch_end - (int32_t)t > bin_final) valid = false;
            float alpha = 0.f, opac = 0.f, beta = 0.f, vis = 0.f, sigma = 0.f;
            float2 delta = {0.f, 0.f};
            float3 conic = {0.f, 0.f, 0.f};
            if (valid) {
                conic = conic_batch[t];
                const float4 xyob = xy_opacity_batch[t];
                opac = xyob.z;
                beta = xyob.w;
                delta = {xyob.x - px, xyob.y - py};
                sigma = (conic.x * delta.x * delta.x + conic.z * delta.y * delta.y) +
                        2.f * conic.y * delta.x * delta.y;
                if (sigma < 0.f || sigma >= 1.f) {
                    valid = false;
                } else {
                    vis = __powf(1.f - sigma, beta);
                    alpha = fminf(0.999f, opac * vis);
                }
            }
            if (!warp.any(valid)) continue;
            float v_rgb_local[CDIM] = {0.f};
            float3 v_conic_local = {0.f, 0.f, 0.f};
            float2 v_xy_local = {0.f, 0.f};
            float v_opacity_local = 0.f;
            float v_beta_local = 0.f;
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
                if (opac * vis <= 0.999f) {
                    const float v_sigma =
                        -v_alpha * opac * beta * __powf(1.f - sigma, beta - 1.f);
                    v_conic_local = {delta.x * delta.x * v_sigma,
                                     2.f * delta.x * delta.y * v_sigma,
                                     delta.y * delta.y * v_sigma};
                    v_xy_local = {2.f * v_sigma * (conic.x * delta.x + conic.y * delta.y),
                                  2.f * v_sigma * (conic.y * delta.x + conic.z * delta.y)};
                    v_opacity_local = vis * v_alpha;
                    v_beta_local = v_alpha * opac * vis * __logf(1.f - sigma);
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
            v_beta_local = cg::reduce(warp, v_beta_local, cg::plus<float>());
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
                gpuAtomicAdd(v_betas + g, v_beta_local);
            }
        }
    }
}

// ------------------------------------------------------------------ Spherical Beta
__global__ void sb_fwd_kernel(const uint32_t N, const uint32_t K, const float3 *__restrict__ dirs,
                              const float *__restrict__ c0, const float *__restrict__ coeffs,
                              const bool *__restrict__ masks, float *__restrict__ colors) {
    const uint32_t idx = cg::this_grid().thread_rank();
    if (idx >= N) return;
    if (masks != nullptr && !masks[idx]) return;  // output stays zero
    const float3 d = dirs[idx];
    const float inorm = rsqrtf(d.x * d.x + d.y * d.y + d.z * d.z);
    const float3 v = {d.x * inorm, d.y * inorm, d.z * inorm};
    float out[3] = {c0[idx * 3], c0[idx * 3 + 1], c0[idx * 3 + 2]};
    const float *p = coeffs + idx * K * 6;
    for (uint32_t m = 0; m < K; ++m) {
        const float theta = p[m * 6 + 3], phi = p[m * 6 + 4], b = p[m * 6 + 5];
        const float dot = v.x * sinf(theta) * cosf(phi) + v.y * sinf(theta) * sinf(phi) +
                          v.z * cosf(theta);
        const float term = dot > 0.f ? __powf(dot, 4.f * __expf(b)) : 0.f;
        out[0] += term * p[m * 6 + 0];
        out[1] += term * p[m * 6 + 1];
        out[2] += term * p[m * 6 + 2];
    }
    colors[idx * 3] = out[0];
    colors[idx * 3 + 1] = out[1];
    colors[idx * 3 + 2] = out[2];
}

__global__ void sb_bwd_kernel(const uint32_t N, const uint32_t K, const float3 *__restrict__ dirs,
                              const float *__restrict__ coeffs, const bool *__restrict__ masks,
                              const float *__restrict__ v_colors, float *__restrict__ v_c0,
                              float *__restrict__ v_coeffs) {
    const uint32_t idx = cg::this_grid().thread_rank();
    if (idx >= N) return;
    if (masks != nullptr && !masks[idx]) return;
    const float g0 = v_colors[idx * 3], g1 = v_colors[idx * 3 + 1], g2 = v_colors[idx * 3 + 2];
    v_c0[idx * 3] = g0;
    v_c0[idx * 3 + 1] = g1;
    v_c0[idx * 3 + 2] = g2;
    const float3 d = dirs[idx];
    const float inorm = rsqrtf(d.x * d.x + d.y * d.y + d.z * d.z);
    const float3 v = {d.x * inorm, d.y * inorm, d.z * inorm};
    const float *p = coeffs + idx * K * 6;
    float *vp = v_coeffs + idx * K * 6;
    for (uint32_t m = 0; m < K; ++m) {
        const float r = p[m * 6], g = p[m * 6 + 1], bl = p[m * 6 + 2];
        const float theta = p[m * 6 + 3], phi = p[m * 6 + 4], b = p[m * 6 + 5];
        const float st = sinf(theta), ct = cosf(theta), sp = sinf(phi), cp = cosf(phi);
        const float dot = v.x * st * cp + v.y * st * sp + v.z * ct;
        float term = 0.f, g_theta = 0.f, g_phi = 0.f, g_b = 0.f;
        if (dot > 0.f) {
            const float expo = 4.f * __expf(b);
            term = __powf(dot, expo);
            const float color_grad = g0 * r + g1 * g + g2 * bl;
            const float d_theta = v.x * ct * cp + v.y * ct * sp - v.z * st;
            const float d_phi = -v.x * st * sp + v.y * st * cp;
            const float dpow = expo * __powf(dot, expo - 1.f);
            g_theta = color_grad * dpow * d_theta;
            g_phi = color_grad * dpow * d_phi;
            g_b = color_grad * expo * term * __logf(dot);
        }
        vp[m * 6] = g0 * term;
        vp[m * 6 + 1] = g1 * term;
        vp[m * 6 + 2] = g2 * term;
        vp[m * 6 + 3] = g_theta;
        vp[m * 6 + 4] = g_phi;
        vp[m * 6 + 5] = g_b;
    }
}

// ------------------------------------------------------------------ host wrappers
static void check_f32(const at::Tensor &t, const char *name) {
    TORCH_CHECK(t.is_cuda() && t.is_contiguous() && t.scalar_type() == at::kFloat, name,
                " must be a contiguous float32 CUDA tensor");
}

static void check_common(const at::Tensor &means2d, const at::Tensor &conics,
                         const at::Tensor &colors, const at::Tensor &opacities,
                         const at::Tensor &betas, const at::Tensor &tile_offsets,
                         const at::Tensor &flatten_ids) {
    check_f32(means2d, "means2d");
    check_f32(conics, "conics");
    check_f32(colors, "colors");
    check_f32(opacities, "opacities");
    check_f32(betas, "betas");
    TORCH_CHECK(means2d.dim() == 2 && means2d.size(1) == 2, "means2d must be [N,2]");
    TORCH_CHECK(conics.dim() == 2 && conics.size(1) == 3, "conics must be [N,3]");
    TORCH_CHECK(colors.dim() == 2 && colors.size(0) == means2d.size(0), "colors must be [N,C]");
    TORCH_CHECK(opacities.numel() == means2d.size(0), "opacities must be [N]");
    TORCH_CHECK(betas.numel() == means2d.size(0), "betas must be [N]");
    TORCH_CHECK(tile_offsets.is_cuda() && tile_offsets.is_contiguous() &&
                    tile_offsets.scalar_type() == at::kInt && tile_offsets.dim() == 2,
                "tile_offsets must be contiguous int32 [th,tw]");
    TORCH_CHECK(flatten_ids.is_cuda() && flatten_ids.is_contiguous() &&
                    flatten_ids.scalar_type() == at::kInt,
                "flatten_ids must be contiguous int32");
}

#define BS_DISPATCH(CD, ...)                                                                    \
    switch (CD) {                                                                               \
    case 1: { constexpr uint32_t CDIM = 1; __VA_ARGS__; break; }                                \
    case 3: { constexpr uint32_t CDIM = 3; __VA_ARGS__; break; }                                \
    case 4: { constexpr uint32_t CDIM = 4; __VA_ARGS__; break; }                                \
    default: TORCH_CHECK(false, "unsupported channel count ", CD, " (1, 3 or 4)");              \
    }

std::vector<at::Tensor> beta_forward(const at::Tensor &means2d, const at::Tensor &conics,
                                     const at::Tensor &colors, const at::Tensor &opacities,
                                     const at::Tensor &betas,
                                     const c10::optional<at::Tensor> &backgrounds,
                                     const int64_t W, const int64_t H, const int64_t tile_size,
                                     const at::Tensor &tile_offsets,
                                     const at::Tensor &flatten_ids) {
    check_common(means2d, conics, colors, opacities, betas, tile_offsets, flatten_ids);
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
    BS_DISPATCH(CD, {
        TORCH_CHECK(cudaFuncSetAttribute(beta_fwd_kernel<CDIM>,
                                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                                         shmem) == cudaSuccess,
                    "cannot set shared memory size");
        beta_fwd_kernel<CDIM><<<grid, threads, shmem, stream>>>(
            n_isects, (const float2 *)means2d.data_ptr<float>(),
            (const float3 *)conics.data_ptr<float>(), colors.data_ptr<float>(),
            opacities.data_ptr<float>(), betas.data_ptr<float>(), bg, W, H, tile_size, tile_w,
            tile_h, tile_offsets.data_ptr<int32_t>(), flatten_ids.data_ptr<int32_t>(),
            renders.data_ptr<float>(), alphas.data_ptr<float>(), last_ids.data_ptr<int32_t>());
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {renders, alphas, last_ids};
}

std::vector<at::Tensor> beta_backward(
    const at::Tensor &means2d, const at::Tensor &conics, const at::Tensor &colors,
    const at::Tensor &opacities, const at::Tensor &betas,
    const c10::optional<at::Tensor> &backgrounds, const int64_t W, const int64_t H,
    const int64_t tile_size, const at::Tensor &tile_offsets, const at::Tensor &flatten_ids,
    const at::Tensor &render_alphas, const at::Tensor &last_ids,
    const at::Tensor &v_render_colors, const at::Tensor &v_render_alphas) {
    check_common(means2d, conics, colors, opacities, betas, tile_offsets, flatten_ids);
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
    auto v_betas = at::zeros_like(betas);
    const uint32_t n_isects = flatten_ids.size(0);
    if (n_isects == 0) return {v_means2d, v_conics, v_colors, v_opacities, v_betas};
    const uint32_t tile_h = tile_offsets.size(0), tile_w = tile_offsets.size(1);
    dim3 threads = {(uint32_t)tile_size, (uint32_t)tile_size, 1};
    dim3 grid = {1, tile_h, tile_w};
    TORCH_CHECK((tile_size * tile_size) % 32 == 0, "tile must hold whole warps");
    auto stream = at::cuda::getCurrentCUDAStream();
    BS_DISPATCH(CD, {
        const int64_t shmem = tile_size * tile_size *
                              (sizeof(int32_t) + sizeof(float4) + sizeof(float3) +
                               sizeof(float) * CDIM);
        TORCH_CHECK(cudaFuncSetAttribute(beta_bwd_kernel<CDIM>,
                                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                                         shmem) == cudaSuccess,
                    "cannot set shared memory size");
        beta_bwd_kernel<CDIM><<<grid, threads, shmem, stream>>>(
            n_isects, (const float2 *)means2d.data_ptr<float>(),
            (const float3 *)conics.data_ptr<float>(), colors.data_ptr<float>(),
            opacities.data_ptr<float>(), betas.data_ptr<float>(), bg, W, H, tile_size, tile_w,
            tile_h, tile_offsets.data_ptr<int32_t>(), flatten_ids.data_ptr<int32_t>(),
            render_alphas.data_ptr<float>(), last_ids.data_ptr<int32_t>(),
            v_render_colors.data_ptr<float>(), v_render_alphas.data_ptr<float>(),
            (float2 *)v_means2d.data_ptr<float>(), (float3 *)v_conics.data_ptr<float>(),
            v_colors.data_ptr<float>(), v_opacities.data_ptr<float>(), v_betas.data_ptr<float>());
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {v_means2d, v_conics, v_colors, v_opacities, v_betas};
}

static void check_sb(const at::Tensor &dirs, const at::Tensor &c0, const at::Tensor &coeffs,
                     const c10::optional<at::Tensor> &masks) {
    check_f32(dirs, "dirs");
    check_f32(c0, "c0");
    check_f32(coeffs, "coeffs");
    TORCH_CHECK(dirs.dim() == 2 && dirs.size(1) == 3, "dirs must be [N,3]");
    TORCH_CHECK(c0.dim() == 2 && c0.size(1) == 3 && c0.size(0) == dirs.size(0), "c0 must be [N,3]");
    TORCH_CHECK(coeffs.dim() == 3 && coeffs.size(2) == 6 && coeffs.size(0) == dirs.size(0),
                "coeffs must be [N,K,6]");
    if (masks.has_value()) {
        TORCH_CHECK(masks->is_cuda() && masks->is_contiguous() &&
                        masks->scalar_type() == at::kBool && masks->numel() == dirs.size(0),
                    "masks must be contiguous bool [N]");
    }
}

at::Tensor sb_forward(const at::Tensor &dirs, const at::Tensor &c0, const at::Tensor &coeffs,
                      const c10::optional<at::Tensor> &masks) {
    check_sb(dirs, c0, coeffs, masks);
    const uint32_t N = dirs.size(0), K = coeffs.size(1);
    auto colors = at::zeros_like(dirs);
    if (N) {
        auto stream = at::cuda::getCurrentCUDAStream();
        sb_fwd_kernel<<<(N + 255) / 256, 256, 0, stream>>>(
            N, K, (const float3 *)dirs.data_ptr<float>(), c0.data_ptr<float>(),
            coeffs.data_ptr<float>(), masks.has_value() ? masks->data_ptr<bool>() : nullptr,
            colors.data_ptr<float>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return colors;
}

std::vector<at::Tensor> sb_backward(const at::Tensor &dirs, const at::Tensor &coeffs,
                                    const c10::optional<at::Tensor> &masks,
                                    const at::Tensor &v_colors) {
    check_sb(dirs, v_colors, coeffs, masks);
    const uint32_t N = dirs.size(0), K = coeffs.size(1);
    auto v_c0 = at::zeros_like(dirs);
    auto v_coeffs = at::zeros_like(coeffs);
    if (N) {
        auto stream = at::cuda::getCurrentCUDAStream();
        sb_bwd_kernel<<<(N + 255) / 256, 256, 0, stream>>>(
            N, K, (const float3 *)dirs.data_ptr<float>(), coeffs.data_ptr<float>(),
            masks.has_value() ? masks->data_ptr<bool>() : nullptr, v_colors.data_ptr<float>(),
            v_c0.data_ptr<float>(), v_coeffs.data_ptr<float>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return {v_c0, v_coeffs};
}
