/*
 * Vol3DGS volumetric rasterization (arXiv 2412.03378) on gsplat 1.5.3 tile lists.
 *
 * Tile traversal / shared-memory batching / back-to-front backward are adapted from
 *   gsplat v1.5.3 (Apache-2.0) gsplat/cuda/csrc/RasterizeToPixels3DGS{Fwd,Bwd}.cu
 *   via reproductions/linear-splatting/plugin/ss_linear_splatting/csrc/linear_raster.cu.
 * The per-Gaussian alpha is the official Vol3DGS one:
 *   slang-gaussian-rasterization@5eca118 (Apache-2.0, chinmay0301ucsd)
 *   internal/slang/utils.slang:evaluate_splat_volr, alphablend_volr_shader.slang
 *
 *   d      = normalize(((x+0.5-cx)/fx, (y+0.5-cy)/fy, 1))     pixel-centre ray, camera space
 *   den    = max(eps, d^T A d),  num = mu^T A d               A = Sigma^-1 (camera space)
 *   num <= 0 -> alpha = 0
 *   t      = num / den  (gamma, Eq. 16);  diff = t d - mu
 *   pdf    = exp(-1/2 diff^T A diff)    (G(gamma d))
 *   sig    = sqrt(1 / (den + eps))      (beta, Eq. 16)
 *   alpha  = min(0.99, 1 - exp(-2.506 kappa sig pdf))          (Eq. 19)
 *   skip alpha < 1/255; stop (exclusive) when T (1 - alpha) < 1e-4 (alphablend_volr_shader)
 *
 * Backward (analytic): x = 2.506 kappa sig pdf, dalpha/dx = exp(-x) (zero when clamped at 0.99).
 * t is the minimiser of q(t) = (t d - mu)^T A (t d - mu), so dq/dt = 0 (envelope theorem) and
 *   dq/dmu = -2 A diff,  dq/dA = diff diff^T,  dden/dA = d d^T (only if den > eps).
 * A is passed as 6 unique entries (a00 a01 a02 a11 a12 a22); off-diagonal gradients are doubled.
 * Like the slang code, a Gaussian's gradient at a pixel is dropped if it is non-finite (slang
 * checks NaN in d(inv_cov3d) only; this also drops inf and checks mean/kappa gradients).
 * Built with --use_fast_math (__expf, approximate division) unlike slang's default float math;
 * probes bound the difference to ~1e-5 relative vs a float64 reference.
 * Single camera, non-packed.
 */

#include <torch/extension.h>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <ATen/cuda/Atomic.cuh>

namespace cg = cooperative_groups;

#define VOL_ALPHA_THRESHOLD (1.f / 255.f)
#define VOL_EPS 1e-7f
#define VOL_SQRT_2PI 2.506f
#define VOL_MAX_ALPHA 0.99f
#define VOL_T_MIN 1e-4f

struct VolInter {
    float3 diff, Adiff;
    float den_raw, sig, pdf, x, a;
};

// Gaussian g: mk = (mu.x, mu.y, mu.z, kappa), pa = (a00, a01, a02), pb = (a11, a12, a22)
__device__ __forceinline__ bool vol_alpha(const float4 mk, const float3 pa, const float3 pb,
                                          const float3 d, float &alpha, VolInter &it) {
    const float3 m = {pa.x * d.x + pa.y * d.y + pa.z * d.z,
                      pa.y * d.x + pb.x * d.y + pb.y * d.z,
                      pa.z * d.x + pb.y * d.y + pb.z * d.z};
    it.den_raw = d.x * m.x + d.y * m.y + d.z * m.z;
    const float den = fmaxf(VOL_EPS, it.den_raw);
    const float num = mk.x * m.x + mk.y * m.y + mk.z * m.z;
    if (num <= 0.f) return false;
    const float t = num / den;
    it.diff = {t * d.x - mk.x, t * d.y - mk.y, t * d.z - mk.z};
    it.Adiff = {pa.x * it.diff.x + pa.y * it.diff.y + pa.z * it.diff.z,
                pa.y * it.diff.x + pb.x * it.diff.y + pb.y * it.diff.z,
                pa.z * it.diff.x + pb.y * it.diff.y + pb.z * it.diff.z};
    const float q = it.diff.x * it.Adiff.x + it.diff.y * it.Adiff.y + it.diff.z * it.Adiff.z;
    it.pdf = __expf(-0.5f * q);
    it.sig = sqrtf(1.0f / (den + VOL_EPS));
    it.x = VOL_SQRT_2PI * mk.w * it.sig * it.pdf;
    it.a = 1.0f - __expf(-it.x);
    alpha = fminf(VOL_MAX_ALPHA, it.a);
    return alpha >= VOL_ALPHA_THRESHOLD;
}

__device__ __forceinline__ float3 pixel_ray(const float px, const float py, const float4 intr) {
    float3 d = {(px - intr.z) / intr.x, (py - intr.w) / intr.y, 1.0f};
    const float inv = rsqrtf(d.x * d.x + d.y * d.y + d.z * d.z);
    return {d.x * inv, d.y * inv, d.z * inv};
}

template <uint32_t CDIM>
__global__ void vol_fwd_kernel(
    const uint32_t n_isects, const float4 *__restrict__ mk, const float3 *__restrict__ pa,
    const float3 *__restrict__ pb, const float *__restrict__ colors, const float4 intr,
    const uint32_t W, const uint32_t H, const uint32_t tile_size, const uint32_t tile_w,
    const uint32_t tile_h, const int32_t *__restrict__ tile_offsets,
    const int32_t *__restrict__ flatten_ids, float *__restrict__ render_colors,
    float *__restrict__ render_alphas, int32_t *__restrict__ last_ids) {
    auto block = cg::this_thread_block();
    const int32_t tile_id = block.group_index().y * tile_w + block.group_index().z;
    const uint32_t i = block.group_index().y * tile_size + block.thread_index().y;
    const uint32_t j = block.group_index().z * tile_size + block.thread_index().x;
    const float3 d = pixel_ray((float)j + 0.5f, (float)i + 0.5f, intr);
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
    float4 *mk_batch = reinterpret_cast<float4 *>(&id_batch[block_size]);
    float3 *pa_batch = reinterpret_cast<float3 *>(&mk_batch[block_size]);
    float3 *pb_batch = reinterpret_cast<float3 *>(&pa_batch[block_size]);

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
            mk_batch[tr] = mk[g];
            pa_batch[tr] = pa[g];
            pb_batch[tr] = pb[g];
        }
        block.sync();
        const uint32_t batch_size = min(block_size, range_end - batch_start);
        for (uint32_t t = 0; (t < batch_size) && !done; ++t) {
            float alpha;
            VolInter it;
            if (!vol_alpha(mk_batch[t], pa_batch[t], pb_batch[t], d, alpha, it)) continue;
            const float next_T = T * (1.0f - alpha);
            if (next_T < VOL_T_MIN) {  // official: new_pixel_state.a < 0.0001 -> break, exclusive
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
        for (uint32_t c = 0; c < CDIM; ++c) render_colors[pix_id * CDIM + c] = pix_out[c];
        last_ids[pix_id] = static_cast<int32_t>(cur_idx);
    }
}

template <uint32_t CDIM>
__global__ void vol_bwd_kernel(
    const uint32_t n_isects, const float4 *__restrict__ mk, const float3 *__restrict__ pa,
    const float3 *__restrict__ pb, const float *__restrict__ colors, const float4 intr,
    const uint32_t W, const uint32_t H, const uint32_t tile_size, const uint32_t tile_w,
    const uint32_t tile_h, const int32_t *__restrict__ tile_offsets,
    const int32_t *__restrict__ flatten_ids, const float *__restrict__ render_alphas,
    const int32_t *__restrict__ last_ids, const float *__restrict__ v_render_colors,
    const float *__restrict__ v_render_alphas, float *__restrict__ v_mk,
    float *__restrict__ v_pa, float *__restrict__ v_pb, float *__restrict__ v_colors) {
    auto block = cg::this_thread_block();
    const uint32_t tile_id = block.group_index().y * tile_w + block.group_index().z;
    const uint32_t i = block.group_index().y * tile_size + block.thread_index().y;
    const uint32_t j = block.group_index().z * tile_size + block.thread_index().x;
    const float3 d = pixel_ray((float)j + 0.5f, (float)i + 0.5f, intr);
    const int32_t pix_id = min(i * W + j, W * H - 1);
    const bool inside = (i < H && j < W);

    const int32_t range_start = tile_offsets[tile_id];
    const int32_t range_end =
        (tile_id == tile_w * tile_h - 1) ? n_isects : tile_offsets[tile_id + 1];
    const uint32_t block_size = block.size();
    const uint32_t num_batches = (range_end - range_start + block_size - 1) / block_size;

    extern __shared__ int s[];
    int32_t *id_batch = (int32_t *)s;
    float4 *mk_batch = reinterpret_cast<float4 *>(&id_batch[block_size]);
    float3 *pa_batch = reinterpret_cast<float3 *>(&mk_batch[block_size]);
    float3 *pb_batch = reinterpret_cast<float3 *>(&pa_batch[block_size]);
    float *rgbs_batch = (float *)&pb_batch[block_size];

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
            mk_batch[tr] = mk[g];
            pa_batch[tr] = pa[g];
            pb_batch[tr] = pb[g];
#pragma unroll
            for (uint32_t c = 0; c < CDIM; ++c) rgbs_batch[tr * CDIM + c] = colors[g * CDIM + c];
        }
        block.sync();
        for (uint32_t t = max(0, batch_end - warp_bin_final); t < batch_size; ++t) {
            bool valid = inside;
            if (batch_end - (int32_t)t > bin_final) valid = false;
            float alpha = 0.f;
            VolInter it;
            float4 g_mk = {0.f, 0.f, 0.f, 0.f};
            float3 g_pa = {0.f, 0.f, 0.f}, g_pb = {0.f, 0.f, 0.f};
            if (valid) {
                g_mk = mk_batch[t];
                g_pa = pa_batch[t];
                g_pb = pb_batch[t];
                valid = vol_alpha(g_mk, g_pa, g_pb, d, alpha, it);
            }
            if (!warp.any(valid)) continue;
            float v_rgb_local[CDIM] = {0.f};
            float v_mk_local[4] = {0.f, 0.f, 0.f, 0.f};
            float v_pa_local[3] = {0.f, 0.f, 0.f};
            float v_pb_local[3] = {0.f, 0.f, 0.f};
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
                if (it.a <= VOL_MAX_ALPHA) {  // min(0.99, a): no gradient through the clamp
                    const float g_x = v_alpha * __expf(-it.x);
                    const float v_kappa = g_x * VOL_SQRT_2PI * it.sig * it.pdf;
                    const float v_pdf = g_x * VOL_SQRT_2PI * g_mk.w * it.sig;
                    const float v_q = -0.5f * it.pdf * v_pdf;
                    const float v_sig = g_x * VOL_SQRT_2PI * g_mk.w * it.pdf;
                    const float v_den =
                        it.den_raw > VOL_EPS ? -0.5f * it.sig * it.sig * it.sig * v_sig : 0.f;
                    const float3 df = it.diff;
                    float lmk[4] = {-2.f * v_q * it.Adiff.x, -2.f * v_q * it.Adiff.y,
                                    -2.f * v_q * it.Adiff.z, v_kappa};
                    // G = v_q diff diff^T + v_den d d^T; off-diagonal entries doubled
                    float lpa[3] = {v_q * df.x * df.x + v_den * d.x * d.x,
                                    2.f * (v_q * df.x * df.y + v_den * d.x * d.y),
                                    2.f * (v_q * df.x * df.z + v_den * d.x * d.z)};
                    float lpb[3] = {v_q * df.y * df.y + v_den * d.y * d.y,
                                    2.f * (v_q * df.y * df.z + v_den * d.y * d.z),
                                    v_q * df.z * df.z + v_den * d.z * d.z};
                    bool finite = true;
#pragma unroll
                    for (int k = 0; k < 4; ++k) finite &= isfinite(lmk[k]);
#pragma unroll
                    for (int k = 0; k < 3; ++k) finite &= isfinite(lpa[k]) && isfinite(lpb[k]);
                    if (finite) {
#pragma unroll
                        for (int k = 0; k < 4; ++k) v_mk_local[k] = lmk[k];
#pragma unroll
                        for (int k = 0; k < 3; ++k) {
                            v_pa_local[k] = lpa[k];
                            v_pb_local[k] = lpb[k];
                        }
                    }
                }
#pragma unroll
                for (uint32_t c = 0; c < CDIM; ++c) buffer[c] += rgbs_batch[t * CDIM + c] * fac;
            }
#pragma unroll
            for (uint32_t c = 0; c < CDIM; ++c)
                v_rgb_local[c] = cg::reduce(warp, v_rgb_local[c], cg::plus<float>());
#pragma unroll
            for (int k = 0; k < 4; ++k) v_mk_local[k] = cg::reduce(warp, v_mk_local[k], cg::plus<float>());
#pragma unroll
            for (int k = 0; k < 3; ++k) {
                v_pa_local[k] = cg::reduce(warp, v_pa_local[k], cg::plus<float>());
                v_pb_local[k] = cg::reduce(warp, v_pb_local[k], cg::plus<float>());
            }
            if (warp.thread_rank() == 0) {
                const int32_t g = id_batch[t];
#pragma unroll
                for (uint32_t c = 0; c < CDIM; ++c)
                    gpuAtomicAdd(v_colors + CDIM * g + c, v_rgb_local[c]);
#pragma unroll
                for (int k = 0; k < 4; ++k) gpuAtomicAdd(v_mk + 4 * g + k, v_mk_local[k]);
#pragma unroll
                for (int k = 0; k < 3; ++k) {
                    gpuAtomicAdd(v_pa + 3 * g + k, v_pa_local[k]);
                    gpuAtomicAdd(v_pb + 3 * g + k, v_pb_local[k]);
                }
            }
        }
    }
}

static void check_f32(const at::Tensor &t, const char *name) {
    TORCH_CHECK(t.is_cuda() && t.is_contiguous() && t.scalar_type() == at::kFloat, name,
                " must be a contiguous float32 CUDA tensor");
}

static void check_common(const at::Tensor &mk, const at::Tensor &pa, const at::Tensor &pb,
                         const at::Tensor &colors, const at::Tensor &tile_offsets,
                         const at::Tensor &flatten_ids) {
    check_f32(mk, "means_kappa");
    check_f32(pa, "precis_a");
    check_f32(pb, "precis_b");
    check_f32(colors, "colors");
    const auto N = mk.size(0);
    TORCH_CHECK(mk.dim() == 2 && mk.size(1) == 4, "means_kappa must be [N,4]");
    TORCH_CHECK(pa.dim() == 2 && pa.size(1) == 3 && pa.size(0) == N, "precis_a must be [N,3]");
    TORCH_CHECK(pb.dim() == 2 && pb.size(1) == 3 && pb.size(0) == N, "precis_b must be [N,3]");
    TORCH_CHECK(colors.dim() == 2 && colors.size(0) == N, "colors must be [N,C]");
    TORCH_CHECK(tile_offsets.is_cuda() && tile_offsets.is_contiguous() &&
                    tile_offsets.scalar_type() == at::kInt && tile_offsets.dim() == 2,
                "tile_offsets must be contiguous int32 [th,tw]");
    TORCH_CHECK(flatten_ids.is_cuda() && flatten_ids.is_contiguous() &&
                    flatten_ids.scalar_type() == at::kInt,
                "flatten_ids must be contiguous int32");
}

#define VOL_DISPATCH(CD, ...)                                                                   \
    switch (CD) {                                                                               \
    case 1: { constexpr uint32_t CDIM = 1; __VA_ARGS__; break; }                                \
    case 3: { constexpr uint32_t CDIM = 3; __VA_ARGS__; break; }                                \
    case 4: { constexpr uint32_t CDIM = 4; __VA_ARGS__; break; }                                \
    default: TORCH_CHECK(false, "unsupported channel count ", CD, " (1, 3 or 4)");              \
    }

std::vector<at::Tensor> vol_forward(const at::Tensor &mk, const at::Tensor &pa,
                                    const at::Tensor &pb, const at::Tensor &colors,
                                    const double fx, const double fy, const double cx,
                                    const double cy, const int64_t W, const int64_t H,
                                    const int64_t tile_size, const at::Tensor &tile_offsets,
                                    const at::Tensor &flatten_ids) {
    check_common(mk, pa, pb, colors, tile_offsets, flatten_ids);
    const int64_t CD = colors.size(1);
    const uint32_t tile_h = tile_offsets.size(0), tile_w = tile_offsets.size(1);
    auto renders = at::empty({H, W, CD}, mk.options());
    auto alphas = at::empty({H, W, 1}, mk.options());
    auto last_ids = at::empty({H, W}, mk.options().dtype(at::kInt));
    const uint32_t n_isects = flatten_ids.size(0);
    const float4 intr = {(float)fx, (float)fy, (float)cx, (float)cy};
    dim3 threads = {(uint32_t)tile_size, (uint32_t)tile_size, 1};
    dim3 grid = {1, tile_h, tile_w};
    const int64_t shmem =
        tile_size * tile_size * (sizeof(int32_t) + sizeof(float4) + 2 * sizeof(float3));
    auto stream = at::cuda::getCurrentCUDAStream();
    VOL_DISPATCH(CD, {
        TORCH_CHECK(cudaFuncSetAttribute(vol_fwd_kernel<CDIM>,
                                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                                         shmem) == cudaSuccess,
                    "cannot set shared memory size");
        vol_fwd_kernel<CDIM><<<grid, threads, shmem, stream>>>(
            n_isects, (const float4 *)mk.data_ptr<float>(), (const float3 *)pa.data_ptr<float>(),
            (const float3 *)pb.data_ptr<float>(), colors.data_ptr<float>(), intr, W, H,
            tile_size, tile_w, tile_h, tile_offsets.data_ptr<int32_t>(),
            flatten_ids.data_ptr<int32_t>(), renders.data_ptr<float>(), alphas.data_ptr<float>(),
            last_ids.data_ptr<int32_t>());
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {renders, alphas, last_ids};
}

std::vector<at::Tensor> vol_backward(
    const at::Tensor &mk, const at::Tensor &pa, const at::Tensor &pb, const at::Tensor &colors,
    const double fx, const double fy, const double cx, const double cy, const int64_t W,
    const int64_t H, const int64_t tile_size, const at::Tensor &tile_offsets,
    const at::Tensor &flatten_ids, const at::Tensor &render_alphas, const at::Tensor &last_ids,
    const at::Tensor &v_render_colors, const at::Tensor &v_render_alphas) {
    check_common(mk, pa, pb, colors, tile_offsets, flatten_ids);
    check_f32(render_alphas, "render_alphas");
    check_f32(v_render_colors, "v_render_colors");
    check_f32(v_render_alphas, "v_render_alphas");
    const int64_t CD = colors.size(1);
    TORCH_CHECK(v_render_colors.numel() == H * W * CD, "v_render_colors must be [H,W,C]");
    TORCH_CHECK(v_render_alphas.numel() == H * W, "v_render_alphas must be [H,W,1]");
    auto v_mk = at::zeros_like(mk);
    auto v_pa = at::zeros_like(pa);
    auto v_pb = at::zeros_like(pb);
    auto v_colors = at::zeros_like(colors);
    const uint32_t n_isects = flatten_ids.size(0);
    if (n_isects == 0) return {v_mk, v_pa, v_pb, v_colors};
    const uint32_t tile_h = tile_offsets.size(0), tile_w = tile_offsets.size(1);
    const float4 intr = {(float)fx, (float)fy, (float)cx, (float)cy};
    dim3 threads = {(uint32_t)tile_size, (uint32_t)tile_size, 1};
    dim3 grid = {1, tile_h, tile_w};
    TORCH_CHECK((tile_size * tile_size) % 32 == 0, "tile must hold whole warps");
    auto stream = at::cuda::getCurrentCUDAStream();
    VOL_DISPATCH(CD, {
        const int64_t shmem = tile_size * tile_size *
                              (sizeof(int32_t) + sizeof(float4) + 2 * sizeof(float3) +
                               sizeof(float) * CDIM);
        TORCH_CHECK(cudaFuncSetAttribute(vol_bwd_kernel<CDIM>,
                                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                                         shmem) == cudaSuccess,
                    "cannot set shared memory size");
        vol_bwd_kernel<CDIM><<<grid, threads, shmem, stream>>>(
            n_isects, (const float4 *)mk.data_ptr<float>(), (const float3 *)pa.data_ptr<float>(),
            (const float3 *)pb.data_ptr<float>(), colors.data_ptr<float>(), intr, W, H,
            tile_size, tile_w, tile_h, tile_offsets.data_ptr<int32_t>(),
            flatten_ids.data_ptr<int32_t>(), render_alphas.data_ptr<float>(),
            last_ids.data_ptr<int32_t>(), v_render_colors.data_ptr<float>(),
            v_render_alphas.data_ptr<float>(), v_mk.data_ptr<float>(), v_pa.data_ptr<float>(),
            v_pb.data_ptr<float>(), v_colors.data_ptr<float>());
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {v_mk, v_pa, v_pb, v_colors};
}
