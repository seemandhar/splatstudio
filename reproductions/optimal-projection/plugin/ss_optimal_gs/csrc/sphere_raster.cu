/*
 * Unit-sphere based rasterizer of "On the Error Analysis of 3D Gaussian Splatting and an
 * Optimal Projection Strategy" (Huang et al., ECCV 2024, arXiv 2402.00752), Sec. 5.2, Eq. 16-17.
 *
 * Per-pixel math ported from the official pinhole rasterizer
 *   https://github.com/LetianHuang/op43dgs  submodules/diff-gaussian-rasterization-pinhole/
 *   cuda_rasterizer/forward.cu  renderCUDA  and  backward.cu  renderCUDA
 *   (Inria Gaussian-Splatting licence, non-commercial research use).
 * Tile traversal / shared-memory batching / back-to-front backward adapted from
 *   reproductions/linear-splatting/plugin/ss_linear_splatting/csrc/linear_raster.cu, itself
 *   adapted from gsplat v1.5.3 (Apache-2.0) RasterizeToPixels3DGS{Fwd,Bwd}.cu.
 *
 * mode 0 (optimal): pixel ray t = normalize((px-cx)/fx, (py-cy)/fy, 1); the Gaussian's ray mu
 *   from its 2D mean the same way; skip if mu.t < 1e-7; tangent-plane coordinates of the pixel
 *     u = f'x (e_u . t) / (mu . t),  v = f'y (e_v . t) / (mu . t)          (Eq. 16, Q of Eq. 18)
 *   with e_u = (cos phi, 0, -sin phi), e_v = (sin phi sin theta, cos theta, sin theta cos phi)
 *   (official atan2/sin/cos written algebraically), d = (0 - u, 0 - v), f' = max(512, f).
 * mode 1 (ewa control): d = mean2d - pixel (standard 3DGS planar evaluation).
 * Blending follows the Inria rasterizer the official code is built on: alpha = min(alpha_max,
 * o G) with the gradient passed straight through the clamp, skip power > 0 and alpha < 1/255,
 * stop when T (1 - alpha) < 1e-4 (that Gaussian excluded).
 * Backward: d(u, v)/d(mean2d) through mu(mean2d) by forward-mode dual numbers (exact; the
 * official backward.cu plugs the unit pixel ray into a closed form derived for the z = 1 ray,
 * so it is not exact: <=1.4% norm difference at p95, see report.md deviation 2; not replicated).
 * Single camera, non-packed. Pixel centres at +0.5 (gsplat convention, = Inria ndc2Pix).
 */

#include <torch/extension.h>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <ATen/cuda/Atomic.cuh>

namespace cg = cooperative_groups;

#define OP_ALPHA_THRESHOLD (1.f / 255.f)
#define OP_T_STOP 1e-4f
#define OP_DOT_EPS 1e-7f

struct CamParams {
    float fx, fy, cx, cy, fpx, fpy;
};

// value + derivatives w.r.t. the normalised mean coordinates (ax, ay)
struct Dual {
    float v, a, b;
};
__device__ __forceinline__ Dual dmk(float v, float a, float b) { return {v, a, b}; }
__device__ __forceinline__ Dual operator+(Dual x, Dual y) { return {x.v + y.v, x.a + y.a, x.b + y.b}; }
__device__ __forceinline__ Dual operator-(Dual x, Dual y) { return {x.v - y.v, x.a - y.a, x.b - y.b}; }
__device__ __forceinline__ Dual operator*(Dual x, Dual y) {
    return {x.v * y.v, x.a * y.v + x.v * y.a, x.b * y.v + x.v * y.b};
}
__device__ __forceinline__ Dual operator*(float s, Dual y) { return {s * y.v, s * y.a, s * y.b}; }
__device__ __forceinline__ Dual operator/(Dual x, Dual y) {
    const float inv = 1.f / y.v;
    const float q = x.v * inv;
    return {q, (x.a - q * y.a) * inv, (x.b - q * y.b) * inv};
}
__device__ __forceinline__ Dual dsqrt(Dual x) {
    const float s = sqrtf(x.v);
    const float h = 0.5f / s;
    return {s, x.a * h, x.b * h};
}

__device__ __forceinline__ float3 pixel_ray(const float px, const float py, const CamParams &c) {
    const float x = (px - c.cx) / c.fx, y = (py - c.cy) / c.fy;
    const float inv = rsqrtf(x * x + y * y + 1.f);
    return {x * inv, y * inv, inv};
}

// Tangent-plane coordinates (u, v) of pixel ray t on the plane of the Gaussian whose 2D mean is
// (mx, my). Returns false when the ray and the Gaussian are not in the same hemisphere.
__device__ __forceinline__ bool tangent_uv(const float mx, const float my, const float3 t,
                                           const CamParams &c, float &u, float &v) {
    const float ax = (mx - c.cx) / c.fx, ay = (my - c.cy) / c.fy;  // m = (ax, ay, 1)
    const float rho2 = ax * ax + 1.f;
    const float rho = sqrtf(rho2), r = sqrtf(rho2 + ay * ay);
    const float mdot = ax * t.x + ay * t.y + t.z;  // = r (mu . t)
    if (mdot < OP_DOT_EPS * r) return false;
    // u = f' (e_u . t)/(mu . t), e_u = (1, 0, -ax)/rho
    u = c.fpx * (t.x - ax * t.z) * r / (rho * mdot);
    // v = f' (e_v . t)/(mu . t), e_v = (-ax ay, rho^2, -ay)/(rho r)
    v = c.fpy * (-ax * ay * t.x + rho2 * t.y - ay * t.z) / (rho * mdot);
    return true;
}

// Same with derivatives w.r.t. the pixel-space mean (mx, my).
__device__ __forceinline__ void tangent_uv_grad(const float mx, const float my, const float3 t,
                                                const CamParams &c, Dual &u, Dual &v) {
    const Dual ax = dmk((mx - c.cx) / c.fx, 1.f, 0.f), ay = dmk((my - c.cy) / c.fy, 0.f, 1.f);
    const Dual one = dmk(1.f, 0.f, 0.f);
    const Dual rho2 = ax * ax + one;
    const Dual rho = dsqrt(rho2), r = dsqrt(rho2 + ay * ay);
    const Dual mdot = t.x * ax + t.y * ay + dmk(t.z, 0.f, 0.f);
    const Dual den = rho * mdot;
    u = c.fpx * (((dmk(t.x, 0.f, 0.f) - t.z * ax) * r) / den);
    v = c.fpy * ((t.y * rho2 - t.x * (ax * ay) - t.z * ay) / den);
    // chain to pixel units
    const float ifx = 1.f / c.fx, ify = 1.f / c.fy;
    u.a *= ifx; u.b *= ify;
    v.a *= ifx; v.b *= ify;
}

// returns false when this Gaussian is skipped for this pixel (Inria `continue`)
template <int MODE>
__device__ __forceinline__ bool eval_alpha(const float3 conic, const float3 xyo, const float px,
                                           const float py, const float3 t, const CamParams &c,
                                           const float alpha_max, float2 &d, float &G,
                                           float &alpha) {
    if (MODE == 0) {
        float u, v;
        if (!tangent_uv(xyo.x, xyo.y, t, c, u, v)) return false;
        d = {-u, -v};
    } else {
        d = {xyo.x - px, xyo.y - py};
    }
    const float power = -0.5f * (conic.x * d.x * d.x + conic.z * d.y * d.y) - conic.y * d.x * d.y;
    if (power > 0.f) return false;
    G = __expf(power);
    alpha = fminf(alpha_max, xyo.z * G);
    return alpha >= OP_ALPHA_THRESHOLD;
}

template <uint32_t CDIM, int MODE>
__global__ void sphere_fwd_kernel(
    const uint32_t n_isects, const float2 *__restrict__ means2d,
    const float3 *__restrict__ conics, const float *__restrict__ colors,
    const float *__restrict__ opacities, const uint32_t W, const uint32_t H,
    const uint32_t tile_size, const uint32_t tile_w, const uint32_t tile_h,
    const int32_t *__restrict__ tile_offsets, const int32_t *__restrict__ flatten_ids,
    const CamParams cam, const float alpha_max, float *__restrict__ render_colors,
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
    const float3 t = pixel_ray(px, py, cam);

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
        for (uint32_t k = 0; (k < batch_size) && !done; ++k) {
            float2 d;
            float G, alpha;
            if (!eval_alpha<MODE>(conic_batch[k], xy_opacity_batch[k], px, py, t, cam, alpha_max,
                                  d, G, alpha))
                continue;
            const float next_T = T * (1.0f - alpha);
            if (next_T < OP_T_STOP) {  // Inria: `test_T < 0.0001f` -> done, Gaussian excluded
                done = true;
                break;
            }
            const int32_t g = id_batch[k];
            const float vis = alpha * T;
            const float *c_ptr = colors + g * CDIM;
#pragma unroll
            for (uint32_t ch = 0; ch < CDIM; ++ch) pix_out[ch] += c_ptr[ch] * vis;
            cur_idx = batch_start + k;
            T = next_T;
        }
    }
    if (inside) {
        render_alphas[pix_id] = 1.0f - T;
#pragma unroll
        for (uint32_t ch = 0; ch < CDIM; ++ch) render_colors[pix_id * CDIM + ch] = pix_out[ch];
        last_ids[pix_id] = static_cast<int32_t>(cur_idx);
    }
}

template <uint32_t CDIM, int MODE>
__global__ void sphere_bwd_kernel(
    const uint32_t n_isects, const float2 *__restrict__ means2d,
    const float3 *__restrict__ conics, const float *__restrict__ colors,
    const float *__restrict__ opacities, const uint32_t W, const uint32_t H,
    const uint32_t tile_size, const uint32_t tile_w, const uint32_t tile_h,
    const int32_t *__restrict__ tile_offsets, const int32_t *__restrict__ flatten_ids,
    const CamParams cam, const float alpha_max, const float *__restrict__ render_alphas,
    const int32_t *__restrict__ last_ids, const float *__restrict__ v_render_colors,
    const float *__restrict__ v_render_alphas, float2 *__restrict__ v_means2d,
    float3 *__restrict__ v_conics, float *__restrict__ v_colors,
    float *__restrict__ v_opacities) {
    auto block = cg::this_thread_block();
    const uint32_t tile_id = block.group_index().y * tile_w + block.group_index().z;
    const uint32_t i = block.group_index().y * tile_size + block.thread_index().y;
    const uint32_t j = block.group_index().z * tile_size + block.thread_index().x;
    const float px = (float)j + 0.5f;
    const float py = (float)i + 0.5f;
    const int32_t pix_id = min(i * W + j, W * H - 1);
    const bool inside = (i < H && j < W);
    const float3 t = pixel_ray(px, py, cam);

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
    for (uint32_t ch = 0; ch < CDIM; ++ch) v_render_c[ch] = v_render_colors[pix_id * CDIM + ch];
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
            for (uint32_t ch = 0; ch < CDIM; ++ch)
                rgbs_batch[tr * CDIM + ch] = colors[g * CDIM + ch];
        }
        block.sync();
        for (uint32_t k = max(0, batch_end - warp_bin_final); k < batch_size; ++k) {
            bool valid = inside;
            if (batch_end - (int32_t)k > bin_final) valid = false;
            float alpha = 0.f, G = 0.f, opac = 0.f;
            float2 d = {0.f, 0.f};
            float3 conic = {0.f, 0.f, 0.f};
            float3 xyo = {0.f, 0.f, 0.f};
            if (valid) {
                conic = conic_batch[k];
                xyo = xy_opacity_batch[k];
                opac = xyo.z;
                valid = eval_alpha<MODE>(conic, xyo, px, py, t, cam, alpha_max, d, G, alpha);
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
                for (uint32_t ch = 0; ch < CDIM; ++ch) v_rgb_local[ch] = fac * v_render_c[ch];
                float v_alpha = 0.f;
#pragma unroll
                for (uint32_t ch = 0; ch < CDIM; ++ch)
                    v_alpha += (rgbs_batch[k * CDIM + ch] * T - buffer[ch] * ra) * v_render_c[ch];
                v_alpha += T_final * ra * v_render_a;
                // Inria: no saturation gate; dalpha/d(oG) = 1 even when clamped
                const float v_G = opac * v_alpha;
                const float v_power = G * v_G;
                // power = -1/2 (a dx^2 + c dy^2) - b dx dy
                v_conic_local = {-0.5f * d.x * d.x * v_power, -d.x * d.y * v_power,
                                 -0.5f * d.y * d.y * v_power};
                const float dp_ddx = -(conic.x * d.x + conic.y * d.y);
                const float dp_ddy = -(conic.z * d.y + conic.y * d.x);
                if (MODE == 0) {
                    Dual u, v;
                    tangent_uv_grad(xyo.x, xyo.y, t, cam, u, v);
                    // d = (-u, -v)
                    v_xy_local = {-v_power * (dp_ddx * u.a + dp_ddy * v.a),
                                  -v_power * (dp_ddx * u.b + dp_ddy * v.b)};
                } else {
                    v_xy_local = {v_power * dp_ddx, v_power * dp_ddy};
                }
                v_opacity_local = G * v_alpha;
#pragma unroll
                for (uint32_t ch = 0; ch < CDIM; ++ch) buffer[ch] += rgbs_batch[k * CDIM + ch] * fac;
            }
#pragma unroll
            for (uint32_t ch = 0; ch < CDIM; ++ch)
                v_rgb_local[ch] = cg::reduce(warp, v_rgb_local[ch], cg::plus<float>());
            v_conic_local.x = cg::reduce(warp, v_conic_local.x, cg::plus<float>());
            v_conic_local.y = cg::reduce(warp, v_conic_local.y, cg::plus<float>());
            v_conic_local.z = cg::reduce(warp, v_conic_local.z, cg::plus<float>());
            v_xy_local.x = cg::reduce(warp, v_xy_local.x, cg::plus<float>());
            v_xy_local.y = cg::reduce(warp, v_xy_local.y, cg::plus<float>());
            v_opacity_local = cg::reduce(warp, v_opacity_local, cg::plus<float>());
            if (warp.thread_rank() == 0) {
                const int32_t g = id_batch[k];
                float *v_rgb_ptr = v_colors + CDIM * g;
#pragma unroll
                for (uint32_t ch = 0; ch < CDIM; ++ch) gpuAtomicAdd(v_rgb_ptr + ch, v_rgb_local[ch]);
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
                         const at::Tensor &tile_offsets, const at::Tensor &flatten_ids,
                         const std::vector<double> &cam, const int64_t mode) {
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
    TORCH_CHECK(cam.size() == 6, "cam must be (fx, fy, cx, cy, fpx, fpy)");
    TORCH_CHECK(mode == 0 || mode == 1, "mode must be 0 (optimal) or 1 (ewa)");
}

static CamParams to_cam(const std::vector<double> &c) {
    return {(float)c[0], (float)c[1], (float)c[2], (float)c[3], (float)c[4], (float)c[5]};
}

#define OP_DISPATCH(CD, MD, ...)                                                               \
    do {                                                                                       \
        if (MD == 0) {                                                                         \
            constexpr int MODE = 0;                                                            \
            switch (CD) {                                                                      \
            case 3: { constexpr uint32_t CDIM = 3; __VA_ARGS__; break; }                       \
            case 4: { constexpr uint32_t CDIM = 4; __VA_ARGS__; break; }                       \
            default: TORCH_CHECK(false, "unsupported channel count ", CD, " (3 or 4)");        \
            }                                                                                  \
        } else {                                                                               \
            constexpr int MODE = 1;                                                            \
            switch (CD) {                                                                      \
            case 3: { constexpr uint32_t CDIM = 3; __VA_ARGS__; break; }                       \
            case 4: { constexpr uint32_t CDIM = 4; __VA_ARGS__; break; }                       \
            default: TORCH_CHECK(false, "unsupported channel count ", CD, " (3 or 4)");        \
            }                                                                                  \
        }                                                                                      \
    } while (0)

std::vector<at::Tensor> sphere_forward(const at::Tensor &means2d, const at::Tensor &conics,
                                       const at::Tensor &colors, const at::Tensor &opacities,
                                       const int64_t W, const int64_t H, const int64_t tile_size,
                                       const at::Tensor &tile_offsets,
                                       const at::Tensor &flatten_ids,
                                       const std::vector<double> &cam, const int64_t mode,
                                       const double alpha_max) {
    check_common(means2d, conics, colors, opacities, tile_offsets, flatten_ids, cam, mode);
    const int64_t CD = colors.size(1);
    const uint32_t tile_h = tile_offsets.size(0), tile_w = tile_offsets.size(1);
    auto renders = at::empty({H, W, CD}, means2d.options());
    auto alphas = at::empty({H, W, 1}, means2d.options());
    auto last_ids = at::empty({H, W}, means2d.options().dtype(at::kInt));
    const uint32_t n_isects = flatten_ids.size(0);
    dim3 threads = {(uint32_t)tile_size, (uint32_t)tile_size, 1};
    dim3 grid = {1, tile_h, tile_w};
    const int64_t shmem = tile_size * tile_size * (sizeof(int32_t) + 2 * sizeof(float3));
    const CamParams cp = to_cam(cam);
    auto stream = at::cuda::getCurrentCUDAStream();
    OP_DISPATCH(CD, mode, {
        TORCH_CHECK(cudaFuncSetAttribute(sphere_fwd_kernel<CDIM, MODE>,
                                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                                         shmem) == cudaSuccess,
                    "cannot set shared memory size");
        sphere_fwd_kernel<CDIM, MODE><<<grid, threads, shmem, stream>>>(
            n_isects, (const float2 *)means2d.data_ptr<float>(),
            (const float3 *)conics.data_ptr<float>(), colors.data_ptr<float>(),
            opacities.data_ptr<float>(), W, H, tile_size, tile_w, tile_h,
            tile_offsets.data_ptr<int32_t>(), flatten_ids.data_ptr<int32_t>(), cp,
            (float)alpha_max, renders.data_ptr<float>(), alphas.data_ptr<float>(),
            last_ids.data_ptr<int32_t>());
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {renders, alphas, last_ids};
}

std::vector<at::Tensor> sphere_backward(
    const at::Tensor &means2d, const at::Tensor &conics, const at::Tensor &colors,
    const at::Tensor &opacities, const int64_t W, const int64_t H, const int64_t tile_size,
    const at::Tensor &tile_offsets, const at::Tensor &flatten_ids,
    const std::vector<double> &cam, const int64_t mode, const double alpha_max,
    const at::Tensor &render_alphas, const at::Tensor &last_ids,
    const at::Tensor &v_render_colors, const at::Tensor &v_render_alphas) {
    check_common(means2d, conics, colors, opacities, tile_offsets, flatten_ids, cam, mode);
    check_f32(render_alphas, "render_alphas");
    check_f32(v_render_colors, "v_render_colors");
    check_f32(v_render_alphas, "v_render_alphas");
    const int64_t CD = colors.size(1);
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
    const CamParams cp = to_cam(cam);
    auto stream = at::cuda::getCurrentCUDAStream();
    OP_DISPATCH(CD, mode, {
        const int64_t shmem =
            tile_size * tile_size * (sizeof(int32_t) + 2 * sizeof(float3) + sizeof(float) * CDIM);
        TORCH_CHECK(cudaFuncSetAttribute(sphere_bwd_kernel<CDIM, MODE>,
                                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                                         shmem) == cudaSuccess,
                    "cannot set shared memory size");
        sphere_bwd_kernel<CDIM, MODE><<<grid, threads, shmem, stream>>>(
            n_isects, (const float2 *)means2d.data_ptr<float>(),
            (const float3 *)conics.data_ptr<float>(), colors.data_ptr<float>(),
            opacities.data_ptr<float>(), W, H, tile_size, tile_w, tile_h,
            tile_offsets.data_ptr<int32_t>(), flatten_ids.data_ptr<int32_t>(), cp,
            (float)alpha_max, render_alphas.data_ptr<float>(), last_ids.data_ptr<int32_t>(),
            v_render_colors.data_ptr<float>(), v_render_alphas.data_ptr<float>(),
            (float2 *)v_means2d.data_ptr<float>(), (float3 *)v_conics.data_ptr<float>(),
            v_colors.data_ptr<float>(), v_opacities.data_ptr<float>());
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {v_means2d, v_conics, v_colors, v_opacities};
}
