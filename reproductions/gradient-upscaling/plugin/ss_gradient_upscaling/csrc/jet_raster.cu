// Gradient-aware 3DGS rasterizer: renders the image together with its analytic spatial
// derivatives I_x, I_y, I_xy (Niedermayr et al., "Lightweight Gradient-Aware Upscaling of 3D
// Gaussian Splatting Images", ICCV 2025, arXiv 2503.14171, Sec. 5 and supplementary "3DGS
// Gradient Computation"), and the backward pass of all four outputs.
//
// No official CUDA implementation is public. Structure, tile traversal and alpha numerics are
// adapted from gsplat 1.5.3 cuda/csrc/RasterizeToPixels3DGS{Fwd,Bwd}.cu (Apache-2.0,
// Copyright the gsplat authors); the derivative recurrences follow the paper and the authors'
// viewer shader brush/crates/brush-render/src/shaders/rasterize.wgsl (Apache-2.0).
//
// Every per-pixel quantity is a "jet" (value, d/dx, d/dy, d2/dxdy) w.r.t. the pixel position.
// Jet multiplication is the product rule truncated at the mixed second derivative, so the
// paper's recurrences (supp. eqs. for B_i, A_i and their derivatives) are exactly
//   w_i = alpha_i (x) T_i,   C += c_i w_i,   T_{i+1} = T_i (x) (1 - alpha_i).
// The backward pass is reverse mode over these products, recovering T_i with the inversion
// trick T_i = T_{i+1} / (1 - alpha_i) in jet arithmetic (supp. "Inversion Trick").

#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

namespace cg = cooperative_groups;

#define ALPHA_THRESHOLD (1.f / 255.f)  // gsplat Common.h
#define CH 3

struct Jet {
    float v, x, y, xy;
};

__device__ inline Jet jmul(const Jet a, const Jet b) {
    return {a.v * b.v, a.v * b.x + a.x * b.v, a.v * b.y + a.y * b.v,
            a.v * b.xy + a.x * b.y + a.y * b.x + a.xy * b.v};
}

__device__ inline Jet jdiv(const Jet a, const Jet b) {  // q = a / b  <=>  a = q (x) b
    const float inv = 1.f / b.v;
    const float qv = a.v * inv;
    const float qx = (a.x - qv * b.x) * inv;
    const float qy = (a.y - qv * b.y) * inv;
    const float qxy = (a.xy - qv * b.xy - qx * b.y - qy * b.x) * inv;
    return {qv, qx, qy, qxy};
}

// alpha_i(x, y) = min(0.999, o exp(-sigma)) and its pixel-space derivatives
// (paper: alpha = sigma_i exp(g_i), g = -sigma here):
//   alpha_x = -alpha s_x, alpha_y = -alpha s_y, alpha_xy = alpha (s_x s_y - b),
//   s_x = a dx + b dy, s_y = b dx + c dy, d = pixel - mean, conic = (a, b, c).
// Where the 0.999 clamp is active alpha is constant, so its derivatives are zero.
struct AlphaEval {
    Jet al;
    float vis, dx, dy, sx, sy;
    bool clamped;
};

__device__ inline bool eval_alpha(const float mx, const float my, const float3 conic,
                                  const float opac, const float px, const float py,
                                  AlphaEval &e) {
    // same expression order as gsplat so that the value channel is bitwise gsplat's
    const float ddx = mx - px, ddy = my - py;
    const float sigma = 0.5f * (conic.x * ddx * ddx + conic.z * ddy * ddy) + conic.y * ddx * ddy;
    e.vis = __expf(-sigma);
    const float alpha = min(0.999f, opac * e.vis);
    if (sigma < 0.f || alpha < ALPHA_THRESHOLD) {
        return false;
    }
    e.dx = -ddx;
    e.dy = -ddy;
    e.sx = conic.x * e.dx + conic.y * e.dy;
    e.sy = conic.y * e.dx + conic.z * e.dy;
    e.clamped = opac * e.vis > 0.999f;
    if (e.clamped) {
        e.al = {alpha, 0.f, 0.f, 0.f};
    } else {
        e.al = {alpha, -alpha * e.sx, -alpha * e.sy, alpha * (e.sx * e.sy - conic.y)};
    }
    return true;
}

__global__ void jet_fwd_kernel(
    const uint32_t n_isects, const float2 *__restrict__ means2d, const float3 *__restrict__ conics,
    const float *__restrict__ colors, const float *__restrict__ opacities,
    const float *__restrict__ bg, const uint32_t W, const uint32_t H, const uint32_t tile_size,
    const uint32_t tile_w, const uint32_t tile_h, const int32_t *__restrict__ tile_offsets,
    const int32_t *__restrict__ flatten_ids, const float offx, const float offy,
    float *__restrict__ out,   // [4, H, W, 3]  (I, I_x, I_y, I_xy)
    float *__restrict__ tfin,  // [H, W, 4]     final transmittance jet
    int32_t *__restrict__ last_ids) {
    auto block = cg::this_thread_block();
    const int32_t tile_id = block.group_index().y * tile_w + block.group_index().z;
    const uint32_t i = block.group_index().y * tile_size + block.thread_index().y;
    const uint32_t j = block.group_index().z * tile_size + block.thread_index().x;
    const float px = (float)j + 0.5f + offx;
    const float py = (float)i + 0.5f + offy;
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

    Jet T = {1.f, 0.f, 0.f, 0.f};
    Jet C[CH];
#pragma unroll
    for (int k = 0; k < CH; ++k) C[k] = {0.f, 0.f, 0.f, 0.f};
    uint32_t cur_idx = 0;
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
            const float3 xyo = xy_opacity_batch[t];
            AlphaEval e;
            if (!eval_alpha(xyo.x, xyo.y, conic_batch[t], xyo.z, px, py, e)) continue;
            const float next_T = T.v * (1.0f - e.al.v);
            if (next_T <= 1e-4f) {  // gsplat: this pixel is done (exclusive)
                done = true;
                break;
            }
            const int32_t g = id_batch[t];
            const Jet w = jmul(e.al, T);  // T_i alpha_i and its derivatives
            const float *c = colors + g * CH;
#pragma unroll
            for (int k = 0; k < CH; ++k) {
                C[k].v += c[k] * w.v;
                C[k].x += c[k] * w.x;
                C[k].y += c[k] * w.y;
                C[k].xy += c[k] * w.xy;
            }
            cur_idx = batch_start + t;
            T = jmul(T, Jet{1.f - e.al.v, -e.al.x, -e.al.y, -e.al.xy});
            T.v = next_T;  // identical value, kept explicit for parity with gsplat
        }
    }

    if (inside) {
        const int64_t plane = (int64_t)H * W * CH;
#pragma unroll
        for (int k = 0; k < CH; ++k) {
            out[pix_id * CH + k] = C[k].v + T.v * bg[k];
            out[plane + pix_id * CH + k] = C[k].x + T.x * bg[k];
            out[2 * plane + pix_id * CH + k] = C[k].y + T.y * bg[k];
            out[3 * plane + pix_id * CH + k] = C[k].xy + T.xy * bg[k];
        }
        tfin[pix_id * 4 + 0] = T.v;
        tfin[pix_id * 4 + 1] = T.x;
        tfin[pix_id * 4 + 2] = T.y;
        tfin[pix_id * 4 + 3] = T.xy;
        last_ids[pix_id] = static_cast<int32_t>(cur_idx);
    }
}

__global__ void jet_bwd_kernel(
    const uint32_t n_isects, const float2 *__restrict__ means2d, const float3 *__restrict__ conics,
    const float *__restrict__ colors, const float *__restrict__ opacities,
    const float *__restrict__ bg, const uint32_t W, const uint32_t H, const uint32_t tile_size,
    const uint32_t tile_w, const uint32_t tile_h, const int32_t *__restrict__ tile_offsets,
    const int32_t *__restrict__ flatten_ids, const float offx, const float offy,
    const float *__restrict__ tfin, const int32_t *__restrict__ last_ids,
    const float *__restrict__ v_out,    // [4, H, W, 3]
    const float *__restrict__ v_alpha,  // [H, W]  (alpha = 1 - T_N)
    float2 *__restrict__ v_means2d, float3 *__restrict__ v_conics, float *__restrict__ v_colors,
    float *__restrict__ v_opacities) {
    auto block = cg::this_thread_block();
    const int32_t tile_id = block.group_index().y * tile_w + block.group_index().z;
    const uint32_t i = block.group_index().y * tile_size + block.thread_index().y;
    const uint32_t j = block.group_index().z * tile_size + block.thread_index().x;
    const float px = (float)j + 0.5f + offx;
    const float py = (float)i + 0.5f + offy;
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

    const Jet TN = {tfin[pix_id * 4 + 0], tfin[pix_id * 4 + 1], tfin[pix_id * 4 + 2],
                    tfin[pix_id * 4 + 3]};
    Jet T = TN;
    Jet buf[CH];  // sum_{j > i} c_j w_j  (colour already composited behind Gaussian i)
    Jet G[CH];    // dL/d(I, I_x, I_y, I_xy)
    Jet R0[CH];   // bg_k T_N
    const int64_t plane = (int64_t)H * W * CH;
#pragma unroll
    for (int k = 0; k < CH; ++k) {
        buf[k] = {0.f, 0.f, 0.f, 0.f};
        G[k] = {v_out[pix_id * CH + k], v_out[plane + pix_id * CH + k],
                v_out[2 * plane + pix_id * CH + k], v_out[3 * plane + pix_id * CH + k]};
        R0[k] = {bg[k] * TN.v, bg[k] * TN.x, bg[k] * TN.y, bg[k] * TN.xy};
    }
    const float v_a = v_alpha[pix_id];
    const int32_t bin_final = inside ? last_ids[pix_id] : 0;

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
            for (int k = 0; k < CH; ++k) rgbs_batch[tr * CH + k] = colors[g * CH + k];
        }
        block.sync();
        for (int32_t t = max(0, batch_end - warp_bin_final); t < batch_size; ++t) {
            bool valid = inside && (batch_end - t <= bin_final);
            AlphaEval e;
            float3 conic;
            float opac = 0.f;
            if (valid) {
                conic = conic_batch[t];
                const float3 xyo = xy_opacity_batch[t];
                opac = xyo.z;
                valid = eval_alpha(xyo.x, xyo.y, conic, opac, px, py, e);
            }
            if (!warp.any(valid)) continue;
            float v_rgb[CH] = {0.f, 0.f, 0.f};
            float3 v_conic = {0.f, 0.f, 0.f};
            float2 v_xy = {0.f, 0.f};
            float v_op = 0.f;
            if (valid) {
                const Jet om = {1.f - e.al.v, -e.al.x, -e.al.y, -e.al.xy};
                T = jdiv(T, om);  // inversion trick: T_i = T_{i+1} / (1 - alpha_i)
                const Jet w = jmul(e.al, T);
                // dL/dalpha-jet = adjoint of (M (x) .) with M_k = c_k T_i - (buf_k + bg_k T_N)/(1-alpha_i)
                Jet A = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
                for (int k = 0; k < CH; ++k) {
                    const float ck = rgbs_batch[t * CH + k];
                    v_rgb[k] = G[k].v * w.v + G[k].x * w.x + G[k].y * w.y + G[k].xy * w.xy;
                    const Jet behind = jdiv(Jet{buf[k].v + R0[k].v, buf[k].x + R0[k].x,
                                                buf[k].y + R0[k].y, buf[k].xy + R0[k].xy},
                                            om);
                    const Jet M = {ck * T.v - behind.v, ck * T.x - behind.x, ck * T.y - behind.y,
                                   ck * T.xy - behind.xy};
                    A.v += G[k].v * M.v + G[k].x * M.x + G[k].y * M.y + G[k].xy * M.xy;
                    A.x += G[k].x * M.v + G[k].xy * M.y;
                    A.y += G[k].y * M.v + G[k].xy * M.x;
                    A.xy += G[k].xy * M.v;
                }
                A.v += v_a * TN.v / om.v;  // alpha output = 1 - T_N
                if (!e.clamped) {
                    const float al = e.al.v;
                    // every alpha-jet component is proportional to o exp(-sigma)
                    const float lam =
                        A.v - A.x * e.sx - A.y * e.sy + A.xy * (e.sx * e.sy - conic.y);
                    v_op = lam * e.vis;
                    const float d_sig = -al * lam;
                    const float d_sx = -al * A.x + al * e.sy * A.xy;
                    const float d_sy = -al * A.y + al * e.sx * A.xy;
                    const float d_b = -al * A.xy;
                    v_conic = {d_sig * 0.5f * e.dx * e.dx + d_sx * e.dx,
                               d_sig * e.dx * e.dy + d_sx * e.dy + d_sy * e.dx + d_b,
                               d_sig * 0.5f * e.dy * e.dy + d_sy * e.dy};
                    const float d_dx = d_sig * e.sx + d_sx * conic.x + d_sy * conic.y;
                    const float d_dy = d_sig * e.sy + d_sx * conic.y + d_sy * conic.z;
                    v_xy = {-d_dx, -d_dy};  // d = pixel - mean
                }
#pragma unroll
                for (int k = 0; k < CH; ++k) {
                    const float ck = rgbs_batch[t * CH + k];
                    buf[k].v += ck * w.v;
                    buf[k].x += ck * w.x;
                    buf[k].y += ck * w.y;
                    buf[k].xy += ck * w.xy;
                }
            }
#pragma unroll
            for (int k = 0; k < CH; ++k) v_rgb[k] = cg::reduce(warp, v_rgb[k], cg::plus<float>());
            v_conic.x = cg::reduce(warp, v_conic.x, cg::plus<float>());
            v_conic.y = cg::reduce(warp, v_conic.y, cg::plus<float>());
            v_conic.z = cg::reduce(warp, v_conic.z, cg::plus<float>());
            v_xy.x = cg::reduce(warp, v_xy.x, cg::plus<float>());
            v_xy.y = cg::reduce(warp, v_xy.y, cg::plus<float>());
            v_op = cg::reduce(warp, v_op, cg::plus<float>());
            if (warp.thread_rank() == 0) {
                const int32_t g = id_batch[t];
#pragma unroll
                for (int k = 0; k < CH; ++k) atomicAdd(v_colors + CH * g + k, v_rgb[k]);
                float *vc = (float *)(v_conics) + 3 * g;
                atomicAdd(vc, v_conic.x);
                atomicAdd(vc + 1, v_conic.y);
                atomicAdd(vc + 2, v_conic.z);
                float *vx = (float *)(v_means2d) + 2 * g;
                atomicAdd(vx, v_xy.x);
                atomicAdd(vx + 1, v_xy.y);
                atomicAdd(v_opacities + g, v_op);
            }
        }
    }
}

static void check_inputs(const at::Tensor &means2d, const at::Tensor &conics,
                         const at::Tensor &colors, const at::Tensor &opacities,
                         const at::Tensor &bg, const at::Tensor &tile_offsets,
                         const at::Tensor &flatten_ids) {
    for (const auto *t : {&means2d, &conics, &colors, &opacities, &bg}) {
        TORCH_CHECK(t->is_cuda() && t->is_contiguous() && t->scalar_type() == at::kFloat,
                    "float32 contiguous CUDA tensors expected");
    }
    TORCH_CHECK(tile_offsets.scalar_type() == at::kInt && flatten_ids.scalar_type() == at::kInt,
                "int32 tile lists expected");
    TORCH_CHECK(colors.size(-1) == CH, "3 colour channels expected");
}

std::vector<at::Tensor> jet_forward(const at::Tensor &means2d, const at::Tensor &conics,
                                    const at::Tensor &colors, const at::Tensor &opacities,
                                    const at::Tensor &bg, const int64_t W, const int64_t H,
                                    const int64_t tile_size, const at::Tensor &tile_offsets,
                                    const at::Tensor &flatten_ids, const double offx,
                                    const double offy) {
    check_inputs(means2d, conics, colors, opacities, bg, tile_offsets, flatten_ids);
    const uint32_t tile_h = tile_offsets.size(0), tile_w = tile_offsets.size(1);
    auto opts = means2d.options();
    auto out = at::empty({4, H, W, CH}, opts);
    auto tfin = at::empty({H, W, 4}, opts);
    auto last = at::empty({H, W}, opts.dtype(at::kInt));
    const uint32_t n_isects = flatten_ids.size(0);
    dim3 threads(tile_size, tile_size, 1);
    dim3 grid(1, tile_h, tile_w);
    const uint32_t shmem = tile_size * tile_size * (sizeof(int32_t) + 2 * sizeof(float3));
    jet_fwd_kernel<<<grid, threads, shmem, at::cuda::getCurrentCUDAStream()>>>(
        n_isects, (const float2 *)means2d.data_ptr<float>(),
        (const float3 *)conics.data_ptr<float>(), colors.data_ptr<float>(),
        opacities.data_ptr<float>(), bg.data_ptr<float>(), W, H, tile_size, tile_w, tile_h,
        tile_offsets.data_ptr<int32_t>(), flatten_ids.data_ptr<int32_t>(), (float)offx,
        (float)offy, out.data_ptr<float>(), tfin.data_ptr<float>(), last.data_ptr<int32_t>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {out, tfin, last};
}

std::vector<at::Tensor> jet_backward(const at::Tensor &means2d, const at::Tensor &conics,
                                     const at::Tensor &colors, const at::Tensor &opacities,
                                     const at::Tensor &bg, const int64_t W, const int64_t H,
                                     const int64_t tile_size, const at::Tensor &tile_offsets,
                                     const at::Tensor &flatten_ids, const double offx,
                                     const double offy, const at::Tensor &tfin,
                                     const at::Tensor &last_ids, const at::Tensor &v_out,
                                     const at::Tensor &v_alpha) {
    check_inputs(means2d, conics, colors, opacities, bg, tile_offsets, flatten_ids);
    TORCH_CHECK(v_out.is_contiguous() && v_alpha.is_contiguous(), "contiguous grads expected");
    const uint32_t tile_h = tile_offsets.size(0), tile_w = tile_offsets.size(1);
    auto v_means2d = at::zeros_like(means2d);
    auto v_conics = at::zeros_like(conics);
    auto v_colors = at::zeros_like(colors);
    auto v_opacities = at::zeros_like(opacities);
    const uint32_t n_isects = flatten_ids.size(0);
    dim3 threads(tile_size, tile_size, 1);
    dim3 grid(1, tile_h, tile_w);
    const uint32_t shmem =
        tile_size * tile_size * (sizeof(int32_t) + 2 * sizeof(float3) + CH * sizeof(float));
    jet_bwd_kernel<<<grid, threads, shmem, at::cuda::getCurrentCUDAStream()>>>(
        n_isects, (const float2 *)means2d.data_ptr<float>(),
        (const float3 *)conics.data_ptr<float>(), colors.data_ptr<float>(),
        opacities.data_ptr<float>(), bg.data_ptr<float>(), W, H, tile_size, tile_w, tile_h,
        tile_offsets.data_ptr<int32_t>(), flatten_ids.data_ptr<int32_t>(), (float)offx,
        (float)offy, tfin.data_ptr<float>(), last_ids.data_ptr<int32_t>(),
        v_out.data_ptr<float>(), v_alpha.data_ptr<float>(),
        (float2 *)v_means2d.data_ptr<float>(), (float3 *)v_conics.data_ptr<float>(),
        v_colors.data_ptr<float>(), v_opacities.data_ptr<float>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {v_means2d, v_conics, v_colors, v_opacities};
}
