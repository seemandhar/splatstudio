/*
 * EfficientGS per-Gaussian statistics on gsplat 1.5.3 tile lists.
 *
 * Adapted from the official EfficientGS rasterizer
 *   https://github.com/wenkaier/EfficientGS (commit 1f70babe)
 *   submodules/diff-gaussian-rasterization/cuda_rasterizer/backward.cu  (dL_norm, Eq. 6)
 *   submodules/diff-gaussian-rasterization/cuda_rasterizer/forward.cu   (top_k max_weight_pids,
 *                                                                        loss_weight, Eq. 8/9)
 * which derives from the Inria 3DGS rasterizer (Copyright (C) 2023, Inria, GRAPHDECO;
 * non-commercial research licence, see LICENSE-3DGS.md), and from gsplat's
 * RasterizeToPixels3DGSBwd.cu (Apache-2.0, https://github.com/nerfstudio-project/gsplat v1.5.3)
 * for the backward formula. Tile traversal layout follows the mini-splatting plugin kernel
 * in this workspace (reproductions/mini-splatting/plugin/ss_mini_splatting/csrc/ms_raster.cu).
 *
 * Numerics are gsplat's (alpha = min(0.999, o exp(-sigma)), skip sigma < 0 or alpha < 1/255,
 * stop when the next transmittance <= 1e-4, pixel centres at +0.5), so statistics describe the
 * image gsplat actually renders and differentiates.
 */

#include <torch/extension.h>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>

namespace cg = cooperative_groups;

#define EGS_ALPHA_THRESHOLD (1.f / 255.f)
#define EGS_MAX_K 10

// --------------------------------------------------------------------------------------------
// Eq. 6: S_g contribution of one view = sum over pixels of || per-pixel d L / d mean2d_ndc ||.
// Pass 1 recomputes the pixel's colour and final transmittance; pass 2 walks front to back and
// evaluates gsplat's backward formula with buffer (= colour of the Gaussians behind) obtained as
// C_total - accumulated colour.
// --------------------------------------------------------------------------------------------
__global__ void egs_gradnorm_kernel(
    const int n_isects, const int W, const int H, const int tile_size, const int tile_w,
    const int tile_h,
    const float2 *__restrict__ means2d, const float3 *__restrict__ conics,
    const float *__restrict__ opacities, const float3 *__restrict__ colors,
    const int *__restrict__ tile_offsets, const int *__restrict__ flatten_ids,
    const float3 *__restrict__ v_colors,  // [H,W,3] dL / d raw rendered colour
    const float *__restrict__ v_alphas,   // [H,W]   dL / d rendered alpha
    const float scale_x, const float scale_y,  // pixel -> NDC gradient factors (W/2, H/2)
    float *__restrict__ out_norm,         // [N]
    float2 *__restrict__ out_sum          // [N] sum of per-pixel vectors (pixel units)
) {
    auto block = cg::this_thread_block();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);
    const int tile_x = blockIdx.x, tile_y = blockIdx.y;
    const int tile_id = tile_y * tile_w + tile_x;
    const int i = tile_y * tile_size + threadIdx.y;
    const int j = tile_x * tile_size + threadIdx.x;
    const float px = (float)j + 0.5f;
    const float py = (float)i + 0.5f;
    const bool inside = (i < H && j < W);
    const int pix_id = inside ? i * W + j : 0;

    const int range_start = tile_offsets[tile_id];
    const int range_end = (tile_id == tile_w * tile_h - 1) ? n_isects : tile_offsets[tile_id + 1];
    const int block_size = blockDim.x * blockDim.y;
    const int num_batches = (range_end - range_start + block_size - 1) / block_size;

    extern __shared__ int s[];
    int *id_batch = (int *)s;
    float3 *xy_opacity_batch = reinterpret_cast<float3 *>(&id_batch[block_size]);
    float3 *conic_batch = reinterpret_cast<float3 *>(&xy_opacity_batch[block_size]);
    float3 *rgb_batch = reinterpret_cast<float3 *>(&conic_batch[block_size]);
    const int tr = block.thread_rank();

    // ---------------- pass 1: forward ----------------
    float T = 1.0f;
    float C0 = 0.f, C1 = 0.f, C2 = 0.f;
    bool done = !inside;
    for (int b = 0; b < num_batches; ++b) {
        if (__syncthreads_count(done) >= block_size) break;
        const int batch_start = range_start + block_size * b;
        const int idx = batch_start + tr;
        if (idx < range_end) {
            const int g = flatten_ids[idx];
            id_batch[tr] = g;
            const float2 xy = means2d[g];
            xy_opacity_batch[tr] = {xy.x, xy.y, opacities[g]};
            conic_batch[tr] = conics[g];
            rgb_batch[tr] = colors[g];
        }
        block.sync();
        const int batch_size = min(block_size, range_end - batch_start);
        for (int t = 0; (t < batch_size) && !done; ++t) {
            const float3 conic = conic_batch[t];
            const float3 xyo = xy_opacity_batch[t];
            const float dx = xyo.x - px, dy = xyo.y - py;
            const float sigma = 0.5f * (conic.x * dx * dx + conic.z * dy * dy) + conic.y * dx * dy;
            const float alpha = min(0.999f, xyo.z * __expf(-sigma));
            if (sigma < 0.f || alpha < EGS_ALPHA_THRESHOLD) continue;
            const float next_T = T * (1.0f - alpha);
            if (next_T <= 1e-4f) {
                done = true;
                break;
            }
            const float vis = alpha * T;
            const float3 c = rgb_batch[t];
            C0 += c.x * vis; C1 += c.y * vis; C2 += c.z * vis;
            T = next_T;
        }
    }
    const float T_final = T;

    // ---------------- pass 2: per-pixel gradients ----------------
    float3 vc = {0.f, 0.f, 0.f};
    float va = 0.f;
    if (inside) {
        vc = v_colors[pix_id];
        va = v_alphas[pix_id];
    }
    T = 1.0f;
    float A0 = 0.f, A1 = 0.f, A2 = 0.f;  // accumulated colour including the current Gaussian
    done = !inside;
    for (int b = 0; b < num_batches; ++b) {
        if (__syncthreads_count(done) >= block_size) break;
        const int batch_start = range_start + block_size * b;
        const int idx = batch_start + tr;
        block.sync();
        if (idx < range_end) {
            const int g = flatten_ids[idx];
            id_batch[tr] = g;
            const float2 xy = means2d[g];
            xy_opacity_batch[tr] = {xy.x, xy.y, opacities[g]};
            conic_batch[tr] = conics[g];
            rgb_batch[tr] = colors[g];
        }
        block.sync();
        const int batch_size = min(block_size, range_end - batch_start);
        for (int t = 0; t < batch_size; ++t) {
            bool valid = !done;
            float alpha = 0.f, vis = 0.f, opac = 0.f, dx = 0.f, dy = 0.f;
            float3 conic = {0.f, 0.f, 0.f};
            if (valid) {
                conic = conic_batch[t];
                const float3 xyo = xy_opacity_batch[t];
                opac = xyo.z;
                dx = xyo.x - px;
                dy = xyo.y - py;
                const float sigma =
                    0.5f * (conic.x * dx * dx + conic.z * dy * dy) + conic.y * dx * dy;
                vis = __expf(-sigma);
                alpha = min(0.999f, opac * vis);
                if (sigma < 0.f || alpha < EGS_ALPHA_THRESHOLD) valid = false;
            }
            if (valid && T * (1.0f - alpha) <= 1e-4f) {
                done = true;
                valid = false;
            }
            if (!warp.any(valid)) continue;
            float gx = 0.f, gy = 0.f, nrm = 0.f;
            if (valid) {
                const float3 c = rgb_batch[t];
                const float w = alpha * T;
                A0 += c.x * w; A1 += c.y * w; A2 += c.z * w;
                const float ra = 1.0f / (1.0f - alpha);
                // gsplat: v_alpha = sum_k (c_k T_i - buffer_k ra) v_k + T_final ra v_a
                const float v_alpha = (c.x * T - (C0 - A0) * ra) * vc.x +
                                      (c.y * T - (C1 - A1) * ra) * vc.y +
                                      (c.z * T - (C2 - A2) * ra) * vc.z + T_final * ra * va;
                if (opac * vis <= 0.999f) {
                    const float v_sigma = -opac * vis * v_alpha;
                    gx = v_sigma * (conic.x * dx + conic.y * dy);
                    gy = v_sigma * (conic.y * dx + conic.z * dy);
                    const float nx = gx * scale_x, ny = gy * scale_y;
                    nrm = sqrtf(nx * nx + ny * ny);  // official backward.cu: dL_norm
                }
                T *= (1.0f - alpha);
            }
            nrm = cg::reduce(warp, nrm, cg::plus<float>());
            gx = cg::reduce(warp, gx, cg::plus<float>());
            gy = cg::reduce(warp, gy, cg::plus<float>());
            if (warp.thread_rank() == 0) {
                const int g = id_batch[t];
                atomicAdd(&out_norm[g], nrm);
                atomicAdd(&out_sum[g].x, gx);
                atomicAdd(&out_sum[g].y, gy);
            }
        }
    }
}

// --------------------------------------------------------------------------------------------
// Eq. 8 (top-K dominant flags) and Eq. 9 (colour-error weights) of one view.
// --------------------------------------------------------------------------------------------
__global__ void egs_stats_kernel(
    const int n_isects, const int W, const int H, const int tile_size, const int tile_w,
    const int tile_h,
    const float2 *__restrict__ means2d, const float3 *__restrict__ conics,
    const float *__restrict__ opacities, const int *__restrict__ tile_offsets,
    const int *__restrict__ flatten_ids, const int top_k, const bool weight_after_T,
    const float3 *__restrict__ pix_loss,  // [H,W,3] or nullptr
    int *__restrict__ dominant,           // [N]
    float *__restrict__ loss_weight       // [N]
) {
    auto block = cg::this_thread_block();
    const int tile_x = blockIdx.x, tile_y = blockIdx.y;
    const int tile_id = tile_y * tile_w + tile_x;
    const int i = tile_y * tile_size + threadIdx.y;
    const int j = tile_x * tile_size + threadIdx.x;
    const float px = (float)j + 0.5f;
    const float py = (float)i + 0.5f;
    const bool inside = (i < H && j < W);
    bool done = !inside;

    const int range_start = tile_offsets[tile_id];
    const int range_end = (tile_id == tile_w * tile_h - 1) ? n_isects : tile_offsets[tile_id + 1];
    const int block_size = blockDim.x * blockDim.y;
    const int num_batches = (range_end - range_start + block_size - 1) / block_size;

    extern __shared__ int s[];
    int *id_batch = (int *)s;
    float3 *xy_opacity_batch = reinterpret_cast<float3 *>(&id_batch[block_size]);
    float3 *conic_batch = reinterpret_cast<float3 *>(&xy_opacity_batch[block_size]);
    const int tr = block.thread_rank();

    float ploss = 0.f;
    if (inside && pix_loss != nullptr) {
        const float3 l = pix_loss[i * W + j];
        ploss = l.x + l.y + l.z;  // official: sum over channels of pix_loss[ch] * weight
    }
    float max_w[EGS_MAX_K];
    int max_id[EGS_MAX_K];
    for (int k = 0; k < EGS_MAX_K; ++k) {
        max_w[k] = -1.0f;
        max_id[k] = -1;
    }
    float T = 1.0f;
    for (int b = 0; b < num_batches; ++b) {
        if (__syncthreads_count(done) >= block_size) break;
        const int batch_start = range_start + block_size * b;
        const int idx = batch_start + tr;
        if (idx < range_end) {
            const int g = flatten_ids[idx];
            id_batch[tr] = g;
            const float2 xy = means2d[g];
            xy_opacity_batch[tr] = {xy.x, xy.y, opacities[g]};
            conic_batch[tr] = conics[g];
        }
        block.sync();
        const int batch_size = min(block_size, range_end - batch_start);
        for (int t = 0; (t < batch_size) && !done; ++t) {
            const float3 conic = conic_batch[t];
            const float3 xyo = xy_opacity_batch[t];
            const float dx = xyo.x - px, dy = xyo.y - py;
            const float sigma = 0.5f * (conic.x * dx * dx + conic.z * dy * dy) + conic.y * dx * dy;
            const float alpha = min(0.999f, xyo.z * __expf(-sigma));
            if (sigma < 0.f || alpha < EGS_ALPHA_THRESHOLD) continue;
            const float next_T = T * (1.0f - alpha);
            if (next_T <= 1e-4f) {
                done = true;
                break;
            }
            const int g = id_batch[t];
            // official forward.cu updates T before computing `weight = alpha * T`
            const float w = alpha * (weight_after_T ? next_T : T);
            if (pix_loss != nullptr) atomicAdd(&loss_weight[g], ploss * w);
            for (int kk = 0; kk < top_k; ++kk) {
                if (w > max_w[kk]) {
                    for (int k3 = top_k - 1; k3 > kk; --k3) {
                        max_w[k3] = max_w[k3 - 1];
                        max_id[k3] = max_id[k3 - 1];
                    }
                    max_w[kk] = w;
                    max_id[kk] = g;
                    break;
                }
            }
            T = next_T;
        }
    }
    if (inside) {
        for (int k = 0; k < top_k; ++k) {
            if (max_id[k] < 0) break;
            dominant[max_id[k]] = 1;
        }
    }
}

static void check_input(const torch::Tensor &t, const char *name, torch::ScalarType type) {
    TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
    TORCH_CHECK(t.scalar_type() == type, name, " has the wrong dtype");
}

static void check_common(const torch::Tensor &means2d, const torch::Tensor &conics,
                         const torch::Tensor &opacities, const torch::Tensor &tile_offsets,
                         const torch::Tensor &flatten_ids) {
    check_input(means2d, "means2d", torch::kFloat32);
    check_input(conics, "conics", torch::kFloat32);
    check_input(opacities, "opacities", torch::kFloat32);
    check_input(tile_offsets, "tile_offsets", torch::kInt32);
    check_input(flatten_ids, "flatten_ids", torch::kInt32);
}

// returns norm_sum [N] (NDC units), vec_sum [N,2] (pixel units)
std::vector<torch::Tensor> egs_gradnorm(
    torch::Tensor means2d, torch::Tensor conics, torch::Tensor opacities, torch::Tensor colors,
    torch::Tensor tile_offsets, torch::Tensor flatten_ids, int64_t width, int64_t height,
    int64_t tile_size, torch::Tensor v_colors, torch::Tensor v_alphas) {
    check_common(means2d, conics, opacities, tile_offsets, flatten_ids);
    check_input(colors, "colors", torch::kFloat32);
    check_input(v_colors, "v_colors", torch::kFloat32);
    check_input(v_alphas, "v_alphas", torch::kFloat32);
    TORCH_CHECK(colors.size(-1) == 3 && v_colors.size(-1) == 3, "RGB colours expected");
    TORCH_CHECK(v_alphas.numel() == width * height, "v_alphas must be [H,W]");
    const int N = means2d.size(0);
    const int tile_h = tile_offsets.size(0), tile_w = tile_offsets.size(1);
    auto norm = torch::zeros({N}, means2d.options());
    auto vec = torch::zeros({N, 2}, means2d.options());
    const int n_isects = flatten_ids.numel();
    if (n_isects == 0) return {norm, vec};
    dim3 grid(tile_w, tile_h);
    dim3 block(tile_size, tile_size);
    TORCH_CHECK((tile_size * tile_size) % 32 == 0, "tile must hold whole warps");
    const size_t shmem = tile_size * tile_size * (sizeof(int) + 3 * sizeof(float3));
    auto stream = at::cuda::getCurrentCUDAStream();
    egs_gradnorm_kernel<<<grid, block, shmem, stream>>>(
        n_isects, width, height, tile_size, tile_w, tile_h,
        (const float2 *)means2d.data_ptr<float>(), (const float3 *)conics.data_ptr<float>(),
        opacities.data_ptr<float>(), (const float3 *)colors.data_ptr<float>(),
        tile_offsets.data_ptr<int>(), flatten_ids.data_ptr<int>(),
        (const float3 *)v_colors.data_ptr<float>(), v_alphas.data_ptr<float>(),
        0.5f * (float)width, 0.5f * (float)height, norm.data_ptr<float>(),
        (float2 *)vec.data_ptr<float>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {norm, vec};
}

// returns dominant [N] int32 (0/1), loss_weight [N]
std::vector<torch::Tensor> egs_stats(
    torch::Tensor means2d, torch::Tensor conics, torch::Tensor opacities,
    torch::Tensor tile_offsets, torch::Tensor flatten_ids, int64_t width, int64_t height,
    int64_t tile_size, int64_t top_k, bool weight_after_T, c10::optional<torch::Tensor> pix_loss) {
    check_common(means2d, conics, opacities, tile_offsets, flatten_ids);
    TORCH_CHECK(top_k >= 1 && top_k <= EGS_MAX_K, "top_k must be in [1, 10]");
    const float3 *pl = nullptr;
    if (pix_loss.has_value()) {
        check_input(*pix_loss, "pix_loss", torch::kFloat32);
        TORCH_CHECK(pix_loss->numel() == width * height * 3, "pix_loss must be [H,W,3]");
        pl = (const float3 *)pix_loss->data_ptr<float>();
    }
    const int N = means2d.size(0);
    const int tile_h = tile_offsets.size(0), tile_w = tile_offsets.size(1);
    auto dominant = torch::zeros({N}, means2d.options().dtype(torch::kInt32));
    auto lw = torch::zeros({N}, means2d.options());
    const int n_isects = flatten_ids.numel();
    if (n_isects == 0) return {dominant, lw};
    dim3 grid(tile_w, tile_h);
    dim3 block(tile_size, tile_size);
    const size_t shmem = tile_size * tile_size * (sizeof(int) + 2 * sizeof(float3));
    auto stream = at::cuda::getCurrentCUDAStream();
    egs_stats_kernel<<<grid, block, shmem, stream>>>(
        n_isects, width, height, tile_size, tile_w, tile_h,
        (const float2 *)means2d.data_ptr<float>(), (const float3 *)conics.data_ptr<float>(),
        opacities.data_ptr<float>(), tile_offsets.data_ptr<int>(), flatten_ids.data_ptr<int>(),
        (int)top_k, weight_after_T, pl, dominant.data_ptr<int>(), lw.data_ptr<float>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {dominant, lw};
}
