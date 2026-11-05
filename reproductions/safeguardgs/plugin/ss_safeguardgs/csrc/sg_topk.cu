/*
 * SafeguardGS pixel-level importance-rank pruning statistics on gsplat 1.5.3 tile lists.
 *
 * Adapted from the official SafeguardGS rasterizer
 *   https://github.com/ASU-ESIC-FAN-Lab/SafeguardGS (commit 539ca22e)
 *   submodules/diff-gaussian-rasterization/cuda_rasterizer/forward.cu
 *     compute_score()            -> sg_score() below (all bit-mask score ids)
 *     renderCUDA_topk_color()    -> sg_tile_keys_kernel() below (slot / overwrite semantics)
 *   submodules/diff-gaussian-rasterization/cuda_rasterizer/rasterizer_impl.cu
 *     forwardTopKColor() + markingTopKMasks() -> sorted per-tile top-k in kernels.py / core.py
 * which derives from the Inria 3DGS rasterizer (Copyright (C) 2023, Inria, GRAPHDECO;
 * non-commercial research licence, see LICENSE-3DGS.md). Tile traversal layout follows the
 * EfficientGS plugin kernel in this workspace (reproductions/efficientgs/plugin/ss_efficientgs/
 * csrc/egs_raster.cu, itself after reproductions/mini-splatting/plugin/.../ms_raster.cu).
 *
 * Numerics are gsplat's (alpha = min(0.999, o exp(-sigma)), skip sigma < 0 or alpha < 1/255,
 * stop when the next transmittance <= 1e-4, pixel centres at +0.5) so scores describe the image
 * gsplat renders.
 *
 * Official slot semantics (mode "tile"): thread t of a tile processes batch b of 256 Gaussians;
 * every contributing Gaussian of the batch overwrites the slot range.x + 256*b + t, so the slot
 * ends with the *last* contributing Gaussian of that batch for that pixel. The keys
 * (tile << 32) | floor((1 - score) * 65535) are then stably sorted and the first k per tile are
 * kept. Here every tile owns a padded slot range (pad_offsets[tile] + 256*b + t), which removes
 * the official cross-tile race of partial last batches.
 *
 * Mode "pixel" (paper text, ablation): per-pixel running top-k over all contributors.
 */

#include <torch/extension.h>
#include <cooperative_groups.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>

namespace cg = cooperative_groups;

#define SG_ALPHA_THRESHOLD (1.f / 255.f)
#define SG_MAX_K 16

// Official compute_score (forward.cu), channel count 3.
__device__ __forceinline__ float sg_score(
    const int func_id, const float opacity, const float alpha, const float T, const float dx,
    const float dy, const float p_coef, const float c_coef, const float3 gt, const float3 c) {
    float score = 0.f;
    const int lo = func_id & 0x0f;
    float pd = 0.f;
    if (lo == 0x05 || lo == 0x06 || lo == 0x07 || lo == 0x08 || lo == 0x09 || lo >= 0x0c)
        pd = expf(-1.0f * (p_coef * sqrtf(dx * dx + dy * dy)));
    switch (lo) {
        case 0x00: score = 1.f; break;
        case 0x01: score = opacity; break;
        case 0x02: score = alpha; break;
        case 0x03: score = opacity * T; break;
        case 0x04: score = alpha * T; break;
        case 0x05: score = pd; break;
        case 0x06: score = pd * opacity; break;
        case 0x07: score = pd * alpha; break;
        case 0x08: score = pd * opacity * T; break;
        case 0x09: score = pd * alpha * T; break;
        case 0x0a: score = opacity + T; break;
        case 0x0b: score = alpha + T; break;
        case 0x0c: score = pd + opacity; break;
        case 0x0d: score = pd + alpha; break;
        case 0x0e: score = pd + opacity + T; break;
        case 0x0f: score = pd + alpha + T; break;
    }
    const float l1 = fabsf(gt.x - c.x) + fabsf(gt.y - c.y) + fabsf(gt.z - c.z);
    switch (func_id & 0xf0) {
        case 0x10: {
            const float gl = sqrtf(gt.x * gt.x + gt.y * gt.y + gt.z * gt.z);
            const float cl = sqrtf(c.x * c.x + c.y * c.y + c.z * c.z);
            return score * (gt.x * c.x + gt.y * c.y + gt.z * c.z) / (gl * cl);
        }
        case 0x20: return score * (1.f - l1 / 3.f);
        case 0x30: return score * expf(-1.0f * c_coef * l1 / 3.f);
        case 0x40: return score + (1.f - l1 / 3.f);
        case 0x50: return score + expf(-1.0f * c_coef * l1 / 3.f);
        default: return score;
    }
}

template <bool TILE_MODE>
__global__ void sg_topk_kernel(
    const int n_isects, const int W, const int H, const int tile_size, const int tile_w,
    const int tile_h,
    const float2 *__restrict__ means2d, const float3 *__restrict__ conics,
    const float *__restrict__ opacities, const float3 *__restrict__ colors,
    const int *__restrict__ tile_offsets, const int *__restrict__ flatten_ids,
    const float3 *__restrict__ image_gt,  // [H,W,3] in [0,1]
    const int func_id, const float p_coef, const float c_coef, const int top_k,
    const long *__restrict__ pad_offsets,  // [n_tiles] (tile mode)
    int *__restrict__ slot_q,              // [n_slots] quantised 1-score, -1 = empty (tile mode)
    int *__restrict__ slot_id,             // [n_slots] Gaussian id (tile mode)
    int *__restrict__ keep                 // [N] (pixel mode)
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
    float3 *rgb_batch = reinterpret_cast<float3 *>(&conic_batch[block_size]);
    const int tr = block.thread_rank();

    float3 gt = {0.f, 0.f, 0.f};
    if (inside) gt = image_gt[i * W + j];

    float best_s[SG_MAX_K];
    int best_id[SG_MAX_K];
    if (!TILE_MODE) {
        for (int k = 0; k < SG_MAX_K; ++k) {
            best_s[k] = -INFINITY;
            best_id[k] = -1;
        }
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
            rgb_batch[tr] = colors[g];
        }
        block.sync();
        const int batch_size = min(block_size, range_end - batch_start);
        int last_q = -1, last_g = -1;
        for (int t = 0; (t < batch_size) && !done; ++t) {
            const float3 conic = conic_batch[t];
            const float3 xyo = xy_opacity_batch[t];
            const float dx = xyo.x - px, dy = xyo.y - py;
            const float sigma = 0.5f * (conic.x * dx * dx + conic.z * dy * dy) + conic.y * dx * dy;
            const float alpha = min(0.999f, xyo.z * __expf(-sigma));
            if (sigma < 0.f || alpha < SG_ALPHA_THRESHOLD) continue;
            const float next_T = T * (1.0f - alpha);
            if (next_T <= 1e-4f) {
                done = true;
                break;
            }
            // official: score uses T before this Gaussian (w_i = alpha_i T_i)
            const float sc =
                sg_score(func_id, xyo.z, alpha, T, dx, dy, p_coef, c_coef, gt, rgb_batch[t]);
            const int g = id_batch[t];
            if (TILE_MODE) {
                // official: scaled_score = __float2uint_rd((1 - score) * 65535), overwritten by
                // every later contributor of the same batch
                last_q = (int)min(__float2uint_rd((1.f - sc) * 65535.f), 0x7fffffffu);
                last_g = g;
            } else {
                for (int kk = 0; kk < top_k; ++kk) {
                    if (sc > best_s[kk]) {
                        for (int k3 = top_k - 1; k3 > kk; --k3) {
                            best_s[k3] = best_s[k3 - 1];
                            best_id[k3] = best_id[k3 - 1];
                        }
                        best_s[kk] = sc;
                        best_id[kk] = g;
                        break;
                    }
                }
            }
            T = next_T;
        }
        if (TILE_MODE && last_g >= 0) {
            const long slot = pad_offsets[tile_id] + (long)block_size * b + tr;
            slot_q[slot] = last_q;
            slot_id[slot] = last_g;
        }
    }
    if (!TILE_MODE && inside) {
        for (int k = 0; k < top_k; ++k) {
            if (best_id[k] < 0) break;
            keep[best_id[k]] = 1;
        }
    }
}

static void check_input(const torch::Tensor &t, const char *name, torch::ScalarType type) {
    TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
    TORCH_CHECK(t.scalar_type() == type, name, " has the wrong dtype");
}

// tile mode: returns {slot_q [n_slots], slot_id [n_slots]}; pixel mode: returns {keep [N]}
std::vector<torch::Tensor> sg_topk(
    torch::Tensor means2d, torch::Tensor conics, torch::Tensor opacities, torch::Tensor colors,
    torch::Tensor tile_offsets, torch::Tensor flatten_ids, int64_t width, int64_t height,
    int64_t tile_size, torch::Tensor image_gt, int64_t func_id, double p_coef, double c_coef,
    int64_t top_k, bool tile_mode, torch::Tensor pad_offsets, int64_t n_slots) {
    check_input(means2d, "means2d", torch::kFloat32);
    check_input(conics, "conics", torch::kFloat32);
    check_input(opacities, "opacities", torch::kFloat32);
    check_input(colors, "colors", torch::kFloat32);
    check_input(tile_offsets, "tile_offsets", torch::kInt32);
    check_input(flatten_ids, "flatten_ids", torch::kInt32);
    check_input(image_gt, "image_gt", torch::kFloat32);
    TORCH_CHECK(image_gt.numel() == width * height * 3, "image_gt must be [H,W,3]");
    TORCH_CHECK(colors.size(-1) == 3, "RGB colours expected");
    TORCH_CHECK(func_id >= 0 && func_id <= 0x5f, "score function id out of range");
    TORCH_CHECK(top_k >= 1 && (tile_mode || top_k <= SG_MAX_K), "top_k out of range");
    const int N = means2d.size(0);
    const int tile_h = tile_offsets.size(0), tile_w = tile_offsets.size(1);
    const int n_isects = flatten_ids.numel();
    dim3 grid(tile_w, tile_h);
    dim3 block(tile_size, tile_size);
    const size_t shmem = tile_size * tile_size * (sizeof(int) + 3 * sizeof(float3));
    auto stream = at::cuda::getCurrentCUDAStream();
    auto iopt = means2d.options().dtype(torch::kInt32);
    if (tile_mode) {
        check_input(pad_offsets, "pad_offsets", torch::kInt64);
        TORCH_CHECK(pad_offsets.numel() == tile_w * tile_h, "pad_offsets must be [n_tiles]");
        auto q = torch::full({n_slots}, -1, iopt);
        auto id = torch::full({n_slots}, -1, iopt);
        if (n_isects == 0) return {q, id};
        sg_topk_kernel<true><<<grid, block, shmem, stream>>>(
            n_isects, width, height, tile_size, tile_w, tile_h,
            (const float2 *)means2d.data_ptr<float>(), (const float3 *)conics.data_ptr<float>(),
            opacities.data_ptr<float>(), (const float3 *)colors.data_ptr<float>(),
            tile_offsets.data_ptr<int>(), flatten_ids.data_ptr<int>(),
            (const float3 *)image_gt.data_ptr<float>(), (int)func_id, (float)p_coef,
            (float)c_coef, (int)top_k, pad_offsets.data_ptr<long>(), q.data_ptr<int>(),
            id.data_ptr<int>(), nullptr);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        return {q, id};
    }
    auto keep = torch::zeros({N}, iopt);
    if (n_isects == 0) return {keep};
    sg_topk_kernel<false><<<grid, block, shmem, stream>>>(
        n_isects, width, height, tile_size, tile_w, tile_h,
        (const float2 *)means2d.data_ptr<float>(), (const float3 *)conics.data_ptr<float>(),
        opacities.data_ptr<float>(), (const float3 *)colors.data_ptr<float>(),
        tile_offsets.data_ptr<int>(), flatten_ids.data_ptr<int>(),
        (const float3 *)image_gt.data_ptr<float>(), (int)func_id, (float)p_coef, (float)c_coef,
        (int)top_k, nullptr, nullptr, nullptr, keep.data_ptr<int>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {keep};
}
