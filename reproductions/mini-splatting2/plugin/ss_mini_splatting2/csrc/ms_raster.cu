// Copied unchanged from reproductions/mini-splatting/plugin/ss_mini_splatting/csrc/ms_raster.cu
// (SplatStudio Mini-Splatting port, plan #28) so the two plugins stay independent.
/*
 * Mini-Splatting per-Gaussian / per-pixel statistics on gsplat 1.5.3 tile lists.
 *
 * Adapted from the official Mini-Splatting rasterizer
 *   https://github.com/fatPeter/mini-splatting
 *   submodules/diff-gaussian-rasterization_ms/cuda_rasterizer/forward.cu
 *   (renderCUDA: accum_weights_p / accum_weights_count / accum_max_count, and
 *    render_depthCUDA: ray/ellipsoid mid-point of the max-weight Gaussian),
 * which is derived from the Inria 3DGS rasterizer (Copyright (C) 2023, Inria, GRAPHDECO;
 * free for non-commercial, research and evaluation use, see LICENSE-3DGS.md).
 *
 * The traversal and alpha numerics follow gsplat's rasterize_to_pixels_3dgs_fwd_kernel
 * (alpha = min(0.999, o * exp(-sigma)), skip sigma < 0 or alpha < 1/255, stop when the next
 * transmittance <= 1e-4, pixel centres at +0.5) so the statistics describe exactly the image
 * gsplat renders for the trained model. Forward only, no gradients.
 */

#include <torch/extension.h>
#include <cooperative_groups.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>

namespace cg = cooperative_groups;

#define MS_ALPHA_THRESHOLD (1.f / 255.f)

template <bool DEPTH, bool WEIGHTS>
__global__ void ms_raster_kernel(
    const int n_isects, const int W, const int H, const int tile_size, const int tile_w,
    const int tile_h,
    const float2 *__restrict__ means2d,     // [N,2]
    const float3 *__restrict__ conics,      // [N,3]
    const float *__restrict__ opacities,    // [N]
    const int *__restrict__ tile_offsets,   // [tile_h, tile_w]
    const int *__restrict__ flatten_ids,    // [n_isects]
    // depth mode
    const float3 *__restrict__ means3d,     // [N,3]
    const float *__restrict__ rot_wl,       // [N,9] row-major world->local rotation (R^T)
    const float3 *__restrict__ inv_scale,   // [N,3] 1 / (3 s)
    const float *__restrict__ c2w,          // [16] row-major camera-to-world
    const float fx, const float fy, const float cx, const float cy,
    // outputs
    float *__restrict__ accum_w,            // [N]   sum_j w_ij               (accum_weights_p)
    int *__restrict__ accum_cnt,            // [N]   #pixels contributed to    (accum_weights_count)
    int *__restrict__ max_cnt,              // [N]   #pixels where argmax      (accum_max_count)
    int *__restrict__ pix_argmax,           // [H,W] argmax Gaussian id or -1
    float *__restrict__ pix_T,              // [H,W] final transmittance
    float *__restrict__ pix_pts             // [H,W,3] mid-point of the argmax Gaussian (DEPTH)
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

    // camera ray through the pixel centre (official: inverse projection of the pixel centre)
    float ox = 0.f, oy = 0.f, oz = 0.f, dx = 0.f, dy = 0.f, dz = 0.f;
    if (DEPTH) {
        const float u = (px - cx) / fx, v = (py - cy) / fy;
        dx = c2w[0] * u + c2w[1] * v + c2w[2];
        dy = c2w[4] * u + c2w[5] * v + c2w[6];
        dz = c2w[8] * u + c2w[9] * v + c2w[10];
        const float inv_n = rsqrtf(dx * dx + dy * dy + dz * dz);
        dx *= inv_n; dy *= inv_n; dz *= inv_n;
        ox = c2w[3]; oy = c2w[7]; oz = c2w[11];
    }

    float T = 1.0f;
    float weight_max = 0.f;
    int idx_max = -1;
    float3 point_rec = {0.f, 0.f, 0.f};
    const int tr = block.thread_rank();

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
            const float3 xy_opac = xy_opacity_batch[t];
            const float ddx = xy_opac.x - px, ddy = xy_opac.y - py;
            const float sigma =
                0.5f * (conic.x * ddx * ddx + conic.z * ddy * ddy) + conic.y * ddx * ddy;
            const float alpha = min(0.999f, xy_opac.z * __expf(-sigma));
            if (sigma < 0.f || alpha < MS_ALPHA_THRESHOLD) continue;
            const float next_T = T * (1.0f - alpha);
            if (next_T <= 1e-4f) {
                done = true;
                break;
            }
            const int g = id_batch[t];
            const float vis = alpha * T;
            if (DEPTH) {
                // App. "Calculation of Gaussian Depth": ray in the frame of the ellipsoid
                // x^2/(3sx)^2 + ... = 1 ; t_mid = -b / 2a (official render_depthCUDA).
                const float *R = rot_wl + 9 * g;
                const float3 mu = means3d[g];
                const float3 is = inv_scale[g];
                const float tx = ox - mu.x, ty = oy - mu.y, tz = oz - mu.z;
                const float rox = (R[0] * tx + R[1] * ty + R[2] * tz) * is.x;
                const float roy = (R[3] * tx + R[4] * ty + R[5] * tz) * is.y;
                const float roz = (R[6] * tx + R[7] * ty + R[8] * tz) * is.z;
                const float rdx = (R[0] * dx + R[1] * dy + R[2] * dz) * is.x;
                const float rdy = (R[3] * dx + R[4] * dy + R[5] * dz) * is.y;
                const float rdz = (R[6] * dx + R[7] * dy + R[8] * dz) * is.z;
                const float a = rdx * rdx + rdy * rdy + rdz * rdz;
                const float bb = 2.f * (rdx * rox + rdy * roy + rdz * roz);
                const float tmid = -bb / (2.f * a);
                // official: `if(depth<0) continue;` before the transmittance update
                if (tmid < 0.f) continue;
                if (weight_max < vis) {
                    weight_max = vis;
                    idx_max = g;
                    point_rec = {ox + tmid * dx, oy + tmid * dy, oz + tmid * dz};
                }
            } else {
                if (weight_max < vis) {
                    weight_max = vis;
                    idx_max = g;
                }
                if (WEIGHTS) {
                    atomicAdd(&accum_w[g], vis);
                    atomicAdd(&accum_cnt[g], 1);
                }
            }
            T = next_T;
        }
    }

    if (inside) {
        const int pix_id = i * W + j;
        if (!DEPTH && idx_max >= 0) atomicAdd(&max_cnt[idx_max], 1);
        pix_argmax[pix_id] = idx_max;
        pix_T[pix_id] = T;
        if (DEPTH) {
            pix_pts[3 * pix_id + 0] = point_rec.x;
            pix_pts[3 * pix_id + 1] = point_rec.y;
            pix_pts[3 * pix_id + 2] = point_rec.z;
        }
    }
}

static void check_input(const torch::Tensor &t, const char *name, torch::ScalarType type) {
    TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
    TORCH_CHECK(t.scalar_type() == type, name, " has the wrong dtype");
}

// returns accum_w [N], accum_cnt [N], max_cnt [N], argmax [H,W], T [H,W]
std::vector<torch::Tensor> ms_stats(
    torch::Tensor means2d, torch::Tensor conics, torch::Tensor opacities,
    torch::Tensor tile_offsets, torch::Tensor flatten_ids, int64_t width, int64_t height,
    int64_t tile_size, bool weights) {
    check_input(means2d, "means2d", torch::kFloat32);
    check_input(conics, "conics", torch::kFloat32);
    check_input(opacities, "opacities", torch::kFloat32);
    check_input(tile_offsets, "tile_offsets", torch::kInt32);
    check_input(flatten_ids, "flatten_ids", torch::kInt32);
    const int N = means2d.size(0);
    const int tile_h = tile_offsets.size(0), tile_w = tile_offsets.size(1);
    auto fopt = means2d.options();
    auto iopt = means2d.options().dtype(torch::kInt32);
    auto accum_w = torch::zeros({N}, fopt);
    auto accum_cnt = torch::zeros({N}, iopt);
    auto max_cnt = torch::zeros({N}, iopt);
    auto argmax = torch::full({height, width}, -1, iopt);
    auto T = torch::ones({height, width}, fopt);
    const int n_isects = flatten_ids.numel();
    if (n_isects == 0) return {accum_w, accum_cnt, max_cnt, argmax, T};
    dim3 grid(tile_w, tile_h);
    dim3 block(tile_size, tile_size);
    const size_t shmem = tile_size * tile_size * (sizeof(int) + 2 * sizeof(float3));
    auto stream = at::cuda::getCurrentCUDAStream();
    auto launch = weights ? ms_raster_kernel<false, true> : ms_raster_kernel<false, false>;
    launch<<<grid, block, shmem, stream>>>(
        n_isects, width, height, tile_size, tile_w, tile_h,
        (const float2 *)means2d.data_ptr<float>(), (const float3 *)conics.data_ptr<float>(),
        opacities.data_ptr<float>(), tile_offsets.data_ptr<int>(), flatten_ids.data_ptr<int>(),
        nullptr, nullptr, nullptr, nullptr, 0.f, 0.f, 0.f, 0.f,
        accum_w.data_ptr<float>(), accum_cnt.data_ptr<int>(), max_cnt.data_ptr<int>(),
        argmax.data_ptr<int>(), T.data_ptr<float>(), nullptr);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {accum_w, accum_cnt, max_cnt, argmax, T};
}

// returns points [H,W,3], T [H,W], argmax [H,W]
std::vector<torch::Tensor> ms_depth(
    torch::Tensor means2d, torch::Tensor conics, torch::Tensor opacities,
    torch::Tensor tile_offsets, torch::Tensor flatten_ids, int64_t width, int64_t height,
    int64_t tile_size, torch::Tensor means3d, torch::Tensor rot_wl, torch::Tensor inv_scale,
    torch::Tensor c2w, double fx, double fy, double cx, double cy) {
    check_input(means2d, "means2d", torch::kFloat32);
    check_input(conics, "conics", torch::kFloat32);
    check_input(opacities, "opacities", torch::kFloat32);
    check_input(tile_offsets, "tile_offsets", torch::kInt32);
    check_input(flatten_ids, "flatten_ids", torch::kInt32);
    check_input(means3d, "means3d", torch::kFloat32);
    check_input(rot_wl, "rot_wl", torch::kFloat32);
    check_input(inv_scale, "inv_scale", torch::kFloat32);
    check_input(c2w, "c2w", torch::kFloat32);
    const int tile_h = tile_offsets.size(0), tile_w = tile_offsets.size(1);
    auto fopt = means2d.options();
    auto iopt = means2d.options().dtype(torch::kInt32);
    auto pts = torch::zeros({height, width, 3}, fopt);
    auto argmax = torch::full({height, width}, -1, iopt);
    auto T = torch::ones({height, width}, fopt);
    const int n_isects = flatten_ids.numel();
    if (n_isects == 0) return {pts, T, argmax};
    dim3 grid(tile_w, tile_h);
    dim3 block(tile_size, tile_size);
    const size_t shmem = tile_size * tile_size * (sizeof(int) + 2 * sizeof(float3));
    auto stream = at::cuda::getCurrentCUDAStream();
    ms_raster_kernel<true, false><<<grid, block, shmem, stream>>>(
        n_isects, width, height, tile_size, tile_w, tile_h,
        (const float2 *)means2d.data_ptr<float>(), (const float3 *)conics.data_ptr<float>(),
        opacities.data_ptr<float>(), tile_offsets.data_ptr<int>(), flatten_ids.data_ptr<int>(),
        (const float3 *)means3d.data_ptr<float>(), rot_wl.data_ptr<float>(),
        (const float3 *)inv_scale.data_ptr<float>(), c2w.data_ptr<float>(), (float)fx,
        (float)fy, (float)cx, (float)cy, nullptr, nullptr, nullptr, argmax.data_ptr<int>(),
        T.data_ptr<float>(), pts.data_ptr<float>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {pts, T, argmax};
}
