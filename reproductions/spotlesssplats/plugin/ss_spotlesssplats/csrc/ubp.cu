// Utilization statistic of SpotLessSplats (arXiv 2406.20055, Eq. 10) as implemented in the
// official gsplat 1.1.1 fork (github.com/lilygoli/SpotLessSplats,
// gsplat/cuda/csrc/rasterize_to_pixels_bwd.cu, `v_means2d_sqr`): for every Gaussian g,
//     sqr[g] = sum over pixels p of (dL_p/dx_g)^2 + (dL_p/dy_g)^2,
// where dL_p/d(x,y)_g is the per-pixel contribution of pixel p to the screen-space mean gradient
// (squared BEFORE the sum over pixels), and dL/dI_p = v_render[p].
//
// Derived from gsplat 1.5.3 RasterizeToPixels3DGSBwd.cu / RasterizeToPixels3DGSFwd.cu
// (Apache-2.0, nerfstudio-project/gsplat): identical tiling, alpha clamp (0.999), alpha
// threshold (1/255), transmittance cut-off (1e-4) and the `opac * vis <= 0.999` gradient gate.
// Only the means2d branch is kept. last_ids are recomputed by a front-to-back pass (gsplat does
// not expose them). Single camera, non-packed, no background (SpotLessSplats renders on black),
// zero alpha gradient (the loss depends on RGB only).

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>

#define ALPHA_THRESHOLD (1.f / 255.f)

__global__ void ubp_sqrgrad_kernel(
    const uint32_t N, const uint32_t n_isects,
    const float2 *__restrict__ means2d,    // [N, 2]
    const float3 *__restrict__ conics,     // [N, 3]
    const float *__restrict__ colors,      // [N, 3]
    const float *__restrict__ opacities,   // [N]
    const uint32_t image_width, const uint32_t image_height, const uint32_t tile_size,
    const uint32_t tile_width, const uint32_t tile_height,
    const int32_t *__restrict__ tile_offsets,  // [tile_height, tile_width]
    const int32_t *__restrict__ flatten_ids,   // [n_isects]
    const float *__restrict__ v_render,        // [H, W, 3]
    float *__restrict__ sqr                    // [N]
) {
    const uint32_t tile_id = blockIdx.y * tile_width + blockIdx.z;
    const uint32_t i = blockIdx.y * tile_size + threadIdx.y;
    const uint32_t j = blockIdx.z * tile_size + threadIdx.x;
    const float px = (float)j + 0.5f;
    const float py = (float)i + 0.5f;
    const bool inside = (i < image_height && j < image_width);
    const int32_t pix_id = min(i * image_width + j, image_width * image_height - 1);

    const int32_t range_start = tile_offsets[tile_id];
    const int32_t range_end = (tile_id == tile_width * tile_height - 1)
                                  ? (int32_t)n_isects : tile_offsets[tile_id + 1];
    const uint32_t block_size = blockDim.x * blockDim.y;
    const uint32_t tr = threadIdx.y * blockDim.x + threadIdx.x;
    const int32_t num_batches = (range_end - range_start + block_size - 1) / block_size;

    extern __shared__ int s[];
    int32_t *id_batch = (int32_t *)s;
    float3 *xy_opacity_batch = reinterpret_cast<float3 *>(&id_batch[block_size]);
    float3 *conic_batch = reinterpret_cast<float3 *>(&xy_opacity_batch[block_size]);
    float *rgbs_batch = (float *)&conic_batch[block_size];

    // ---------------- forward pass: transmittance after the last Gaussian and last index
    float T = 1.0f;
    int32_t cur_idx = 0;
    bool done = !inside;
    for (int32_t b = 0; b < num_batches; ++b) {
        if (__syncthreads_count(done) >= (int)block_size) break;
        const int32_t batch_start = range_start + block_size * b;
        const int32_t idx = batch_start + tr;
        if (idx < range_end) {
            const int32_t g = flatten_ids[idx];
            id_batch[tr] = g;
            const float2 xy = means2d[g];
            xy_opacity_batch[tr] = make_float3(xy.x, xy.y, opacities[g]);
            conic_batch[tr] = conics[g];
        }
        __syncthreads();
        const int32_t batch_size = min((int32_t)block_size, range_end - batch_start);
        for (int32_t t = 0; (t < batch_size) && !done; ++t) {
            const float3 conic = conic_batch[t];
            const float3 xy_opac = xy_opacity_batch[t];
            const float opac = xy_opac.z;
            const float2 delta = make_float2(xy_opac.x - px, xy_opac.y - py);
            const float sigma = 0.5f * (conic.x * delta.x * delta.x + conic.z * delta.y * delta.y) +
                                conic.y * delta.x * delta.y;
            const float alpha = min(0.999f, opac * __expf(-sigma));
            if (sigma < 0.f || alpha < ALPHA_THRESHOLD) continue;
            const float next_T = T * (1.0f - alpha);
            if (next_T <= 1e-4f) { done = true; break; }
            cur_idx = batch_start + t;
            T = next_T;
        }
    }
    __syncthreads();

    // ---------------- backward pass (means2d branch of gsplat's bwd kernel), squared per pixel
    const float T_final = T;
    T = T_final;
    float buffer[3] = {0.f, 0.f, 0.f};
    const int32_t bin_final = inside ? cur_idx : 0;
    float v_render_c[3];
    for (int k = 0; k < 3; ++k) v_render_c[k] = v_render[pix_id * 3 + k];

    // warp-wide maximum of bin_final
    int32_t warp_bin_final = bin_final;
    for (int offset = 16; offset > 0; offset /= 2)
        warp_bin_final = max(warp_bin_final, __shfl_xor_sync(0xffffffff, warp_bin_final, offset));
    const uint32_t lane = tr % 32;

    for (int32_t b = 0; b < num_batches; ++b) {
        __syncthreads();
        const int32_t batch_end = range_end - 1 - block_size * b;
        const int32_t batch_size = min((int32_t)block_size, batch_end + 1 - range_start);
        const int32_t idx = batch_end - (int32_t)tr;
        if (idx >= range_start) {
            const int32_t g = flatten_ids[idx];
            id_batch[tr] = g;
            const float2 xy = means2d[g];
            xy_opacity_batch[tr] = make_float3(xy.x, xy.y, opacities[g]);
            conic_batch[tr] = conics[g];
            for (int k = 0; k < 3; ++k) rgbs_batch[tr * 3 + k] = colors[g * 3 + k];
        }
        __syncthreads();
        for (int32_t t = max(0, batch_end - warp_bin_final); t < batch_size; ++t) {
            bool valid = inside;
            if (batch_end - t > bin_final) valid = false;
            float alpha = 0.f, opac = 0.f, vis = 0.f;
            float2 delta = make_float2(0.f, 0.f);
            float3 conic = make_float3(0.f, 0.f, 0.f);
            if (valid) {
                conic = conic_batch[t];
                const float3 xy_opac = xy_opacity_batch[t];
                opac = xy_opac.z;
                delta = make_float2(xy_opac.x - px, xy_opac.y - py);
                const float sigma = 0.5f * (conic.x * delta.x * delta.x +
                                            conic.z * delta.y * delta.y) +
                                    conic.y * delta.x * delta.y;
                vis = __expf(-sigma);
                alpha = min(0.999f, opac * vis);
                if (sigma < 0.f || alpha < ALPHA_THRESHOLD) valid = false;
            }
            if (!__any_sync(0xffffffff, valid)) continue;
            float local = 0.f;
            if (valid) {
                const float ra = 1.0f / (1.0f - alpha);
                T *= ra;
                const float fac = alpha * T;
                float v_alpha = 0.f;
                for (int k = 0; k < 3; ++k)
                    v_alpha += (rgbs_batch[t * 3 + k] * T - buffer[k] * ra) * v_render_c[k];
                if (opac * vis <= 0.999f) {
                    const float v_sigma = -opac * vis * v_alpha;
                    const float vx = v_sigma * (conic.x * delta.x + conic.y * delta.y);
                    const float vy = v_sigma * (conic.y * delta.x + conic.z * delta.y);
                    local = vx * vx + vy * vy;
                }
                for (int k = 0; k < 3; ++k) buffer[k] += rgbs_batch[t * 3 + k] * fac;
            }
            for (int offset = 16; offset > 0; offset /= 2)
                local += __shfl_down_sync(0xffffffff, local, offset);
            if (lane == 0 && local != 0.f) atomicAdd(sqr + id_batch[t], local);
        }
    }
}

torch::Tensor ubp_sqrgrad(torch::Tensor means2d, torch::Tensor conics, torch::Tensor colors,
                          torch::Tensor opacities, torch::Tensor tile_offsets,
                          torch::Tensor flatten_ids, torch::Tensor v_render, int64_t tile_size) {
    TORCH_CHECK(means2d.is_cuda() && means2d.dim() == 2 && means2d.size(1) == 2, "means2d [N,2] cuda");
    TORCH_CHECK(conics.dim() == 2 && conics.size(1) == 3, "conics [N,3]");
    TORCH_CHECK(colors.dim() == 2 && colors.size(1) == 3, "colors [N,3]");
    TORCH_CHECK(v_render.dim() == 3 && v_render.size(2) == 3, "v_render [H,W,3]");
    TORCH_CHECK(tile_offsets.dim() == 2, "tile_offsets [th,tw]");
    for (auto *t : {&means2d, &conics, &colors, &opacities, &v_render})
        TORCH_CHECK(t->scalar_type() == torch::kFloat32 && t->is_contiguous(), "float32 contiguous");
    TORCH_CHECK(tile_offsets.scalar_type() == torch::kInt32 && flatten_ids.scalar_type() == torch::kInt32,
                "int32 offsets/ids");
    const uint32_t N = means2d.size(0);
    const uint32_t H = v_render.size(0), W = v_render.size(1);
    const uint32_t th = tile_offsets.size(0), tw = tile_offsets.size(1);
    auto sqr = torch::zeros({(int64_t)N}, means2d.options());
    const uint32_t n_isects = flatten_ids.size(0);
    if (n_isects == 0 || N == 0) return sqr;
    dim3 threads(tile_size, tile_size, 1);
    dim3 grid(1, th, tw);
    const size_t shmem = tile_size * tile_size * (sizeof(int32_t) + 2 * sizeof(float3) + 3 * sizeof(float));
    ubp_sqrgrad_kernel<<<grid, threads, shmem, at::cuda::getCurrentCUDAStream()>>>(
        N, n_isects, reinterpret_cast<const float2 *>(means2d.data_ptr<float>()),
        reinterpret_cast<const float3 *>(conics.data_ptr<float>()), colors.data_ptr<float>(),
        opacities.data_ptr<float>(), W, H, tile_size, tw, th, tile_offsets.data_ptr<int32_t>(),
        flatten_ids.data_ptr<int32_t>(), v_render.data_ptr<float>(), sqr.data_ptr<float>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return sqr;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("ubp_sqrgrad", &ubp_sqrgrad, "SpotLessSplats utilization (sum of squared per-pixel means2d grads)");
}
