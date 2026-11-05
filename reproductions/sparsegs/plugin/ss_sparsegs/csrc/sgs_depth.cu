// SparseGS depth rasterizer on gsplat's projection / tile-intersection outputs.
//
// Forward follows the official diff-gaussian-rasterization-softmax renderCUDA
// (cuda_rasterizer/forward.cu:262-424, ForMyCat/SparseGS, Inria-derived): front-to-back,
// alpha = min(0.99, o * exp(power)), skip power > 0 and alpha < 1/255, stop when T*(1-alpha)
// < 1e-4; w = alpha*T;
//   D += d w exp(beta (w - 0.3)),  Z += w exp(beta (w - 0.3))      softmax depth log(D/Z + 1e-8)
//   AD += d w,  AZ += w                                             alpha depth AD/(AZ + 1e-8)
//   mode = d of argmax w (first maximum), start = first contributor.
// (Z := 1 and AZ := 1 when empty, as in the official code.)
// The official start/end indices are range.x + round*256 + thread_rank (the thread's own fetch
// index, a bug); both that and the true list index are written out.
//
// Backward (exact derivative of the forward, = supp. Sec. A / backward.cu:564-615):
//   dlogD/dd_k = w_k e_k / D
//   dout/dalpha_k = T_k (e'_k d_k - SN_k/(T_k(1-a_k)))/D - T_k (e'_k - SD_k/(T_k(1-a_k)))/Z
// with e_k = exp(beta(w_k-0.3)), e'_k = e_k (1 + beta w_k) and the suffix sums
// SN_k = sum_{j>k} w_j e'_j d_j, SD_k = sum_{j>k} w_j e'_j, evaluated front-to-back as
// (total - prefix) from totals stored by the forward. Pixel centres follow gsplat (j + 0.5).
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>

namespace {

constexpr int TILE = 16;
constexpr int BLOCK = TILE * TILE;

__device__ inline void tile_range(const int32_t* offsets, int tile_id, int n_tiles, int n_isects,
                                  int& lo, int& hi) {
    lo = offsets[tile_id];
    hi = (tile_id == n_tiles - 1) ? n_isects : offsets[tile_id + 1];
}

__global__ void depth_fwd_kernel(
    const float2* __restrict__ means2d, const float3* __restrict__ conics,
    const float* __restrict__ opac, const float* __restrict__ depths,
    const int32_t* __restrict__ offsets, const int32_t* __restrict__ flatten_ids,
    int W, int H, int tiles_x, int tiles_y, int n_isects, float beta,
    float* __restrict__ out_soft, float* __restrict__ out_alpha, float* __restrict__ out_mode,
    float* __restrict__ out_D, float* __restrict__ out_Z, float* __restrict__ out_SN,
    float* __restrict__ out_SD, int32_t* __restrict__ out_ids, int32_t* __restrict__ out_ids_off) {
    const int tx = threadIdx.x, ty = threadIdx.y;
    const int rank = ty * TILE + tx;
    const int px = blockIdx.x * TILE + tx, py = blockIdx.y * TILE + ty;
    const bool inside = px < W && py < H;
    const int tile_id = blockIdx.y * tiles_x + blockIdx.x;
    int lo, hi;
    tile_range(offsets, tile_id, tiles_x * tiles_y, n_isects, lo, hi);
    const float fx = px + 0.5f, fy = py + 0.5f;

    __shared__ int s_id[BLOCK];
    __shared__ float2 s_xy[BLOCK];
    __shared__ float3 s_con[BLOCK];
    __shared__ float s_op[BLOCK];
    __shared__ float s_d[BLOCK];

    float T = 1.f, D = 0.f, Z = 0.f, AD = 0.f, AZ = 0.f, SN = 0.f, SDn = 0.f, topW = 0.f, topD = 0.f;
    int start = -1, end = -1, start_off = -1, end_off = -1;
    bool done = !inside;
    const int rounds = (hi - lo + BLOCK - 1) / BLOCK;
    for (int r = 0; r < rounds; ++r) {
        if (__syncthreads_count(done) == BLOCK) break;
        const int k = lo + r * BLOCK + rank;
        if (k < hi) {
            const int g = flatten_ids[k];
            s_id[rank] = g;
            s_xy[rank] = means2d[g];
            s_con[rank] = conics[g];
            s_op[rank] = opac[g];
            s_d[rank] = depths[g];
        }
        __syncthreads();
        const int batch = min(BLOCK, hi - lo - r * BLOCK);
        for (int j = 0; !done && j < batch; ++j) {
            const float dx = s_xy[j].x - fx, dy = s_xy[j].y - fy;
            const float3 c = s_con[j];
            const float power = -0.5f * (c.x * dx * dx + c.z * dy * dy) - c.y * dx * dy;
            if (power > 0.f) continue;
            const float alpha = fminf(0.99f, s_op[j] * __expf(power));
            if (alpha < 1.f / 255.f) continue;
            const float nT = T * (1.f - alpha);
            if (nT < 1e-4f) { done = true; continue; }
            const float w = alpha * T, d = s_d[j];
            const float e = __expf(beta * (w - 0.3f));
            D += d * w * e;
            Z += w * e;
            AD += d * w;
            AZ += w;
            const float ep = e * (1.f + beta * w);
            SN += w * ep * d;
            SDn += w * ep;
            if (start == -1) { start = lo + r * BLOCK + j; start_off = lo + r * BLOCK + rank; }
            if (w > topW) { topW = w; topD = d; end = lo + r * BLOCK + j; end_off = lo + r * BLOCK + rank; }
            T = nT;
        }
        __syncthreads();
    }
    if (inside) {
        const int pix = py * W + px;
        const float Zs = (Z == 0.f) ? 1.f : Z;
        const float AZs = (AZ == 0.f) ? 1.f : AZ;
        out_soft[pix] = logf(D / Zs + 1e-8f);
        out_alpha[pix] = AD / (AZs + 1e-8f);
        out_mode[pix] = topD;
        out_D[pix] = D;
        out_Z[pix] = Zs;
        out_SN[pix] = SN;
        out_SD[pix] = SDn;
        out_ids[pix] = start;
        out_ids[H * W + pix] = end;
        out_ids_off[pix] = start_off;
        out_ids_off[H * W + pix] = end_off;
    }
}

__global__ void depth_bwd_kernel(
    const float2* __restrict__ means2d, const float3* __restrict__ conics,
    const float* __restrict__ opac, const float* __restrict__ depths,
    const int32_t* __restrict__ offsets, const int32_t* __restrict__ flatten_ids,
    int W, int H, int tiles_x, int tiles_y, int n_isects, float beta,
    const float* __restrict__ in_D, const float* __restrict__ in_Z,
    const float* __restrict__ in_SN, const float* __restrict__ in_SD,
    const float* __restrict__ g_soft,
    float2* __restrict__ g_means2d, float3* __restrict__ g_conics, float* __restrict__ g_opac,
    float* __restrict__ g_depths) {
    const int tx = threadIdx.x, ty = threadIdx.y;
    const int rank = ty * TILE + tx;
    const int px = blockIdx.x * TILE + tx, py = blockIdx.y * TILE + ty;
    const bool inside = px < W && py < H;
    const int tile_id = blockIdx.y * tiles_x + blockIdx.x;
    int lo, hi;
    tile_range(offsets, tile_id, tiles_x * tiles_y, n_isects, lo, hi);
    const float fx = px + 0.5f, fy = py + 0.5f;

    __shared__ int s_id[BLOCK];
    __shared__ float2 s_xy[BLOCK];
    __shared__ float3 s_con[BLOCK];
    __shared__ float s_op[BLOCK];
    __shared__ float s_d[BLOCK];

    float D = 1.f, Z = 1.f, SNt = 0.f, SDt = 0.f, gout = 0.f;
    if (inside) {
        const int pix = py * W + px;
        D = in_D[pix];
        Z = in_Z[pix];
        SNt = in_SN[pix];
        SDt = in_SD[pix];
        gout = g_soft[pix];
    }
    // Empty pixel or D == 0: the official forward clamps Z to 1 and returns log(1e-8); no
    // contributor -> no gradient.
    bool done = !inside || D == 0.f;
    float T = 1.f, pN = 0.f, pD = 0.f;
    const int rounds = (hi - lo + BLOCK - 1) / BLOCK;
    for (int r = 0; r < rounds; ++r) {
        if (__syncthreads_count(done) == BLOCK) break;
        const int k = lo + r * BLOCK + rank;
        if (k < hi) {
            const int g = flatten_ids[k];
            s_id[rank] = g;
            s_xy[rank] = means2d[g];
            s_con[rank] = conics[g];
            s_op[rank] = opac[g];
            s_d[rank] = depths[g];
        }
        __syncthreads();
        const int batch = min(BLOCK, hi - lo - r * BLOCK);
        for (int j = 0; !done && j < batch; ++j) {
            const float dx = s_xy[j].x - fx, dy = s_xy[j].y - fy;
            const float3 c = s_con[j];
            const float power = -0.5f * (c.x * dx * dx + c.z * dy * dy) - c.y * dx * dy;
            if (power > 0.f) continue;
            const float G = __expf(power);
            const float alpha = fminf(0.99f, s_op[j] * G);
            if (alpha < 1.f / 255.f) continue;
            const float nT = T * (1.f - alpha);
            if (nT < 1e-4f) { done = true; continue; }
            const float w = alpha * T, d = s_d[j];
            const float e = __expf(beta * (w - 0.3f));
            const float ep = e * (1.f + beta * w);
            pN += w * ep * d;
            pD += w * ep;
            const float sufN = SNt - pN, sufD = SDt - pD;  // sum over j > k
            const float inv1a = 1.f / (1.f - alpha);
            const float dalpha = gout * ((T * ep * d - sufN * inv1a) / D - (T * ep - sufD * inv1a) / Z);
            const float ddepth = gout * w * e / D;
            const int g = s_id[j];
            // official: no clamp handling (dL/dopacity = G dL/dalpha even when alpha = 0.99)
            const float dG = s_op[j] * dalpha;
            const float gdx = G * dx, gdy = G * dy;
            // power = -0.5 (a dx^2 + c dy^2) - b dx dy, dx = mean.x - px
            atomicAdd(&g_means2d[g].x, dG * (-gdx * c.x - gdy * c.y));
            atomicAdd(&g_means2d[g].y, dG * (-gdy * c.z - gdx * c.y));
            atomicAdd(&g_conics[g].x, -0.5f * gdx * dx * dG);
            atomicAdd(&g_conics[g].y, -gdx * dy * dG);
            atomicAdd(&g_conics[g].z, -0.5f * gdy * dy * dG);
            atomicAdd(&g_opac[g], G * dalpha);
            atomicAdd(&g_depths[g], ddepth);
            T = nT;
        }
        __syncthreads();
    }
}

// Official prune_floaters inner loop: for every flagged pixel, every flatten-list entry in
// [start, end] whose power at the pixel exceeds power_thresh (calc_alpha: power > 0 -> -100).
__global__ void prune_mark_kernel(
    const float2* __restrict__ means2d, const float3* __restrict__ conics,
    const int32_t* __restrict__ flatten_ids, int n_isects,
    const int32_t* __restrict__ px, const int32_t* __restrict__ py,
    const int32_t* __restrict__ lo, const int32_t* __restrict__ hi, int n_pix, float thresh,
    bool* __restrict__ mark) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_pix) return;
    const float fx = px[i] + 0.5f, fy = py[i] + 0.5f;
    const int a = max(lo[i], 0), b = min(hi[i], n_isects - 1);
    for (int k = a; k <= b; ++k) {
        const int g = flatten_ids[k];
        const float dx = means2d[g].x - fx, dy = means2d[g].y - fy;
        const float3 c = conics[g];
        float power = -0.5f * (c.x * dx * dx + c.z * dy * dy) - c.y * dx * dy;
        if (power > 0.f) power = -100.f;
        if (power > thresh) mark[g] = true;
    }
}

void check(const torch::Tensor& t, const char* name) {
    TORCH_CHECK(t.is_cuda() && t.is_contiguous(), name, " must be a contiguous CUDA tensor");
}

}  // namespace

std::vector<torch::Tensor> depth_forward(torch::Tensor means2d, torch::Tensor conics,
                                         torch::Tensor opac, torch::Tensor depths,
                                         torch::Tensor offsets, torch::Tensor flatten_ids,
                                         int64_t W, int64_t H, double beta) {
    check(means2d, "means2d"); check(conics, "conics"); check(opac, "opacities");
    check(depths, "depths"); check(offsets, "isect_offsets"); check(flatten_ids, "flatten_ids");
    const int tiles_x = (W + TILE - 1) / TILE, tiles_y = (H + TILE - 1) / TILE;
    TORCH_CHECK(offsets.numel() == tiles_x * tiles_y, "isect_offsets must be [tiles_y, tiles_x] for tile 16");
    auto f = means2d.options();
    auto soft = torch::empty({H, W}, f), alpha = torch::empty({H, W}, f), mode = torch::empty({H, W}, f);
    auto Dt = torch::empty({H, W}, f), Zt = torch::empty({H, W}, f);
    auto SN = torch::empty({H, W}, f), SD = torch::empty({H, W}, f);
    auto ids = torch::empty({2, H, W}, offsets.options());
    auto ids_off = torch::empty({2, H, W}, offsets.options());
    dim3 block(TILE, TILE), grid(tiles_x, tiles_y);
    depth_fwd_kernel<<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const float2*>(means2d.data_ptr<float>()),
        reinterpret_cast<const float3*>(conics.data_ptr<float>()), opac.data_ptr<float>(),
        depths.data_ptr<float>(), offsets.data_ptr<int32_t>(), flatten_ids.data_ptr<int32_t>(),
        W, H, tiles_x, tiles_y, flatten_ids.numel(), (float)beta, soft.data_ptr<float>(),
        alpha.data_ptr<float>(), mode.data_ptr<float>(), Dt.data_ptr<float>(), Zt.data_ptr<float>(),
        SN.data_ptr<float>(), SD.data_ptr<float>(), ids.data_ptr<int32_t>(), ids_off.data_ptr<int32_t>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {soft, alpha, mode, Dt, Zt, SN, SD, ids, ids_off};
}

std::vector<torch::Tensor> depth_backward(torch::Tensor means2d, torch::Tensor conics,
                                          torch::Tensor opac, torch::Tensor depths,
                                          torch::Tensor offsets, torch::Tensor flatten_ids,
                                          int64_t W, int64_t H, double beta, torch::Tensor Dt,
                                          torch::Tensor Zt, torch::Tensor SN, torch::Tensor SD,
                                          torch::Tensor g_soft) {
    check(g_soft, "grad");
    const int tiles_x = (W + TILE - 1) / TILE, tiles_y = (H + TILE - 1) / TILE;
    auto gm = torch::zeros_like(means2d), gc = torch::zeros_like(conics);
    auto go = torch::zeros_like(opac), gd = torch::zeros_like(depths);
    dim3 block(TILE, TILE), grid(tiles_x, tiles_y);
    depth_bwd_kernel<<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const float2*>(means2d.data_ptr<float>()),
        reinterpret_cast<const float3*>(conics.data_ptr<float>()), opac.data_ptr<float>(),
        depths.data_ptr<float>(), offsets.data_ptr<int32_t>(), flatten_ids.data_ptr<int32_t>(),
        W, H, tiles_x, tiles_y, flatten_ids.numel(), (float)beta, Dt.data_ptr<float>(),
        Zt.data_ptr<float>(), SN.data_ptr<float>(), SD.data_ptr<float>(), g_soft.data_ptr<float>(),
        reinterpret_cast<float2*>(gm.data_ptr<float>()), reinterpret_cast<float3*>(gc.data_ptr<float>()),
        go.data_ptr<float>(), gd.data_ptr<float>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {gm, gc, go, gd};
}

torch::Tensor prune_mark(torch::Tensor means2d, torch::Tensor conics, torch::Tensor flatten_ids,
                         torch::Tensor px, torch::Tensor py, torch::Tensor lo, torch::Tensor hi,
                         double thresh, int64_t n_gauss) {
    check(px, "px"); check(py, "py"); check(lo, "lo"); check(hi, "hi");
    auto mark = torch::zeros({n_gauss}, means2d.options().dtype(torch::kBool));
    const int n = px.numel();
    if (n > 0 && flatten_ids.numel() > 0) {
        prune_mark_kernel<<<(n + 255) / 256, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
            reinterpret_cast<const float2*>(means2d.data_ptr<float>()),
            reinterpret_cast<const float3*>(conics.data_ptr<float>()), flatten_ids.data_ptr<int32_t>(),
            flatten_ids.numel(), px.data_ptr<int32_t>(), py.data_ptr<int32_t>(), lo.data_ptr<int32_t>(),
            hi.data_ptr<int32_t>(), n, (float)thresh, mark.data_ptr<bool>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return mark;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("depth_forward", &depth_forward);
    m.def("depth_backward", &depth_backward);
    m.def("prune_mark", &prune_mark);
}
