// Loss-aware density-control statistics (arXiv 2611.02044, Sec. 4 + App. A.4/A.5).
//
// Written for this plugin (no official code exists). It replays the per-pixel front-to-back
// traversal of gsplat 1.5.3 `rasterize_to_pixels_3dgs_fwd_kernel` (same pixel centres, alpha =
// min(0.999, o * exp(-sigma)), skip sigma < 0 or alpha < 1/255, stop when T_next <= 1e-4) over
// gsplat's own tile intersections, and accumulates per Gaussian for ONE camera:
//   out[:, 0:5]   g_z = sum_x dL/dalpha * d alpha / d z          (App. Eq. app_per_sample_screen_grad_hess)
//   out[:, 5:20]  H_z = sum_x dL/dalpha * d^2 alpha / d z^2      (upper triangle, row major)
//   out[:, 20]    g_dup = sum_{x,c} dL/dI_c * (1 - alpha) Delta_c        (Eq. clone_gh)
//   out[:, 21]    h_dup = sum_{x,c} omega_c * ((1 - alpha) Delta_c)^2
//   out[:, 22]    g_dec = sum_{x,c} dL/dI_c * Delta_c                    (Eq. prune_gh)
//   out[:, 23]    h_dec = sum_{x,c} omega_c * Delta_c^2
// with z = (u_x, u_y, q_xx, q_xy, q_yy), d = u - x, sigma = 1/2 d^T Q d and
// Delta_c(x) = T alpha (c_c - b_c), b = background-inclusive colour behind the Gaussian (Eq. 11).
// T_i * b_{i+1} = (C_total - sum_{j<=i} c_j alpha_j T_j) / (1 - alpha_i) (C_total includes the
// background term), so no division by T is needed.
// dL/dalpha = sum_c dL/dI_c T (c_c - b_c) = (dL/dI . Delta) / alpha, and the alpha derivative is
// zero where gsplat clamps alpha at 0.999 (as in gsplat's backward).
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

namespace cg = cooperative_groups;

constexpr int NSTAT = 24;

__global__ void surrogate_stats_kernel(
    const int N, const int n_isects,
    const float2* __restrict__ means2d, const float3* __restrict__ conics,
    const float* __restrict__ opacities, const float* __restrict__ colors,
    const int W, const int H, const int tile_size, const int tile_width, const int tile_height,
    const int* __restrict__ tile_offsets, const int* __restrict__ flatten_ids,
    const float* __restrict__ c_total, const float* __restrict__ dl_di,
    const float* __restrict__ omega, float* __restrict__ out) {
  auto block = cg::this_thread_block();
  auto warp = cg::tiled_partition<32>(block);
  const int tile_id = block.group_index().y * tile_width + block.group_index().z;
  const int i = block.group_index().y * tile_size + block.thread_index().y;
  const int j = block.group_index().z * tile_size + block.thread_index().x;
  const float px = (float)j + 0.5f, py = (float)i + 0.5f;
  const bool inside = (i < H && j < W);
  const int pix = inside ? i * W + j : 0;
  bool done = !inside;

  float Ct[3] = {0.f, 0.f, 0.f}, G[3] = {0.f, 0.f, 0.f}, Om[3] = {0.f, 0.f, 0.f};
  if (inside) {
    for (int c = 0; c < 3; ++c) {
      Ct[c] = c_total[pix * 3 + c];
      G[c] = dl_di[pix * 3 + c];
      Om[c] = omega[pix * 3 + c];
    }
  }
  const int range_start = tile_offsets[tile_id];
  const int range_end = (tile_id == tile_width * tile_height - 1) ? n_isects : tile_offsets[tile_id + 1];
  const int bs = block.size();
  const int num_batches = (range_end - range_start + bs - 1) / bs;

  extern __shared__ int s[];
  int* id_batch = s;
  float3* xyo = reinterpret_cast<float3*>(&id_batch[bs]);
  float3* con = reinterpret_cast<float3*>(&xyo[bs]);
  float3* col = reinterpret_cast<float3*>(&con[bs]);

  float T = 1.f;
  float acc[3] = {0.f, 0.f, 0.f};
  const int tr = block.thread_rank();
  for (int b = 0; b < num_batches; ++b) {
    if (__syncthreads_count(done) >= bs) break;
    const int start = range_start + bs * b;
    const int idx = start + tr;
    if (idx < range_end) {
      const int g = flatten_ids[idx];
      id_batch[tr] = g;
      const float2 m = means2d[g];
      xyo[tr] = make_float3(m.x, m.y, opacities[g]);
      con[tr] = conics[g];
      col[tr] = make_float3(colors[3 * g], colors[3 * g + 1], colors[3 * g + 2]);
    }
    block.sync();
    const int batch_size = min(bs, range_end - start);
    for (int t = 0; t < batch_size; ++t) {
      bool valid = !done;
      float alpha = 0.f, vis = 0.f, dx = 0.f, dy = 0.f, next_T = T;
      const float3 q = con[t];
      const float3 xo = xyo[t];
      if (valid) {
        dx = xo.x - px;
        dy = xo.y - py;
        const float sigma = 0.5f * (q.x * dx * dx + q.z * dy * dy) + q.y * dx * dy;
        vis = xo.z * __expf(-sigma);
        alpha = fminf(0.999f, vis);
        if (sigma < 0.f || alpha < 1.f / 255.f) valid = false;
      }
      if (valid) {
        next_T = T * (1.f - alpha);
        if (next_T <= 1e-4f) {  // gsplat forward: exclusive stop
          done = true;
          valid = false;
        }
      }
      float v[NSTAT];
#pragma unroll
      for (int k = 0; k < NSTAT; ++k) v[k] = 0.f;
      if (valid) {
        const float3 c3 = col[t];
        const float cc[3] = {c3.x, c3.y, c3.z};
        float dot = 0.f, wsq = 0.f;
#pragma unroll
        for (int c = 0; c < 3; ++c) {
          const float a_new = acc[c] + cc[c] * alpha * T;
          const float Tb = (Ct[c] - a_new) / (1.f - alpha);  // T_i * b_{i+1}
          const float delta = alpha * (T * cc[c] - Tb);       // Eq. 11
          dot += G[c] * delta;
          wsq += Om[c] * delta * delta;
          acc[c] = a_new;
        }
        const float om1 = 1.f - alpha;
        v[20] = om1 * dot;
        v[21] = om1 * om1 * wsq;
        v[22] = dot;
        v[23] = wsq;
        if (vis <= 0.999f) {
          const float w = dot / alpha;  // dL/dalpha
          // grad of sigma wrt z
          const float gs[5] = {q.x * dx + q.y * dy, q.y * dx + q.z * dy, 0.5f * dx * dx, dx * dy,
                               0.5f * dy * dy};
          // Hessian of sigma wrt z (symmetric, qq block zero)
          float hs[5][5] = {{q.x, q.y, dx, dy, 0.f},
                            {q.y, q.z, 0.f, dx, dy},
                            {dx, 0.f, 0.f, 0.f, 0.f},
                            {dy, dx, 0.f, 0.f, 0.f},
                            {0.f, dy, 0.f, 0.f, 0.f}};
          // alpha = o exp(-sigma): grad = -alpha gs, hess = alpha (gs gs^T - hs)
#pragma unroll
          for (int a = 0; a < 5; ++a) v[a] = -w * alpha * gs[a];
          int k = 5;
#pragma unroll
          for (int a = 0; a < 5; ++a)
#pragma unroll
            for (int c = a; c < 5; ++c) v[k++] = w * alpha * (gs[a] * gs[c] - hs[a][c]);
        }
        T = next_T;
      }
      if (warp.any(valid)) {
#pragma unroll
        for (int k = 0; k < NSTAT; ++k) v[k] = cg::reduce(warp, v[k], cg::plus<float>());
        if (warp.thread_rank() == 0) {
          float* o = out + (size_t)id_batch[t] * NSTAT;
#pragma unroll
          for (int k = 0; k < NSTAT; ++k)
            if (v[k] != 0.f) atomicAdd(o + k, v[k]);
        }
      }
    }
    block.sync();
  }
}

torch::Tensor surrogate_stats(torch::Tensor means2d, torch::Tensor conics, torch::Tensor opacities,
                              torch::Tensor colors, torch::Tensor tile_offsets,
                              torch::Tensor flatten_ids, torch::Tensor c_total, torch::Tensor dl_di,
                              torch::Tensor omega, int64_t W, int64_t H, int64_t tile_size) {
  TORCH_CHECK(means2d.is_cuda() && means2d.is_contiguous() && means2d.dtype() == torch::kFloat32);
  for (auto* t : {&conics, &opacities, &colors, &c_total, &dl_di, &omega})
    TORCH_CHECK(t->is_cuda() && t->is_contiguous() && t->dtype() == torch::kFloat32);
  TORCH_CHECK(tile_offsets.dtype() == torch::kInt32 && flatten_ids.dtype() == torch::kInt32);
  TORCH_CHECK(tile_offsets.is_contiguous() && flatten_ids.is_contiguous());
  const int N = means2d.size(0);
  const int tile_height = tile_offsets.size(0), tile_width = tile_offsets.size(1);
  auto out = torch::zeros({N, NSTAT}, means2d.options());
  const int n_isects = flatten_ids.size(0);
  if (N == 0 || n_isects == 0) return out;
  dim3 threads(tile_size, tile_size, 1);
  dim3 grid(1, tile_height, tile_width);
  const int bs = tile_size * tile_size;
  const size_t shmem = bs * (sizeof(int) + 3 * sizeof(float3));
  surrogate_stats_kernel<<<grid, threads, shmem, at::cuda::getCurrentCUDAStream()>>>(
      N, n_isects, reinterpret_cast<const float2*>(means2d.data_ptr<float>()),
      reinterpret_cast<const float3*>(conics.data_ptr<float>()), opacities.data_ptr<float>(),
      colors.data_ptr<float>(), (int)W, (int)H, (int)tile_size, tile_width, tile_height,
      tile_offsets.data_ptr<int>(), flatten_ids.data_ptr<int>(), c_total.data_ptr<float>(),
      dl_di.data_ptr<float>(), omega.data_ptr<float>(), out.data_ptr<float>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}


// ---------------------------------------------------------------------------------------------
// Per-view joint splitting matrix (App. Eq. app_joint_splitting_exact with Eq. app_Sps_impl):
//   S = J_z^T H_z J_z + sum_m g_zm d^2 z_m / d phi^2   (p-s block of the second term dropped)
// z(phi) re-implements gsplat 1.5.3 fully_fused_projection (classic, eps2d = 0.3, clamped
// perspective Jacobian) and is differentiated with hyper-dual numbers (exact 1st/2nd
// derivatives), in `real` precision. One thread per visible Gaussian; S is added to S_acc.
// Working precision of the joint-split kernel (float: verified against the float64 torch
// reference in the lad-joint-split-cuda probe).
using real = float;
struct HD {
  real v, a, b, ab;
};
__device__ inline HD hd(real v) { return {v, 0., 0., 0.}; }
__device__ inline HD operator+(HD x, HD y) { return {x.v + y.v, x.a + y.a, x.b + y.b, x.ab + y.ab}; }
__device__ inline HD operator-(HD x, HD y) { return {x.v - y.v, x.a - y.a, x.b - y.b, x.ab - y.ab}; }
__device__ inline HD operator-(HD x) { return {-x.v, -x.a, -x.b, -x.ab}; }
__device__ inline HD operator*(HD x, HD y) {
  return {x.v * y.v, x.a * y.v + x.v * y.a, x.b * y.v + x.v * y.b,
          x.ab * y.v + x.a * y.b + x.b * y.a + x.v * y.ab};
}
__device__ inline HD operator*(real c, HD x) { return {c * x.v, c * x.a, c * x.b, c * x.ab}; }
__device__ inline HD inv(HD x) {
  const real i = 1.f / x.v, i2 = i * i;
  return {i, -x.a * i2, -x.b * i2, -x.ab * i2 + 2.0 * x.a * x.b * i2 * i};
}
__device__ inline HD operator/(HD x, HD y) { return x * inv(y); }
__device__ inline HD hexp(HD x) {
  const real e = expf(x.v);
  return {e, e * x.a, e * x.b, e * (x.ab + x.a * x.b)};
}
__device__ inline HD hclamp(HD x, real lo, real hi) {
  if (x.v < lo) return hd(lo);
  if (x.v > hi) return hd(hi);
  return x;
}

__device__ void screen_z(const HD* phi, const real* Rq, const real* Rw, const real* tw,
                         real fx, real fy, real cx, real cy, real W, real H, HD* z) {
  HD pc[3];
  for (int r = 0; r < 3; ++r)
    pc[r] = Rw[3 * r] * phi[0] + Rw[3 * r + 1] * phi[1] + Rw[3 * r + 2] * phi[2] + hd(tw[r]);
  const HD tx = pc[0], ty = pc[1], tz = pc[2];
  const real tanx = 0.5 * W / fx, tany = 0.5 * H / fy;
  const real lxp = (W - cx) / fx + 0.3 * tanx, lxn = cx / fx + 0.3 * tanx;
  const real lyp = (H - cy) / fy + 0.3 * tany, lyn = cy / fy + 0.3 * tany;
  const HD rz = inv(tz);
  const HD x = tz * hclamp(tx * rz, -lxn, lxp);
  const HD y = tz * hclamp(ty * rz, -lyn, lyp);
  const HD rz2 = rz * rz;
  HD J[2][3] = {{fx * rz, hd(0.), -(fx * (x * rz2))}, {hd(0.), fy * rz, -(fy * (y * rz2))}};
  HD s2[3];
  for (int k = 0; k < 3; ++k) s2[k] = hexp(2.0 * phi[3 + k]);
  // Sigma = Rq diag(s^2) Rq^T ; cov_c = Rw Sigma Rw^T ; A = Rw Rq (constant)
  real A[9];
  for (int r = 0; r < 3; ++r)
    for (int c = 0; c < 3; ++c)
      A[3 * r + c] = Rw[3 * r] * Rq[c] + Rw[3 * r + 1] * Rq[3 + c] + Rw[3 * r + 2] * Rq[6 + c];
  HD cc[3][3];
  for (int r = 0; r < 3; ++r)
    for (int c = r; c < 3; ++c) {
      HD acc = hd(0.);
      for (int k = 0; k < 3; ++k) acc = acc + (A[3 * r + k] * A[3 * c + k]) * s2[k];
      cc[r][c] = acc;
      cc[c][r] = acc;
    }
  HD JC[2][3];
  for (int r = 0; r < 2; ++r)
    for (int c = 0; c < 3; ++c) JC[r][c] = J[r][0] * cc[0][c] + J[r][1] * cc[1][c] + J[r][2] * cc[2][c];
  HD a = JC[0][0] * J[0][0] + JC[0][1] * J[0][1] + JC[0][2] * J[0][2] + hd(0.3);
  HD b = JC[0][0] * J[1][0] + JC[0][1] * J[1][1] + JC[0][2] * J[1][2];
  HD d = JC[1][0] * J[1][0] + JC[1][1] * J[1][1] + JC[1][2] * J[1][2] + hd(0.3);
  const HD rdet = inv(a * d - b * b);
  z[0] = fx * (tx * rz) + hd(cx);
  z[1] = fy * (ty * rz) + hd(cy);
  z[2] = d * rdet;
  z[3] = -(b * rdet);
  z[4] = a * rdet;
}

__global__ void joint_split_kernel(const int M, const long* __restrict__ ids,
                                   const float* __restrict__ means, const float* __restrict__ logs,
                                   const float* __restrict__ quats, const float* __restrict__ stats,
                                   const float* __restrict__ view, const float* __restrict__ Kmat,
                                   const int W, const int H, const bool drop_mixed,
                                   float* __restrict__ S_acc) {
  const int t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= M) return;
  const long g = ids[t];
  real Rw[9], tw[3];
  for (int r = 0; r < 3; ++r) {
    for (int c = 0; c < 3; ++c) Rw[3 * r + c] = view[4 * r + c];
    tw[r] = view[4 * r + 3];
  }
  const real fx = Kmat[0], fy = Kmat[4], cx = Kmat[2], cy = Kmat[5];
  real qw = quats[4 * g], qx = quats[4 * g + 1], qy = quats[4 * g + 2], qz = quats[4 * g + 3];
  const real qn = rsqrtf(qw * qw + qx * qx + qy * qy + qz * qz);
  qw *= qn; qx *= qn; qy *= qn; qz *= qn;
  const real Rq[9] = {1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qw * qz), 2 * (qx * qz + qw * qy),
                        2 * (qx * qy + qw * qz), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qw * qx),
                        2 * (qx * qz - qw * qy), 2 * (qy * qz + qw * qx), 1 - 2 * (qx * qx + qy * qy)};
  real phi0[6];
  for (int k = 0; k < 3; ++k) {
    phi0[k] = means[3 * g + k];
    phi0[3 + k] = logs[3 * g + k];
  }
  const float* st = stats + (size_t)g * NSTAT;
  real gz[5], Hz[5][5];
  for (int m = 0; m < 5; ++m) gz[m] = st[m];
  int k = 5;
  for (int r = 0; r < 5; ++r)
    for (int c = r; c < 5; ++c) {
      Hz[r][c] = st[k];
      Hz[c][r] = st[k];
      ++k;
    }
  real Jz[5][6], Hg[6][6];
  for (int r = 0; r < 6; ++r)
    for (int c = 0; c < 6; ++c) Hg[r][c] = 0.;
  HD phi[6], z[5];
  for (int i = 0; i < 6; ++i)
    for (int j = i; j < 6; ++j) {
      const bool mixed = (i < 3) != (j < 3);
      if (mixed && drop_mixed) continue;  // Eq. app_Sps_impl
      for (int c = 0; c < 6; ++c) phi[c] = hd(phi0[c]);
      phi[i].a = 1.;
      phi[j].b = 1.;
      screen_z(phi, Rq, Rw, tw, fx, fy, cx, cy, W, H, z);
      if (i == j)
        for (int m = 0; m < 5; ++m) Jz[m][i] = z[m].a;
      real h = 0.;
      for (int m = 0; m < 5; ++m) h += gz[m] * z[m].ab;
      Hg[i][j] = h;
      Hg[j][i] = h;
    }
  // S = Jz^T Hz Jz + Hg
  real HJ[5][6];
  for (int r = 0; r < 5; ++r)
    for (int c = 0; c < 6; ++c) {
      real acc = 0.;
      for (int m = 0; m < 5; ++m) acc += Hz[r][m] * Jz[m][c];
      HJ[r][c] = acc;
    }
  float* out = S_acc + (size_t)g * 36;
  for (int r = 0; r < 6; ++r)
    for (int c = 0; c < 6; ++c) {
      real acc = Hg[r][c];
      for (int m = 0; m < 5; ++m) acc += Jz[m][r] * HJ[m][c];
      out[6 * r + c] += (float)acc;
    }
}

void joint_split_accumulate(torch::Tensor ids, torch::Tensor means, torch::Tensor logs,
                            torch::Tensor quats, torch::Tensor stats, torch::Tensor viewmat,
                            torch::Tensor K, int64_t W, int64_t H, bool drop_mixed,
                            torch::Tensor S_acc) {
  TORCH_CHECK(ids.dtype() == torch::kInt64 && ids.is_contiguous() && ids.is_cuda());
  for (auto* t : {&means, &logs, &quats, &stats, &viewmat, &K, &S_acc})
    TORCH_CHECK(t->is_cuda() && t->is_contiguous() && t->dtype() == torch::kFloat32);
  TORCH_CHECK(S_acc.size(0) == means.size(0) && S_acc.numel() == means.size(0) * 36);
  const int M = ids.size(0);
  if (M == 0) return;
  const int threads = 128;
  joint_split_kernel<<<(M + threads - 1) / threads, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
      M, ids.data_ptr<long>(), means.data_ptr<float>(), logs.data_ptr<float>(),
      quats.data_ptr<float>(), stats.data_ptr<float>(), viewmat.data_ptr<float>(),
      K.data_ptr<float>(), (int)W, (int)H, drop_mixed, S_acc.data_ptr<float>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("surrogate_stats", &surrogate_stats);
  m.def("joint_split_accumulate", &joint_split_accumulate);
}
