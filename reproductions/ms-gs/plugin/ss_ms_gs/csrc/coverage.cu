// MS-GS pixel coverage + selective-rendering mask, one thread per Gaussian.
// Numerics follow MS-GS-rasterizer @ e7a2b4f cuda_rasterizer/forward.cu (computeCov2D,
// in_frustum z <= 0.2, conic without the 0.3 low-pass, level set 1/255, culling rule) and
// mirror ss_ms_gs/coverage.py (the CPU reference used by the probes).
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void coverage_kernel(int N, const float* __restrict__ means, const float* __restrict__ quats,
    const float* __restrict__ log_scales, const float* __restrict__ opac_logit,
    const float* __restrict__ max_px, const float* __restrict__ min_px, const float* __restrict__ V,
    float fx, float fy, float tanx, float tany, float opacity_level, float near,
    bool filter_small, bool filter_large, float min_rel, float max_rel, float abs_thr,
    float* __restrict__ px_out, bool* __restrict__ keep_out) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= N) return;
  float mx = means[3 * i], my = means[3 * i + 1], mz = means[3 * i + 2];
  float tx = V[0] * mx + V[1] * my + V[2] * mz + V[3];
  float ty = V[4] * mx + V[5] * my + V[6] * mz + V[7];
  float tz = V[8] * mx + V[9] * my + V[10] * mz + V[11];
  float s = 0.f;
  if (tz > near) {
    float limx = 1.3f * tanx, limy = 1.3f * tany;
    float cx = fminf(limx, fmaxf(-limx, tx / tz)) * tz;
    float cy = fminf(limy, fmaxf(-limy, ty / tz)) * tz;
    float qw = quats[4 * i], qx = quats[4 * i + 1], qy = quats[4 * i + 2], qz = quats[4 * i + 3];
    float qn = rsqrtf(qw * qw + qx * qx + qy * qy + qz * qz);
    qw *= qn; qx *= qn; qy *= qn; qz *= qn;
    float R[9] = {1.f - 2.f * (qy * qy + qz * qz), 2.f * (qx * qy - qw * qz), 2.f * (qx * qz + qw * qy),
                  2.f * (qx * qy + qw * qz), 1.f - 2.f * (qx * qx + qz * qz), 2.f * (qy * qz - qw * qx),
                  2.f * (qx * qz - qw * qy), 2.f * (qy * qz + qw * qx), 1.f - 2.f * (qx * qx + qy * qy)};
    float sc[3] = {expf(log_scales[3 * i]), expf(log_scales[3 * i + 1]), expf(log_scales[3 * i + 2])};
    // T = J W (2x3); A = T R diag(s) (2x3); cov = A A^T
    float j00 = fx / tz, j02 = -fx * cx / (tz * tz), j11 = fy / tz, j12 = -fy * cy / (tz * tz);
    float T0[3], T1[3];
    for (int c = 0; c < 3; c++) {
      T0[c] = j00 * V[c] + j02 * V[8 + c];
      T1[c] = j11 * V[4 + c] + j12 * V[8 + c];
    }
    float a = 0.f, b = 0.f, cc = 0.f;
    for (int k = 0; k < 3; k++) {
      float A0 = (T0[0] * R[k] + T0[1] * R[3 + k] + T0[2] * R[6 + k]) * sc[k];
      float A1 = (T1[0] * R[k] + T1[1] * R[3 + k] + T1[2] * R[6 + k]) * sc[k];
      a += A0 * A0; b += A0 * A1; cc += A1 * A1;
    }
    float det = a * cc - b * b;
    if (det > 0.f && a > 0.f && cc > 0.f) {
      float sig = 1.f / (1.f + expf(-opac_logit[i]));
      float ls = fmaxf(0.f, -2.f * logf(opacity_level / sig));
      s = fminf(sqrtf(ls * det / cc), sqrtf(ls * det / a));
    }
  }
  px_out[i] = s;
  float rmin = min_px[i] > 0.f ? s / min_px[i] : 1.f;
  float rmax = max_px[i] > 0.f ? s / max_px[i] : 1.f;
  bool keep = true;
  if (filter_small) {
    if (rmin < min_rel && s < abs_thr) keep = false;
    if (filter_large && rmax > max_rel) keep = false;
  }
  keep_out[i] = keep;
}

std::vector<torch::Tensor> coverage_select(torch::Tensor means, torch::Tensor quats, torch::Tensor log_scales,
    torch::Tensor opac_logit, torch::Tensor max_px, torch::Tensor min_px, torch::Tensor viewmat,
    double fx, double fy, double tanx, double tany, double opacity_level, double near,
    bool filter_small, bool filter_large, double min_rel, double max_rel, double abs_thr) {
  int N = means.size(0);
  auto px = torch::empty({N}, means.options());
  auto keep = torch::empty({N}, means.options().dtype(torch::kBool));
  auto V = viewmat.contiguous();
  if (N > 0) {
    int threads = 256, blocks = (N + threads - 1) / threads;
    coverage_kernel<<<blocks, threads>>>(N, means.data_ptr<float>(), quats.data_ptr<float>(),
        log_scales.data_ptr<float>(), opac_logit.data_ptr<float>(), max_px.data_ptr<float>(),
        min_px.data_ptr<float>(), V.data_ptr<float>(), (float)fx, (float)fy, (float)tanx, (float)tany,
        (float)opacity_level, (float)near, filter_small, filter_large, (float)min_rel, (float)max_rel,
        (float)abs_thr, px.data_ptr<float>(), keep.data_ptr<bool>());
  }
  return {px, keep};
}
