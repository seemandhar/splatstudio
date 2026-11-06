// GS-IR native port: CUDA kernels written for this plugin.
//
// 1. GGX split-sum pre-filtering of a cube map (bounds / forward / backward). Same weights as
//    nvdiffrec `specular_cubemap` (Munkberg et al. 2022, github.com/NVlabs/nvdiffrec,
//    render/renderutils/c_src/cubemap.cu), which GS-IR calls from pbr/light.py:
//    w = max(L.N, 0) * D_GGX(alpha^2, N.H) * texel_solid_angle / 4, N = V = R, with L restricted
//    to the cone dot(L, N) >= costheta_cutoff; the forward returns (sum w L_rgb, sum w).
// 2. SH occlusion reconstruction of GS-IR (github.com/lzhnb/GS-IR gs-ir/src/occlusion_kernel.cu
//    SH_reconstruction_kernel): mean over `num_samples` Hammersley GGX importance samples around
//    the normal (roughness 1 -> cosine-weighted hemisphere, one curand jitter of phi per pixel,
//    curand_init(1234, pixel, 0)) of clamp(sum_lm f_lm Y_lm(dir), 0, 1), bands 0..3.
#include <torch/extension.h>
#include <curand_kernel.h>
#include <cfloat>

#define CHECK_INPUT(x)                                                                    \
    TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor");                                \
    TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")

struct V3 {
    float x, y, z;
};
__device__ __forceinline__ V3 v3(float x, float y, float z) { return V3{x, y, z}; }
__device__ __forceinline__ float dot3(V3 a, V3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
__device__ __forceinline__ V3 add3(V3 a, V3 b) { return v3(a.x + b.x, a.y + b.y, a.z + b.z); }
__device__ __forceinline__ V3 scale3(V3 a, float s) { return v3(a.x * s, a.y * s, a.z * s); }
__device__ __forceinline__ V3 cross3(V3 a, V3 b) {
    return v3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}
__device__ __forceinline__ V3 safe_normalize(V3 a) {
    float l = sqrtf(dot3(a, a));
    return l > 0.0f ? scale3(a, 1.0f / l) : v3(0.0f, 0.0f, 0.0f);
}

// Texel centre (x, y) of face `side` of an N x N cube face -> unit direction (OpenGL layout:
// +X, -X, +Y, -Y, +Z, -Z), identical to nvdiffrec cube_to_dir.
__device__ V3 cube_dir(int x, int y, int side, int N) {
    float fx = 2.0f * (((float)x + 0.5f) / (float)N) - 1.0f;
    float fy = 2.0f * (((float)y + 0.5f) / (float)N) - 1.0f;
    switch (side) {
        case 0: return safe_normalize(v3(1.0f, -fy, -fx));
        case 1: return safe_normalize(v3(-1.0f, -fy, fx));
        case 2: return safe_normalize(v3(fx, 1.0f, fy));
        case 3: return safe_normalize(v3(fx, -1.0f, -fy));
        case 4: return safe_normalize(v3(fx, -fy, 1.0f));
        default: return safe_normalize(v3(-fx, -fy, -1.0f));
    }
}

// Solid-angle weight of texel (x, y) (product of the two atan extents).
__device__ float texel_area(int x, int y, int N) {
    if (N <= 1) return 1.0f;
    int H = N / 2;
    x = abs(x - H);
    y = abs(y - H);
    float dx = atanf((float)(x + 1) / (float)H) - atanf((float)x / (float)H);
    float dy = atanf((float)(y + 1) / (float)H) - atanf((float)y / (float)H);
    return dx * dy;
}

__device__ __forceinline__ float ndf_ggx(float alpha_sqr, float cos_theta) {
    float c = fminf(fmaxf(cos_theta, 0.0f), 1.0f);
    float d = (c * alpha_sqr - c) * c + 1.0f;
    return alpha_sqr / (d * d * 3.14159265358979f);
}

// ---------------------------------------------------------------- specular bounds
// out[face, y, x, s * 4 + {xmin, xmax, ymin, ymax}] = texel bbox on face s of the cone around
// the direction of output texel (x, y, face); empty -> xmin > xmax.
__global__ void spec_bounds_kernel(int N, float cutoff, int* __restrict__ out) {
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;
    int pz = blockIdx.z;
    if (px >= N || py >= N || pz >= 6) return;
    V3 R = cube_dir(px, py, pz, N);
    const int TILE = 16;
    int* o = out + ((pz * N + py) * N + px) * 24;
    for (int s = 0; s < 6; ++s) {
        int xmin = N - 1, xmax = 0, ymin = N - 1, ymax = 0;
        for (int tx = 0; tx < (N + TILE - 1) / TILE; ++tx) {
            for (int ty = 0; ty < (N + TILE - 1) / TILE; ++ty) {
                int sx = tx * TILE, sy = ty * TILE;
                int ex = min((tx + 1) * TILE, N), ey = min((ty + 1) * TILE, N);
                // conservative tile cull: max of dot(L, R) over the tile's corner box
                V3 c0 = cube_dir(sx, sy, s, N), c1 = cube_dir(ex, sy, s, N);
                V3 c2 = cube_dir(sx, ey, s, N), c3 = cube_dir(ex, ey, s, N);
                float mnx = fminf(fminf(c0.x, c1.x), fminf(c2.x, c3.x));
                float mxx = fmaxf(fmaxf(c0.x, c1.x), fmaxf(c2.x, c3.x));
                float mny = fminf(fminf(c0.y, c1.y), fminf(c2.y, c3.y));
                float mxy = fmaxf(fmaxf(c0.y, c1.y), fmaxf(c2.y, c3.y));
                float mnz = fminf(fminf(c0.z, c1.z), fminf(c2.z, c3.z));
                float mxz = fmaxf(fmaxf(c0.z, c1.z), fmaxf(c2.z, c3.z));
                float maxdp = fmaxf(mnx * R.x, mxx * R.x) + fmaxf(mny * R.y, mxy * R.y) +
                              fmaxf(mnz * R.z, mxz * R.z);
                if (maxdp < cutoff) continue;
                for (int y = sy; y < ey; ++y) {
                    for (int x = sx; x < ex; ++x) {
                        if (dot3(cube_dir(x, y, s, N), R) >= cutoff) {
                            xmin = min(xmin, x);
                            xmax = max(xmax, x);
                            ymin = min(ymin, y);
                            ymax = max(ymax, y);
                        }
                    }
                }
            }
        }
        o[s * 4 + 0] = xmin;
        o[s * 4 + 1] = xmax;
        o[s * 4 + 2] = ymin;
        o[s * 4 + 3] = ymax;
    }
}

__device__ __forceinline__ float spec_weight(V3 L, V3 R, float alpha_sqr, int x, int y, int N) {
    V3 H = safe_normalize(add3(L, R));
    float wi_n = fmaxf(dot3(L, R), 0.0f);
    float r_h = fmaxf(dot3(R, H), 0.0f);
    return wi_n * ndf_ggx(alpha_sqr, r_h) * texel_area(x, y, N) / 4.0f;
}

__global__ void spec_fwd_kernel(int N, float roughness, float cutoff,
                                const float* __restrict__ cube,   // [6, N, N, 3]
                                const int* __restrict__ bounds,   // [6, N, N, 24]
                                float* __restrict__ out) {        // [6, N, N, 4]
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;
    int pz = blockIdx.z;
    if (px >= N || py >= N || pz >= 6) return;
    V3 R = cube_dir(px, py, pz, N);
    float alpha = roughness * roughness, a2 = alpha * alpha;
    const int* b = bounds + ((pz * N + py) * N + px) * 24;
    float cr = 0.0f, cg = 0.0f, cb = 0.0f, wsum = 0.0f;
    for (int s = 0; s < 6; ++s) {
        int xmin = b[s * 4 + 0], xmax = b[s * 4 + 1], ymin = b[s * 4 + 2], ymax = b[s * 4 + 3];
        if (xmin > xmax) continue;
        for (int y = ymin; y <= ymax; ++y) {
            for (int x = xmin; x <= xmax; ++x) {
                V3 L = cube_dir(x, y, s, N);
                if (dot3(L, R) < cutoff) continue;
                float w = spec_weight(L, R, a2, x, y, N);
                const float* t = cube + ((s * N + y) * N + x) * 3;
                cr += t[0] * w;
                cg += t[1] * w;
                cb += t[2] * w;
                wsum += w;
            }
        }
    }
    float* o = out + ((pz * N + py) * N + px) * 4;
    o[0] = cr;
    o[1] = cg;
    o[2] = cb;
    o[3] = wsum;
}

__global__ void spec_bwd_kernel(int N, float roughness, float cutoff,
                                const int* __restrict__ bounds,  // [6, N, N, 24]
                                const float* __restrict__ dout,  // [6, N, N, 3]
                                float* __restrict__ dcube) {     // [6, N, N, 3]
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;
    int pz = blockIdx.z;
    if (px >= N || py >= N || pz >= 6) return;
    V3 R = cube_dir(px, py, pz, N);
    float alpha = roughness * roughness, a2 = alpha * alpha;
    const int* b = bounds + ((pz * N + py) * N + px) * 24;
    const float* g = dout + ((pz * N + py) * N + px) * 3;
    float gr = g[0], gg = g[1], gb = g[2];
    for (int s = 0; s < 6; ++s) {
        int xmin = b[s * 4 + 0], xmax = b[s * 4 + 1], ymin = b[s * 4 + 2], ymax = b[s * 4 + 3];
        if (xmin > xmax) continue;
        for (int y = ymin; y <= ymax; ++y) {
            for (int x = xmin; x <= xmax; ++x) {
                V3 L = cube_dir(x, y, s, N);
                if (dot3(L, R) < cutoff) continue;
                float w = spec_weight(L, R, a2, x, y, N);
                float* d = dcube + ((s * N + y) * N + x) * 3;
                atomicAdd(d + 0, gr * w);
                atomicAdd(d + 1, gg * w);
                atomicAdd(d + 2, gb * w);
            }
        }
    }
}

// ---------------------------------------------------------------- SH occlusion reconstruction
__device__ __forceinline__ float radical_inverse(unsigned int bits) {
    bits = (bits << 16u) | (bits >> 16u);
    bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);
    bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);
    bits = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u);
    bits = ((bits & 0x00FF00FFu) << 8u) | ((bits & 0xFF00FF00u) >> 8u);
    return float(bits) * 2.3283064365386963e-10f;
}

__global__ void sh_recon_kernel(int num_rays, int bands, int channels, int num_samples,
                                bool jitter,
                                const float* __restrict__ coeffs,     // [rays, C, bands^2]
                                const float* __restrict__ normals,    // [rays, 3]
                                const float* __restrict__ roughness,  // [rays]
                                float* __restrict__ out) {            // [rays, C]
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= num_rays) return;
    V3 Nn = v3(normals[r * 3 + 0], normals[r * 3 + 1], normals[r * 3 + 2]);
    if (Nn.x == 0.0f && Nn.y == 0.0f && Nn.z == 0.0f) return;
    float a = roughness[r] * roughness[r];
    V3 up = fabsf(Nn.z) < 0.999f ? v3(0.0f, 0.0f, 1.0f) : v3(1.0f, 0.0f, 0.0f);
    V3 tangent = safe_normalize(cross3(up, Nn));
    V3 bitangent = cross3(Nn, tangent);
    float eps = 0.0f;
    if (jitter) {
        curandState_t state;
        curand_init(1234, r, 0, &state);
        eps = curand_uniform(&state);
    }
    const int d2 = bands * bands;
    for (int c = 0; c < channels; ++c) out[r * channels + c] = 0.0f;
    for (int i = 0; i < num_samples; ++i) {
        float xi_x = float(i) / float(num_samples), xi_y = radical_inverse((unsigned int)i);
        float phi = 2.0f * 3.14159265358979f * (xi_x + eps);
        float ct = sqrtf((1.0f - xi_y) / (1.0f + (a * a - 1.0f) * xi_y));
        float st = sqrtf(fmaxf(1.0f - ct * ct, 0.0f));
        V3 Ht = v3(cosf(phi) * st, sinf(phi) * st, ct);
        V3 d = safe_normalize(
            add3(add3(scale3(tangent, Ht.x), scale3(bitangent, Ht.y)), scale3(Nn, Ht.z)));
        float x = d.x, y = d.y, z = d.z;
        float xx = x * x, yy = y * y, zz = z * z, xy = x * y, yz = y * z, xz = x * z;
        float Y[16];
        Y[0] = 0.28209479177387814f;
        Y[1] = -0.4886025119029199f * y;
        Y[2] = 0.4886025119029199f * z;
        Y[3] = -0.4886025119029199f * x;
        Y[4] = 1.0925484305920792f * xy;
        Y[5] = -1.0925484305920792f * yz;
        Y[6] = 0.31539156525252005f * (2.0f * zz - xx - yy);
        Y[7] = -1.0925484305920792f * xz;
        Y[8] = 0.5462742152960396f * (xx - yy);
        Y[9] = -0.5900435899266435f * y * (3.0f * xx - yy);
        Y[10] = 2.890611442640554f * xy * z;
        Y[11] = -0.4570457994644658f * y * (4.0f * zz - xx - yy);
        Y[12] = 0.3731763325901154f * z * (2.0f * zz - 3.0f * xx - 3.0f * yy);
        Y[13] = -0.4570457994644658f * x * (4.0f * zz - xx - yy);
        Y[14] = 1.4453057213202769f * z * (xx - yy);
        Y[15] = -0.5900435899266435f * x * (xx - 3.0f * yy);
        for (int c = 0; c < channels; ++c) {
            const float* f = coeffs + (r * channels + c) * d2;
            float acc = 0.0f;
            for (int k = 0; k < d2 && k < 16; ++k) acc += Y[k] * f[k];
            out[r * channels + c] += fminf(fmaxf(acc, 0.0f), 1.0f);
        }
    }
    for (int c = 0; c < channels; ++c) out[r * channels + c] /= float(num_samples);
}

// ---------------------------------------------------------------- bindings
static dim3 face_grid(int N) { return dim3((N + 15) / 16, (N + 15) / 16, 6); }

torch::Tensor specular_bounds(int64_t N, double cutoff) {
    auto out = torch::empty({6, N, N, 24},
                            torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA));
    spec_bounds_kernel<<<face_grid(N), dim3(16, 16)>>>((int)N, (float)cutoff,
                                                        out.data_ptr<int>());
    return out;
}

torch::Tensor specular_fwd(torch::Tensor cube, torch::Tensor bounds, double roughness,
                           double cutoff) {
    CHECK_INPUT(cube);
    CHECK_INPUT(bounds);
    int N = cube.size(1);
    auto out = torch::zeros({6, N, N, 4}, cube.options());
    spec_fwd_kernel<<<face_grid(N), dim3(16, 16)>>>(N, (float)roughness, (float)cutoff,
                                                     cube.data_ptr<float>(),
                                                     bounds.data_ptr<int>(),
                                                     out.data_ptr<float>());
    return out;
}

torch::Tensor specular_bwd(torch::Tensor bounds, torch::Tensor dout, double roughness,
                           double cutoff) {
    CHECK_INPUT(bounds);
    CHECK_INPUT(dout);
    int N = dout.size(1);
    auto dcube = torch::zeros({6, N, N, 3}, dout.options());
    spec_bwd_kernel<<<face_grid(N), dim3(16, 16)>>>(N, (float)roughness, (float)cutoff,
                                                     bounds.data_ptr<int>(),
                                                     dout.data_ptr<float>(),
                                                     dcube.data_ptr<float>());
    return dcube;
}

torch::Tensor sh_reconstruction(torch::Tensor coeffs, torch::Tensor normals,
                                torch::Tensor roughness, int64_t num_samples, int64_t bands,
                                bool jitter) {
    CHECK_INPUT(coeffs);
    CHECK_INPUT(normals);
    CHECK_INPUT(roughness);
    TORCH_CHECK(bands >= 1 && bands <= 4, "bands must be in [1, 4]");
    int rays = normals.size(0), C = coeffs.size(1);
    auto out = torch::zeros({rays, C}, normals.options());
    if (rays == 0) return out;
    sh_recon_kernel<<<(rays + 255) / 256, 256>>>(rays, (int)bands, C, (int)num_samples, jitter,
                                                 coeffs.data_ptr<float>(),
                                                 normals.data_ptr<float>(),
                                                 roughness.data_ptr<float>(),
                                                 out.data_ptr<float>());
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("specular_bounds", &specular_bounds);
    m.def("specular_fwd", &specular_fwd);
    m.def("specular_bwd", &specular_bwd);
    m.def("sh_reconstruction", &sh_reconstruction);
}
