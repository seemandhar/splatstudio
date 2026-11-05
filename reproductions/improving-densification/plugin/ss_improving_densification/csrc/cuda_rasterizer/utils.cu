#include "utils.h"
#include "auxiliary.h"


__global__ void compute_relocation(
    int P,
    float* opacity_old,
    float* scale_old,
    int* N,
    float* binoms,
    int n_max,
    float* opacity_new,
    float* scale_new)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= P) return;

    int N_idx = N[idx];
    float denom_sum = 0.0f;

    opacity_new[idx] = 1.0f - powf(1.0f - opacity_old[idx], 1.0f / N_idx);

    for (int i = 1; i <= N_idx; ++i) {
        for (int k = 0; k <= (i - 1); ++k) {
            float bin_coeff = binoms[(i - 1) * n_max + k];
            float term = (pow(-1.0f, k) / sqrtf(k + 1.0f)) * powf(opacity_new[idx], k + 1.0f);
            denom_sum += (bin_coeff * term);
        }
    }
    float coeff = opacity_old[idx] / denom_sum;
    for (int axis = 0; axis < 3; ++axis) {
        scale_new[idx * 3 + axis] = coeff * scale_old[idx * 3 + axis];
    }
}

void UTILS::ComputeRelocation(
    int P,
    float* opacity_old,
    float* scale_old,
    int* N,
    float* binoms,
    int n_max,
    float* opacity_new,
    float* scale_new)
{
    int num_blocks = (P + 255) / 256;
    dim3 block(256, 1, 1);
    dim3 grid(num_blocks, 1, 1);
    compute_relocation<<<grid, block>>>(P, opacity_old, scale_old, N, binoms, n_max, opacity_new, scale_new);
}
