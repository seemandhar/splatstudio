/*
 * 1-D nearest-centre assignment for the Reduced-3DGS codebook K-means, vendored from the
 * official code (https://github.com/graphdeco-inria/reduced-3dgs, commit dc905c1,
 * submodules/diff-gaussian-rasterization/reduced_3dgs/kmeans.cu: distanceCUDA, updateIdsCUDA),
 * Copyright (C) 2023, Inria, GRAPHDECO research group; free for non-commercial, research and
 * evaluation use (LICENSE-3DGS.md). Ties go to the lowest centre index (strict `<`), exactly as
 * in the original. Centres are staged in shared memory, so at most 256 centres are supported
 * (the paper's codebook size). The centre update (mean of members, empty -> 0) is done in
 * PyTorch by the caller, as in Reduced3DGS::kmeans.
 */
#include <torch/extension.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

__device__ float distanceCUDA(const float x1, const float x2) {
    return sqrt((x2 - x1) * (x2 - x1));
}

__global__ void updateIdsCUDA(const float *values, int *ids, const float *centers,
                              const int n_values, const int n_centers) {
    auto idx = cg::this_grid().thread_rank();
    auto block = cg::this_thread_block();
    __shared__ float collected_centers[256];
    // all threads of the block help staging, including those past n_values
    if (block.thread_rank() < n_centers) collected_centers[block.thread_rank()] = centers[block.thread_rank()];
    block.sync();
    if (idx >= n_values) return;
    float min_dist = INFINITY;
    int closest_centroid = 0;
    for (int i = 0; i < n_centers; ++i) {
        float dist = distanceCUDA(values[idx], collected_centers[i]);
        if (dist < min_dist) {
            min_dist = dist;
            closest_centroid = i;
        }
    }
    ids[idx] = closest_centroid;
}

torch::Tensor kmeans_assign(torch::Tensor values, torch::Tensor centers) {
    TORCH_CHECK(values.is_cuda() && values.is_contiguous() && values.scalar_type() == torch::kFloat32,
                "values must be contiguous float32 CUDA");
    TORCH_CHECK(centers.is_cuda() && centers.is_contiguous() && centers.scalar_type() == torch::kFloat32,
                "centers must be contiguous float32 CUDA");
    const int n_values = values.numel();
    const int n_centers = centers.numel();
    TORCH_CHECK(n_centers >= 1 && n_centers <= 256, "1..256 centres supported");
    auto ids = torch::zeros({n_values}, values.options().dtype(torch::kInt32));
    if (n_values == 0) return ids;
    auto stream = at::cuda::getCurrentCUDAStream();
    updateIdsCUDA<<<(n_values + 255) / 256, 256, 0, stream>>>(
        values.data_ptr<float>(), ids.data_ptr<int>(), centers.data_ptr<float>(), n_values,
        n_centers);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return ids;
}
