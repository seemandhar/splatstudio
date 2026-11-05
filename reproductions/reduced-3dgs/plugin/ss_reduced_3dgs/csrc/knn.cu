/*
 * Exact K-nearest-neighbour indices (self excluded), vendored from the official Reduced-3DGS
 * simple-knn module (https://github.com/graphdeco-inria/reduced-3dgs, commit dc905c1,
 * submodules/simple-knn/simple_knn.cu: coord2Morton, boxMinMax, distBoxPoint, updateKBest2,
 * boxKnn2, SimpleKNN::knn_index2; spatial.cu: distCudaIndices2 a.k.a. `distIndex2`).
 * Copyright (C) 2023, Inria, GRAPHDECO research group; free for non-commercial, research and
 * evaluation use (LICENSE-3DGS.md).
 * Changes: only the knn_index2 path is kept; boxMinMax reduces over its template block size B
 * (the original reduces over BOX_SIZE=1024 while launched with 128 threads, which reads
 * uninitialised shared memory and can only enlarge boxes, i.e. it is slower but equally exact);
 * returns [P,K] tensors; data<float>() -> data_ptr<float>().
 */

#define BOX_SIZE2 128
#include <torch/extension.h>
#include <c10/cuda/CUDAException.h>

#include <cfloat>
#include <cub/cub.cuh>
#include <cub/device/device_radix_sort.cuh>
#include <cuda_runtime_api.h>
#include <thrust/device_vector.h>
#include <thrust/sequence.h>
#include <cooperative_groups.h>
#include <vector>

namespace cg = cooperative_groups;

struct CustomMin {
    __device__ __forceinline__ float3 operator()(const float3 &a, const float3 &b) const {
        return {min(a.x, b.x), min(a.y, b.y), min(a.z, b.z)};
    }
};

struct CustomMax {
    __device__ __forceinline__ float3 operator()(const float3 &a, const float3 &b) const {
        return {max(a.x, b.x), max(a.y, b.y), max(a.z, b.z)};
    }
};

__host__ __device__ uint32_t prepMorton(uint32_t x) {
    x = (x | (x << 16)) & 0x030000FF;
    x = (x | (x << 8)) & 0x0300F00F;
    x = (x | (x << 4)) & 0x030C30C3;
    x = (x | (x << 2)) & 0x09249249;
    return x;
}

__host__ __device__ uint32_t coord2Morton(float3 coord, float3 minn, float3 maxx) {
    uint32_t x = prepMorton(((coord.x - minn.x) / (maxx.x - minn.x)) * ((1 << 10) - 1));
    uint32_t y = prepMorton(((coord.y - minn.y) / (maxx.y - minn.y)) * ((1 << 10) - 1));
    uint32_t z = prepMorton(((coord.z - minn.z) / (maxx.z - minn.z)) * ((1 << 10) - 1));
    return x | (y << 1) | (z << 2);
}

__global__ void coord2MortonKernel(int P, const float3 *points, float3 minn, float3 maxx,
                                   uint32_t *codes) {
    auto idx = cg::this_grid().thread_rank();
    if (idx >= P) return;
    codes[idx] = coord2Morton(points[idx], minn, maxx);
}

struct MinMax {
    float3 minn;
    float3 maxx;
};

template <uint32_t B>
__global__ void boxMinMax(uint32_t P, const float3 *points, const uint32_t *indices,
                          MinMax *boxes) {
    auto idx = cg::this_grid().thread_rank();
    MinMax me;
    if (idx < P) {
        me.minn = points[indices[idx]];
        me.maxx = points[indices[idx]];
    } else {
        me.minn = {FLT_MAX, FLT_MAX, FLT_MAX};
        me.maxx = {-FLT_MAX, -FLT_MAX, -FLT_MAX};
    }
    __shared__ MinMax redResult[B];
    for (int off = B / 2; off >= 1; off /= 2) {
        if (threadIdx.x < 2 * off) redResult[threadIdx.x] = me;
        __syncthreads();
        if (threadIdx.x < off) {
            MinMax other = redResult[threadIdx.x + off];
            me.minn.x = min(me.minn.x, other.minn.x);
            me.minn.y = min(me.minn.y, other.minn.y);
            me.minn.z = min(me.minn.z, other.minn.z);
            me.maxx.x = max(me.maxx.x, other.maxx.x);
            me.maxx.y = max(me.maxx.y, other.maxx.y);
            me.maxx.z = max(me.maxx.z, other.maxx.z);
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) boxes[blockIdx.x] = me;
}

__device__ __host__ float distBoxPoint(const MinMax &box, const float3 &p) {
    float3 diff = {0, 0, 0};
    if (p.x < box.minn.x || p.x > box.maxx.x)
        diff.x = min(abs(p.x - box.minn.x), abs(p.x - box.maxx.x));
    if (p.y < box.minn.y || p.y > box.maxx.y)
        diff.y = min(abs(p.y - box.minn.y), abs(p.y - box.maxx.y));
    if (p.z < box.minn.z || p.z > box.maxx.z)
        diff.z = min(abs(p.z - box.minn.z), abs(p.z - box.maxx.z));
    return diff.x * diff.x + diff.y * diff.y + diff.z * diff.z;
}

__device__ void updateKBest2(float &reject, int K, int index, const float3 &ref,
                             const float3 &point, float *__restrict__ knn,
                             int *__restrict__ indices) {
    float3 d = {point.x - ref.x, point.y - ref.y, point.z - ref.z};
    float dist = d.x * d.x + d.y * d.y + d.z * d.z;
    if (dist >= reject) return;
    float test_reject = dist;
    int maxint = -1;
    for (int j = 0; j < K; j++) {
        if (test_reject < knn[j]) {
            test_reject = knn[j];
            maxint = j;
        }
    }
    if (maxint != -1) {
        knn[maxint] = dist;
        indices[maxint] = index;
    }
    reject = min(reject, test_reject);
}

__global__ void boxKnn2(int K, uint32_t P, const float3 *__restrict__ points,
                        const uint32_t *__restrict__ indices, const MinMax *__restrict__ boxes,
                        float *__restrict__ dists, int *__restrict__ index_space) {
    int idx = cg::this_grid().thread_rank();
    if (idx >= P) return;
    const float3 point = points[indices[idx]];
    float *best = dists + indices[idx] * K;
    int *best_ind = index_space + indices[idx] * K;
    for (int i = 0; i < K; i++) best[i] = FLT_MAX;
    float reject = FLT_MAX;
    int b = idx / BOX_SIZE2;
    int lo = b, hi = b;
    const int num_boxes = (P + BOX_SIZE2 - 1) / BOX_SIZE2;
    for (int iter = 0; iter < num_boxes; iter++) {
        MinMax box = boxes[b];
        float dist = distBoxPoint(box, point);
        if (dist < reject) {
            for (int i = b * BOX_SIZE2; i < min(P, (b + 1) * BOX_SIZE2); i++) {
                if (i == idx) continue;
                const int other_idx = indices[i];
                updateKBest2(reject, K, other_idx, point, points[other_idx], best, best_ind);
            }
        }
        bool odd = iter & 1;
        b = (odd && hi == num_boxes - 1) || (!odd && lo > 0) ? --lo : ++hi;
    }
}

// returns squared distances [P,K] (float) and neighbour indices [P,K] (int32; -1 if P <= K)
std::vector<torch::Tensor> knn_index(torch::Tensor points_t, int64_t K) {
    TORCH_CHECK(points_t.is_cuda() && points_t.is_contiguous(), "points must be contiguous CUDA");
    TORCH_CHECK(points_t.scalar_type() == torch::kFloat32 && points_t.size(1) == 3, "points [P,3] f32");
    const int P = points_t.size(0);
    auto dists_t = torch::full({P, K}, 0.0, points_t.options());
    auto idx_t = torch::full({P, K}, -1, points_t.options().dtype(torch::kInt32));
    if (P == 0) return {dists_t, idx_t};
    float3 *points = (float3 *)points_t.data_ptr<float>();

    float3 *result;
    cudaMalloc(&result, sizeof(float3));
    size_t temp_storage_bytes;
    float3 init = {0, 0, 0}, minn, maxx;
    cub::DeviceReduce::Reduce(nullptr, temp_storage_bytes, points, result, P, CustomMin(), init);
    thrust::device_vector<char> temp_storage(temp_storage_bytes);
    cub::DeviceReduce::Reduce(temp_storage.data().get(), temp_storage_bytes, points, result, P,
                              CustomMin(), init);
    cudaMemcpy(&minn, result, sizeof(float3), cudaMemcpyDeviceToHost);
    cub::DeviceReduce::Reduce(temp_storage.data().get(), temp_storage_bytes, points, result, P,
                              CustomMax(), init);
    cudaMemcpy(&maxx, result, sizeof(float3), cudaMemcpyDeviceToHost);

    thrust::device_vector<uint32_t> morton(P);
    thrust::device_vector<uint32_t> morton_sorted(P);
    coord2MortonKernel<<<(P + 255) / 256, 256>>>(P, points, minn, maxx, morton.data().get());
    thrust::device_vector<uint32_t> indices(P);
    thrust::sequence(indices.begin(), indices.end());
    thrust::device_vector<uint32_t> indices_sorted(P);
    cub::DeviceRadixSort::SortPairs(nullptr, temp_storage_bytes, morton.data().get(),
                                    morton_sorted.data().get(), indices.data().get(),
                                    indices_sorted.data().get(), P);
    temp_storage.resize(temp_storage_bytes);
    cub::DeviceRadixSort::SortPairs(temp_storage.data().get(), temp_storage_bytes,
                                    morton.data().get(), morton_sorted.data().get(),
                                    indices.data().get(), indices_sorted.data().get(), P);

    uint32_t num_boxes2 = (P + BOX_SIZE2 - 1) / BOX_SIZE2;
    thrust::device_vector<MinMax> boxes(num_boxes2);
    boxMinMax<BOX_SIZE2><<<num_boxes2, BOX_SIZE2>>>(P, points, indices_sorted.data().get(),
                                                   boxes.data().get());
    boxKnn2<<<(P + 255) / 256, 256>>>(K, P, points, indices_sorted.data().get(),
                                      boxes.data().get(), dists_t.data_ptr<float>(),
                                      idx_t.data_ptr<int>());
    cudaFree(result);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {dists_t, idx_t};
}
