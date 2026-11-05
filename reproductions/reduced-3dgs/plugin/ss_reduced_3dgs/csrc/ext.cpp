// Python bindings for the Reduced-3DGS statistics kernels (transmittance.cu, knn.cu, kmeans.cu).
#include <torch/extension.h>

std::vector<torch::Tensor> transmittance_stats(
    torch::Tensor means2d, torch::Tensor conics, torch::Tensor opacities,
    torch::Tensor tile_offsets, torch::Tensor flatten_ids, int64_t width, int64_t height,
    int64_t tile_size, double alpha_max);

std::vector<torch::Tensor> knn_index(torch::Tensor points, int64_t K);

torch::Tensor kmeans_assign(torch::Tensor values, torch::Tensor centers);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("transmittance_stats", &transmittance_stats,
          "Per-Gaussian accumulated transmittance and blended-pixel count (gsplat tile lists)");
    m.def("kmeans_assign", &kmeans_assign, "1-D nearest codebook centre (official updateIds)");
    m.def("knn_index", &knn_index, "Exact K nearest neighbours (self excluded), simple-knn boxKnn2");
}
