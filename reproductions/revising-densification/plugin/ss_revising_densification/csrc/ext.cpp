// Python bindings for the Revising-Densification error-attribution kernel.
#include <torch/extension.h>

std::vector<torch::Tensor> err_stats(
    torch::Tensor means2d, torch::Tensor conics, torch::Tensor opacities,
    torch::Tensor tile_offsets, torch::Tensor flatten_ids, torch::Tensor error, int64_t width,
    int64_t height, int64_t tile_size);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("err_stats", &err_stats,
          "Per-Gaussian error attribution sum_u E(u) alpha T and blend-weight sum sum_u alpha T");
}
