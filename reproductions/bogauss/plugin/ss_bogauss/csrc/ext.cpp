// Python bindings for the BOGausS per-Gaussian statistics kernel.
#include <torch/extension.h>

std::vector<torch::Tensor> bogauss_stats(
    torch::Tensor means2d, torch::Tensor conics, torch::Tensor opacities,
    torch::Tensor tile_offsets, torch::Tensor flatten_ids, torch::Tensor pixel_error,
    int64_t width, int64_t height, int64_t tile_size);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("bogauss_stats", &bogauss_stats, "BOGausS per-Gaussian blend/SE/footprint sums");
}
