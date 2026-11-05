// Python bindings for the Taming 3DGS statistics kernel.
#include <torch/extension.h>

std::vector<torch::Tensor> taming_stats(
    torch::Tensor means2d, torch::Tensor conics, torch::Tensor opacities,
    torch::Tensor tile_offsets, torch::Tensor flatten_ids, torch::Tensor pixel_weights,
    int64_t width, int64_t height, int64_t tile_size);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("taming_stats", &taming_stats, "Taming 3DGS per-Gaussian pixel statistics");
}
