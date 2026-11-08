// Python bindings for the LiteGS opacity-gradient statistics kernel.
#include <torch/extension.h>

std::vector<torch::Tensor> litegs_stats(
    torch::Tensor means2d, torch::Tensor conics, torch::Tensor opacities, torch::Tensor colors,
    torch::Tensor tile_offsets, torch::Tensor flatten_ids, torch::Tensor final_rgb,
    torch::Tensor v_rgb, int64_t width, int64_t height, int64_t tile_size);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("litegs_stats", &litegs_stats, "LiteGS per-Gaussian opacity-gradient moments");
}
