// Python bindings for the Improving-ADC statistics kernel.
#include <torch/extension.h>

std::vector<torch::Tensor> adc_stats(
    torch::Tensor means2d, torch::Tensor conics, torch::Tensor opacities,
    torch::Tensor tile_offsets, torch::Tensor flatten_ids, int64_t width, int64_t height,
    int64_t tile_size);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("adc_stats", &adc_stats, "Per-Gaussian contributing-pixel count and blend-weight sum");
}
