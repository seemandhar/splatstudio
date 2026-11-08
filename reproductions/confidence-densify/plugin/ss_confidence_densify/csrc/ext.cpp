// Python bindings for the confidence-densify blend-statistics kernel.
#include <torch/extension.h>

std::vector<torch::Tensor> blend_stats(
    torch::Tensor means2d, torch::Tensor conics, torch::Tensor opacities,
    torch::Tensor tile_offsets, torch::Tensor flatten_ids, torch::Tensor error, int64_t width,
    int64_t height, int64_t tile_size);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("blend_stats", &blend_stats,
          "Per-Gaussian max_p alpha*T and sum_p error(p)*alpha*T for one camera");
}
