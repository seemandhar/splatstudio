// Python binding for the LP-3DGS RadSplat max-contribution score kernel.
#include <torch/extension.h>

torch::Tensor max_contribution(
    torch::Tensor means2d, torch::Tensor conics, torch::Tensor opacities,
    torch::Tensor tile_offsets, torch::Tensor flatten_ids, int64_t width, int64_t height,
    int64_t tile_size, torch::Tensor score);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("max_contribution", &max_contribution,
          "In-place score_i = max(score_i, max over pixels of alpha_i T_i) for one view");
}
