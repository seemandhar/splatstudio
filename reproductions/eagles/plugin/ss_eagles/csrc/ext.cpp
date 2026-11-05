// Python binding for the EAGLES influence kernel (influence.cu).
#include <torch/extension.h>

torch::Tensor influence_stats(
    torch::Tensor means2d, torch::Tensor conics, torch::Tensor opacities,
    torch::Tensor tile_offsets, torch::Tensor flatten_ids, int64_t width, int64_t height,
    int64_t tile_size, double alpha_max);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("influence_stats", &influence_stats,
          "Per-Gaussian influence sum_p alpha*T on gsplat tile lists (EAGLES Eq. 3)");
}
