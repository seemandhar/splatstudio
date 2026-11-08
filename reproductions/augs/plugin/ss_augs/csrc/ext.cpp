#include <torch/extension.h>

std::vector<at::Tensor> contrib_stats(const at::Tensor &means2d, const at::Tensor &conics,
                                      const at::Tensor &opacities, const int64_t W,
                                      const int64_t H, const int64_t tile_size,
                                      const at::Tensor &tile_offsets,
                                      const at::Tensor &flatten_ids, const double t_threshold,
                                      const double alpha_max, const bool inclusive_stop);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("contrib_stats", &contrib_stats,
          "Forward-only blending statistics: T-threshold Gaussian per pixel and per-Gaussian "
          "contributing-pixel counts");
}
