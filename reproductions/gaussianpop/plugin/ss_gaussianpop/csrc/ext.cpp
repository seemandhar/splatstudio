#include <torch/extension.h>

std::vector<at::Tensor> pop_error(const at::Tensor &means2d, const at::Tensor &conics,
                                  const at::Tensor &opacities, const at::Tensor &colors,
                                  const at::Tensor &background, const int64_t W, const int64_t H,
                                  const int64_t tile_size, const at::Tensor &tile_offsets,
                                  const at::Tensor &flatten_ids, const int64_t n_max,
                                  const double eps, at::Tensor delta_se);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("pop_error", &pop_error,
          "GaussianPOP Alg. 1: accumulate per-Gaussian removal squared error for one view; "
          "returns the rendered image [H,W,3]");
}
