#include <torch/extension.h>

void significance(const at::Tensor &means2d, const at::Tensor &conics,
                  const at::Tensor &opacities, const int64_t W, const int64_t H,
                  const int64_t tile_size, const at::Tensor &tile_offsets,
                  const at::Tensor &flatten_ids, at::Tensor count, at::Tensor score);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("significance", &significance,
          "LightGaussian renderCUDA_count: accumulate per-Gaussian hit count and sum of "
          "opacity * T over one view's pixels (in place)");
}
