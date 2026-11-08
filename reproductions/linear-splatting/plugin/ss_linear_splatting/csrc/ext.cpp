#include <torch/extension.h>

std::vector<at::Tensor> linear_forward(const at::Tensor &means2d, const at::Tensor &conics,
                                       const at::Tensor &colors, const at::Tensor &opacities,
                                       const c10::optional<at::Tensor> &backgrounds,
                                       const int64_t W, const int64_t H, const int64_t tile_size,
                                       const at::Tensor &tile_offsets,
                                       const at::Tensor &flatten_ids, const double lam);

std::vector<at::Tensor> linear_backward(
    const at::Tensor &means2d, const at::Tensor &conics, const at::Tensor &colors,
    const at::Tensor &opacities, const c10::optional<at::Tensor> &backgrounds, const int64_t W,
    const int64_t H, const int64_t tile_size, const at::Tensor &tile_offsets,
    const at::Tensor &flatten_ids, const double lam, const double ags_c,
    const at::Tensor &render_alphas, const at::Tensor &last_ids,
    const at::Tensor &v_render_colors, const at::Tensor &v_render_alphas);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("linear_forward", &linear_forward, "3DLS linear-kernel rasterization (forward)");
    m.def("linear_backward", &linear_backward, "3DLS linear-kernel rasterization (backward, AGS)");
}
