#include <torch/extension.h>

std::vector<at::Tensor> sphere_forward(const at::Tensor &means2d, const at::Tensor &conics,
                                       const at::Tensor &colors, const at::Tensor &opacities,
                                       const int64_t W, const int64_t H, const int64_t tile_size,
                                       const at::Tensor &tile_offsets,
                                       const at::Tensor &flatten_ids,
                                       const std::vector<double> &cam, const int64_t mode,
                                       const double alpha_max);

std::vector<at::Tensor> sphere_backward(
    const at::Tensor &means2d, const at::Tensor &conics, const at::Tensor &colors,
    const at::Tensor &opacities, const int64_t W, const int64_t H, const int64_t tile_size,
    const at::Tensor &tile_offsets, const at::Tensor &flatten_ids,
    const std::vector<double> &cam, const int64_t mode, const double alpha_max,
    const at::Tensor &render_alphas, const at::Tensor &last_ids,
    const at::Tensor &v_render_colors, const at::Tensor &v_render_alphas);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("sphere_forward", &sphere_forward, "Unit-sphere (tangent-plane) rasterization forward");
    m.def("sphere_backward", &sphere_backward, "Unit-sphere (tangent-plane) rasterization backward");
}
