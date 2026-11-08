#include <torch/extension.h>

std::vector<at::Tensor> beta_forward(const at::Tensor &means2d, const at::Tensor &conics,
                                     const at::Tensor &colors, const at::Tensor &opacities,
                                     const at::Tensor &betas,
                                     const c10::optional<at::Tensor> &backgrounds,
                                     const int64_t W, const int64_t H, const int64_t tile_size,
                                     const at::Tensor &tile_offsets,
                                     const at::Tensor &flatten_ids);

std::vector<at::Tensor> beta_backward(
    const at::Tensor &means2d, const at::Tensor &conics, const at::Tensor &colors,
    const at::Tensor &opacities, const at::Tensor &betas,
    const c10::optional<at::Tensor> &backgrounds, const int64_t W, const int64_t H,
    const int64_t tile_size, const at::Tensor &tile_offsets, const at::Tensor &flatten_ids,
    const at::Tensor &render_alphas, const at::Tensor &last_ids,
    const at::Tensor &v_render_colors, const at::Tensor &v_render_alphas);

at::Tensor sb_forward(const at::Tensor &dirs, const at::Tensor &c0, const at::Tensor &coeffs,
                      const c10::optional<at::Tensor> &masks);

std::vector<at::Tensor> sb_backward(const at::Tensor &dirs, const at::Tensor &coeffs,
                                    const c10::optional<at::Tensor> &masks,
                                    const at::Tensor &v_colors);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("beta_forward", &beta_forward, "Beta-kernel rasterization (forward)");
    m.def("beta_backward", &beta_backward, "Beta-kernel rasterization (backward)");
    m.def("sb_forward", &sb_forward, "Spherical Beta colour (forward)");
    m.def("sb_backward", &sb_backward, "Spherical Beta colour (backward)");
}
