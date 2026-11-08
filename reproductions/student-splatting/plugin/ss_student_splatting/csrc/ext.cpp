#include <torch/extension.h>

std::vector<at::Tensor> t_forward(const at::Tensor &means2d, const at::Tensor &conics,
                                  const at::Tensor &colors, const at::Tensor &opacities,
                                  const at::Tensor &nus,
                                  const c10::optional<at::Tensor> &backgrounds,
                                  const int64_t W, const int64_t H, const int64_t tile_size,
                                  const at::Tensor &tile_offsets, const at::Tensor &flatten_ids,
                                  const double alpha_clip);

std::vector<at::Tensor> t_backward(
    const at::Tensor &means2d, const at::Tensor &conics, const at::Tensor &colors,
    const at::Tensor &opacities, const at::Tensor &nus,
    const c10::optional<at::Tensor> &backgrounds, const int64_t W, const int64_t H,
    const int64_t tile_size, const at::Tensor &tile_offsets, const at::Tensor &flatten_ids,
    const at::Tensor &render_alphas, const at::Tensor &last_ids,
    const at::Tensor &v_render_colors, const at::Tensor &v_render_alphas,
    const double alpha_clip, const bool exact_nu_grad);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("t_forward", &t_forward, "Student's t splatting with signed opacity (forward)");
    m.def("t_backward", &t_backward, "Student's t splatting with signed opacity (backward)");
}
