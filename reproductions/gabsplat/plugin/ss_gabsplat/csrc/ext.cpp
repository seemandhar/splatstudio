#include <torch/extension.h>

std::vector<at::Tensor> gabor_forward(const at::Tensor &means2d, const at::Tensor &conics,
                                      const at::Tensor &colors, const at::Tensor &opacities,
                                      const at::Tensor &gabs,
                                      const c10::optional<at::Tensor> &backgrounds,
                                      const int64_t W, const int64_t H, const int64_t tile_size,
                                      const at::Tensor &tile_offsets,
                                      const at::Tensor &flatten_ids);

std::vector<at::Tensor> gabor_backward(
    const at::Tensor &means2d, const at::Tensor &conics, const at::Tensor &colors,
    const at::Tensor &opacities, const at::Tensor &gabs,
    const c10::optional<at::Tensor> &backgrounds, const int64_t W, const int64_t H,
    const int64_t tile_size, const at::Tensor &tile_offsets, const at::Tensor &flatten_ids,
    const at::Tensor &render_alphas, const at::Tensor &last_ids,
    const at::Tensor &v_render_colors, const at::Tensor &v_render_alphas);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("gabor_forward", &gabor_forward, "3DGabSplat 2D Gabor rasterization (forward)");
    m.def("gabor_backward", &gabor_backward, "3DGabSplat 2D Gabor rasterization (backward)");
}
