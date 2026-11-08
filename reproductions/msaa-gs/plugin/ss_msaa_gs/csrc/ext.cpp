#include <torch/extension.h>

std::vector<at::Tensor> msaa_forward(const at::Tensor &means2d, const at::Tensor &conics,
                                     const at::Tensor &colors, const at::Tensor &opacities,
                                     const at::Tensor &offsets, const int64_t W, const int64_t H,
                                     const int64_t tile_size, const at::Tensor &tile_offsets,
                                     const at::Tensor &flatten_ids);

std::vector<at::Tensor> msaa_backward(
    const at::Tensor &means2d, const at::Tensor &conics, const at::Tensor &colors,
    const at::Tensor &opacities, const at::Tensor &offsets, const int64_t W, const int64_t H,
    const int64_t tile_size, const at::Tensor &tile_offsets, const at::Tensor &flatten_ids,
    const at::Tensor &sample_T, const at::Tensor &last_ids, const at::Tensor &v_render_colors,
    const at::Tensor &v_render_alphas);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("msaa_forward", &msaa_forward, "MSAA-GS multi-sample 3DGS rasterization (forward)");
    m.def("msaa_backward", &msaa_backward, "MSAA-GS multi-sample 3DGS rasterization (backward)");
}
