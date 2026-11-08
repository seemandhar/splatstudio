#include <torch/extension.h>

std::vector<at::Tensor> vol_forward(const at::Tensor &mk, const at::Tensor &pa,
                                    const at::Tensor &pb, const at::Tensor &colors,
                                    const double fx, const double fy, const double cx,
                                    const double cy, const int64_t W, const int64_t H,
                                    const int64_t tile_size, const at::Tensor &tile_offsets,
                                    const at::Tensor &flatten_ids);

std::vector<at::Tensor> vol_backward(
    const at::Tensor &mk, const at::Tensor &pa, const at::Tensor &pb, const at::Tensor &colors,
    const double fx, const double fy, const double cx, const double cy, const int64_t W,
    const int64_t H, const int64_t tile_size, const at::Tensor &tile_offsets,
    const at::Tensor &flatten_ids, const at::Tensor &render_alphas, const at::Tensor &last_ids,
    const at::Tensor &v_render_colors, const at::Tensor &v_render_alphas);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("vol_forward", &vol_forward, "Vol3DGS volumetric rasterization (forward)");
    m.def("vol_backward", &vol_backward, "Vol3DGS volumetric rasterization (backward)");
}
