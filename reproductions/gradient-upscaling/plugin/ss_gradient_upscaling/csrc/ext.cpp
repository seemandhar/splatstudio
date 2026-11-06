#include <torch/extension.h>

std::vector<at::Tensor> jet_forward(const at::Tensor &means2d, const at::Tensor &conics,
                                    const at::Tensor &colors, const at::Tensor &opacities,
                                    const at::Tensor &bg, const int64_t W, const int64_t H,
                                    const int64_t tile_size, const at::Tensor &tile_offsets,
                                    const at::Tensor &flatten_ids, const double offx,
                                    const double offy);

std::vector<at::Tensor> jet_backward(const at::Tensor &means2d, const at::Tensor &conics,
                                     const at::Tensor &colors, const at::Tensor &opacities,
                                     const at::Tensor &bg, const int64_t W, const int64_t H,
                                     const int64_t tile_size, const at::Tensor &tile_offsets,
                                     const at::Tensor &flatten_ids, const double offx,
                                     const double offy, const at::Tensor &tfin,
                                     const at::Tensor &last_ids, const at::Tensor &v_out,
                                     const at::Tensor &v_alpha);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("jet_forward", &jet_forward, "3DGS image + analytic spatial derivatives (forward)");
    m.def("jet_backward", &jet_backward, "backward of jet_forward");
}
