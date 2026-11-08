#include <torch/extension.h>

std::vector<at::Tensor> wsr_forward(const at::Tensor &means2d, const at::Tensor &conics,
                                    const at::Tensor &colors, const at::Tensor &opac,
                                    const at::Tensor &weights, const int64_t W, const int64_t H,
                                    const int64_t tile_size, const at::Tensor &tile_offsets,
                                    const at::Tensor &flatten_ids);

std::vector<at::Tensor> wsr_backward(const at::Tensor &means2d, const at::Tensor &conics,
                                     const at::Tensor &colors, const at::Tensor &opac,
                                     const at::Tensor &weights, const int64_t W, const int64_t H,
                                     const int64_t tile_size, const at::Tensor &tile_offsets,
                                     const at::Tensor &flatten_ids, const at::Tensor &v_feats,
                                     const at::Tensor &v_wsum);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("wsr_forward", &wsr_forward, "Weighted Sum Rendering rasterization (forward)");
    m.def("wsr_backward", &wsr_backward, "Weighted Sum Rendering rasterization (backward)");
}
