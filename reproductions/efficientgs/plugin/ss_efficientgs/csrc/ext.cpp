// Python bindings for the EfficientGS statistics kernels (egs_raster.cu).
#include <torch/extension.h>

std::vector<torch::Tensor> egs_gradnorm(
    torch::Tensor means2d, torch::Tensor conics, torch::Tensor opacities, torch::Tensor colors,
    torch::Tensor tile_offsets, torch::Tensor flatten_ids, int64_t width, int64_t height,
    int64_t tile_size, torch::Tensor v_colors, torch::Tensor v_alphas);

std::vector<torch::Tensor> egs_stats(
    torch::Tensor means2d, torch::Tensor conics, torch::Tensor opacities,
    torch::Tensor tile_offsets, torch::Tensor flatten_ids, int64_t width, int64_t height,
    int64_t tile_size, int64_t top_k, bool weight_after_T, c10::optional<torch::Tensor> pix_loss);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("egs_gradnorm", &egs_gradnorm, "EfficientGS Eq. 6 per-pixel gradient-norm sums");
    m.def("egs_stats", &egs_stats, "EfficientGS Eq. 8 top-K flags and Eq. 9 loss weights");
}
