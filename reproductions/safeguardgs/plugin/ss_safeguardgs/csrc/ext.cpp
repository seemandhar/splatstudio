// Python bindings for the SafeguardGS top-k statistics kernel (sg_topk.cu).
#include <torch/extension.h>

std::vector<torch::Tensor> sg_topk(
    torch::Tensor means2d, torch::Tensor conics, torch::Tensor opacities, torch::Tensor colors,
    torch::Tensor tile_offsets, torch::Tensor flatten_ids, int64_t width, int64_t height,
    int64_t tile_size, torch::Tensor image_gt, int64_t func_id, double p_coef, double c_coef,
    int64_t top_k, bool tile_mode, torch::Tensor pad_offsets, int64_t n_slots);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("sg_topk", &sg_topk, "SafeguardGS per-tile slot keys (official) / per-pixel top-k flags");
}
