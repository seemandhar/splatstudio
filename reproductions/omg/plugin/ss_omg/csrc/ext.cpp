// Python bindings for the Mini-Splatting statistics kernels and the vendored simple-knn.
#include <torch/extension.h>

std::vector<torch::Tensor> ms_stats(
    torch::Tensor means2d, torch::Tensor conics, torch::Tensor opacities,
    torch::Tensor tile_offsets, torch::Tensor flatten_ids, int64_t width, int64_t height,
    int64_t tile_size, bool weights);

std::vector<torch::Tensor> ms_depth(
    torch::Tensor means2d, torch::Tensor conics, torch::Tensor opacities,
    torch::Tensor tile_offsets, torch::Tensor flatten_ids, int64_t width, int64_t height,
    int64_t tile_size, torch::Tensor means3d, torch::Tensor rot_wl, torch::Tensor inv_scale,
    torch::Tensor c2w, double fx, double fy, double cx, double cy);

torch::Tensor distCUDA2(const torch::Tensor& points);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("ms_stats", &ms_stats, "Mini-Splatting per-Gaussian blending statistics");
    m.def("ms_depth", &ms_depth, "Mini-Splatting max-weight mid-point depth points");
    m.def("dist_cuda2", &distCUDA2, "simple-knn mean squared distance to 3 neighbours");
}
