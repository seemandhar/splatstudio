// Python binding for the vendored simple-knn distCUDA2 (Inria 3DGS licence, csrc/LICENSE-3DGS.md).
#include <torch/extension.h>

torch::Tensor distCUDA2(const torch::Tensor& points);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("dist_cuda2", &distCUDA2, "simple-knn mean squared distance to 3 neighbours");
}
