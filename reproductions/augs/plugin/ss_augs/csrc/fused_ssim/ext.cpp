// 2D-only binding for the vendored fused-ssim kernels (github.com/rahul-goel/fused-ssim, MIT).
#include <torch/extension.h>
#include "ssim.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("fusedssim", &fusedssim);
  m.def("fusedssim_backward", &fusedssim_backward);
}
