// Adapted from FasterGSCudaBackend/torch_bindings/bindings.cpp (nerficg-project/faster-gaussian-splatting
// @ 090e8ad, Apache-2.0). Only the rasterization and Adam modules are bound: the filter3d
// (Mip-Splatting) and densification (3DGS-MCMC) modules are optional borrowed features that are
// not part of the paper's main Faster-GS configuration.
#include <torch/extension.h>
#include "rasterization_api.h"
#include "adam.h"

namespace rasterization_api = faster_gs::rasterization;
namespace adam_api = faster_gs::adam;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &rasterization_api::forward_wrapper);
    m.def("backward", &rasterization_api::backward_wrapper);
    m.def("inference", &rasterization_api::inference_wrapper);
    m.def("pruning_scores", &rasterization_api::pruning_scores_wrapper);
    m.def("adam_step", &adam_api::adam_step_wrapper);
}
