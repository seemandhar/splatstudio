// From the official c3dgs code (submodules/weighted_distance/ext.cpp), see LICENSE-c3dgs.md.
#include <torch/extension.h>
#include "weighted_distance.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("weightedDistance", &weightedDistanceCUDA);
}
