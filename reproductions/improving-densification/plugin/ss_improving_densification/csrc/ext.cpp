/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#include <torch/extension.h>
#include "rasterize_points.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("rasterize_gaussians", &RasterizeGaussiansCUDA);
  m.def("minigs_rasterize_gaussians_aux", &MiniGSRasterizeGaussiansAuxCUDA);
  m.def("minigs_rasterize_gaussians_depth", &MiniGSRasterizeGaussiansDepthCUDA);
  m.def("rasterize_gaussians_backward", &RasterizeGaussiansBackwardCUDA);
  m.def("adamUpdateTorch", &adamUpdateTorch);
  m.def("compute_relocation", &ComputeRelocationCUDA);
}
