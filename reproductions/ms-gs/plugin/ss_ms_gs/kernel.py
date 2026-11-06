"""JIT-built CUDA kernel fusing pixel coverage and the selection rule (csrc/coverage.cu).

CUDA tensors use the kernel; CPU tensors use the torch reference in coverage.py (the probe
`msgs-cuda-kernel` checks both agree)."""

from pathlib import Path

import torch

_EXT = None
CPP = ("std::vector<torch::Tensor> coverage_select(torch::Tensor, torch::Tensor, torch::Tensor, "
       "torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, double, double, double, double, "
       "double, double, bool, bool, double, double, double);")


def extension():
    global _EXT
    if _EXT is None:
        from torch.utils.cpp_extension import load_inline

        src = (Path(__file__).parent / "csrc" / "coverage.cu").read_text()
        _EXT = load_inline("ss_ms_gs_coverage", cpp_sources=CPP, cuda_sources=src,
                           functions=["coverage_select"], extra_cuda_cflags=["-O3"], verbose=False)
    return _EXT


@torch.no_grad()
def coverage_select_cuda(params, camera, cfg, near=0.2):
    f = lambda t: t.detach().float().contiguous()  # noqa: E731
    K = camera["K"]
    fx, fy = float(K[0, 0]), float(K[1, 1])
    W, H = int(camera["width"]), int(camera["height"])
    return extension().coverage_select(
        f(params["means"]), f(params["quats"]), f(params["scales"]), f(params["opacities"]),
        f(params["ms_max_px"]), f(params["ms_min_px"]), f(camera["viewmat"]),
        fx, fy, W / (2 * fx), H / (2 * fy), float(cfg["opacity_level"]), near,
        bool(cfg["filter_small"]), bool(cfg["filter_large"]), float(cfg["min_rel"]),
        float(cfg["max_rel"]), float(cfg["abs_threshold"]))
