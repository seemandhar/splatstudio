Vendored from github.com/MooreThreads/StableGS @ 397a9c00cd322b769801e24b00968cf4f2e24001,
`submodules/diff-gaussian-rasterization_radegs` (the RaDe-GS rasterizer of
github.com/HKUST-SAIL/RaDe-GS @ ec9199f with `.data<>` -> `.data_ptr<>`, `W/2.` -> `W/2.f`, and an
added MCMC `compute_relocation` kernel in `cuda_rasterizer/utils.cu`). Unmodified, except that
`setup.py` and `CMakeLists.txt` were removed (the plugin JIT-builds with
torch.utils.cpp_extension.load). Licence: Gaussian-Splatting licence (Inria / MPII), non-commercial
research and evaluation use, see LICENSE.md.
