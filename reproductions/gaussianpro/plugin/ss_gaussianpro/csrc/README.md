Vendored CUDA sources, unmodified:
- `propagation/`: github.com/kcheng1021/GaussianPro `submodules/Propagation` @ 86bb04848c35bf7a919da870046adfcf5bb67a33
  (MIT licence file `LICENSE-GaussianPro.md`; the README adds "for commercial use contact Xuejin Chen"; derived from ACMH/ACMM).
- `simple_knn/`: gitlab.inria.fr/bkerbl/simple-knn @ 86710c2d4b46680c02301765dd79e465819c8f19
  (Inria Gaussian-Splatting licence: `LICENSE-3DGS.md`).
Both are JIT-built by `ss_gaussianpro/kernels.py` (torch.utils.cpp_extension.load).
