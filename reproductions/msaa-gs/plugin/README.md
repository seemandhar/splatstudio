# splatstudio-method-msaa-gs

Native SplatStudio port of *Multi-Sample Anti-Aliasing and Constrained Optimization for 3D
Gaussian Splatting* (arXiv 2508.10507). No official code exists; implemented from the paper.

Mechanism
- **MSAA (C1)**: custom CUDA rasterizer (`ss_msaa_gs/csrc/msaa_raster.cu`, JIT-built, adapted
  from gsplat 1.5.3 RasterizeToPixels3DGS{Fwd,Bwd}.cu, Apache-2.0). Each pixel composites 4
  sub-samples independently at the standard rotated-grid positions (0.375,0.125), (0.875,0.375),
  (0.125,0.625), (0.625,0.875) of the pixel square, and averages them (Eq. 4-5). Train + test.
- **AWS (C2)**: 0.8 · mean(w·|e|) with w = α + (1-α) e/(max e + ε), α 0.5, ε 1e-6 (α, ε not in
  the paper), + 0.2 D-SSIM (Eq. 6-9).
- **GDC (C3)**: 0.1 · (mean|∇x diff| + mean|∇y diff|), forward differences (Eq. 10-13).
- Density/LRs: 3DGS (gsplat DefaultStrategy with the runtime's opacity reset).

Methods: `msaa-gs`, ablations `msaa-gs-msaa-only`, `msaa-gs-aws-gdc-only`.
Fidelity gaps: see `../plan.md` and `../paper.md` §7.

    tools/ssenv ss-train msaa-gs --data ~/datasets/mipnerf360/native/garden --recipe 3dgs ...
