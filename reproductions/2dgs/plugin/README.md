# splatstudio-method-2dgs

Native SplatStudio port of **2D Gaussian Splatting for Geometrically Accurate Radiance Fields**
(Huang et al., SIGGRAPH 2024, arXiv 2403.17888), written against the official code
github.com/hbb1/2d-gaussian-splatting @f3e3b9f. The surfel rasteriser
github.com/hbb1/diff-surfel-rasterization @e0ed020 is vendored **unmodified** in `ss_twodgs/csrc/`
(Inria/MPII Gaussian-Splatting licence, non-commercial research use; `csrc/SHA256SUMS`; glm from
reproductions/pgsr) and JIT-built with `torch.utils.cpp_extension` on first use.

Methods: `2dgs` (released-code M360 settings, lambda_dist 0), `2dgs-paperdist` (lambda_dist 100, paper §5).

| component | what | values (source) |
|---|---|---|
| `2dgs-surfel` renderer | official ray-splat CUDA, low-pass rho=min(rho3d, 2|x-c|^2) with c = projected bbox centre, alpha<=0.99, per-pixel near 0.2, black bg | depth_ratio 0 (code) |
| `2dgs-density` | 3DGS control on the projected 3D-centre grad, in-plane split, cull 0.05, reset 0.01/3000, world prune 0.1 extent after 3000, quats torch.rand init | gaussian_model.py, train.py |
| `photometric` | 0.8 L1 + 0.2 D-SSIM | |
| `2dgs-distortion` | lambda_dist * mean(dist), step > 3000; dist = sum w_i w_j (m_i-m_j)^2, NDC m | 0 code / 100 paper |
| `2dgs-normal` | 0.05 * mean(1 - <sum w n, N(depth) alpha>), step > 7000 | train.py |

Why not the built-in `gsplat-surfel`: gsplat 1.5.3 uses per-Gaussian centre depth (not intersection
depth) in depth/median/distortion, an L1 raw-depth distortion, alpha clamp 0.999, a different
densification statistic, and its built-in normal loss normalises the rendered normal.

Fidelity gaps: optimiser step before densification (runtime order); random view sampling with
replacement; SSIM valid convolution; released code differs from the paper code (README: low-pass
gradient removed from densification -> slightly lower PSNR); lambda_dist code 0 vs paper 100;
metrics on float renders (official: 8-bit PNGs). Mesh extraction / DTU geometry not ported.

```bash
pip install -e . && ss-inspect 2dgs && ss-verify 2dgs --device cuda
ss-train 2dgs --data ~/datasets/mipnerf360/native/garden --output runs/x --steps 30000 --recipe 3dgs --num-gaussians 10000000
```
