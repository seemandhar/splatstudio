# splatstudio-method-3dhgs

Native SplatStudio port of **3D-HGS: 3D Half-Gaussian Splatting** (Li, Liu, Sznaier, Camps,
CVPR 2025, [arXiv 2406.02720](https://arxiv.org/abs/2406.02720)), following the official code
[lihaolin88/3DHGS](https://github.com/lihaolin88/3DHGS) @ dde1bd5. Paper fidelity unverified.

## Mechanism
Each primitive is split by a plane through its centre (learned normal `hgs_normal`) into two
halves with opacities α1 (`opacities`) and α2 (`opacity2`). Splatted alpha (Eq. 10):
`½(α1 G (1+erf a) + α2 G (1−erf a))`, `a` from the conditional z-statistics of the ray-space
covariance and the camera-space normal (Eq. 7–8).

## Rasterizer
`ss_3dhgs/csrc/` = the official `submodules/diff-gaussian-rasterization`, **unmodified**
(Gaussian-Splatting licence, non-commercial research use: `csrc/LICENSE.md`), JIT-built on first
use with `torch.utils.cpp_extension.load` (run through `tools/ssenv`). Training renders use the
official training path; evaluation renders (no autograd) use the official test path with the
half-aware tile rectangles (`test_time_cull`).

## Config (defaults = official code)
| key | value | source |
|---|---|---|
| normal lr / init | 0.003 / 0.6 | gaussian_model.training_setup / create_from_pcd |
| opacity lr (both halves) | 0.05, ÷1.4 every 5000 | train.py:128-132 |
| densify from/until/every | 500 / 20000 / 100 | arguments |
| min opacity (max of halves) | 0.01 | train.py:136 |
| opacity reset | 0.02 every 3000 | reset_opacity |
| clone opacity | 1 − √(1 − α) on the copy | densify_and_clone |
| background | black | train.py |
`3dhgs-db-paper` uses the paper's Deep Blending normal LR 0.3 (ignored by the official code).

## Fidelity gaps
Runtime order (optimizer step before density, LR decay one step later), SH degree switch one
step later, random-with-replacement sampling, SSIM valid convolution. Official quirks kept:
normal moved to camera space with the translation added, forward-only power < −4.8 skip
(backward mismatch for opacity ≳ 0.48 tails), no normal LR decay (paper says /1.4).

## Commands
```bash
cd plugin && ../../../tools/ssenv pip install -e . -q
tools/ssenv ss-inspect 3dhgs && tools/ssenv ss-verify 3dhgs --device cuda
tools/ssenv ss-train 3dhgs --data ~/datasets/mipnerf360/native/garden --output runs/x --recipe 3dgs --num-gaussians 10000000
```
