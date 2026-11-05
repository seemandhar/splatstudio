# splatstudio-method-steepgs

SteepGS — *Steepest Descent Density Control for Compact 3D Gaussian Splatting* (CVPR 2025,
arXiv 2505.05587). Native SplatStudio port of https://github.com/facebookresearch/SteepGS
(commit 63eee45, `train.py --densify_strategy steepest --S_estimator inv_cov`).
Paper fidelity unverified.

## Mechanism
- `ss_steepgs/csrc/`: the official SteepGS `diff-gaussian-rasterization` (Inria 3DGS rasterizer +
  splitting-matrix accumulation in `backward.cu`), **unmodified**, Gaussian-Splatting licence
  (non-commercial research; `csrc/LICENSE.md`, `csrc/LICENSE-SteepGS.md`); glm headers from
  upstream glm. JIT-built on first use (`torch.utils.cpp_extension.load`, run via `tools/ssenv`).
- Renderer (`renderer.py`/`raster.py`): official `render()`; black background; returns the
  Inria NDC screen gradient and the per-Gaussian `sum_pix dL/dG` buffer.
- Density (`density.py`, `core.py`): `add_densification_stats` (S += c Σ⁻¹, inv_cov) and
  `densify_and_split_steepest`: select = mean screen grad ≥ 2e-4 ∧ λ_min(S̄) < −1e-6; split
  (max s > 0.01·extent) into 2 offspring at p ± ‖n‖ v_min (n ~ N(0, diag s), official sign
  pattern of `repeat(2,1)` rows), scale s/1.6, opacity copied; clone: copy and original at
  p + lr·v_min; prune o < 0.005 and (after 3000) max s > 0.1·extent; opacity reset 0.01 every
  3000 until 15000; densify every 100 in (500, 15000).

## Config (density, defaults = official)
densify_from 500, densify_until 15000, densify_every 100, grad_threshold 2e-4,
S_threshold −1e-6, min_grad 1e-4, percent_dense 0.01, min_opacity 0.005, big_world 0.1 after
3000, reset_every 3000 / reset_value 0.01, options [] (official attributes `stationary`,
`no_uncertain`, `no_eig_cond`, `no_saddle`, `no_eig_upd`, `zero_upd`, `no_div_opacity`),
strategy `steepest` (`adc` = official 3DGS branch).

Methods: `steepgs`; `steepgs-adc` (control: same rasterizer/code base, official ADC branch).

## Fidelity gaps
- Runtime applies the optimizer step before density control (official: densify, then step).
- View sampling with replacement, SSIM valid convolution, SH degree increments one step later.
- Opacity halving of the paper (Thm. 2 iii) is off — official default (`no_div_opacity`
  enables it). Split sign quirk and clone placement follow the official code.
- Only the `inv_cov` splitting estimator post-processing is ported (README default).

## Commands
```bash
cd plugin && ../../../tools/ssenv pip install -e .
tools/ssenv ss-verify steepgs --device cuda
tools/ssenv ss-train steepgs --data ~/datasets/mipnerf360/native/garden --output runs/x \
  --recipe 3dgs --num-gaussians 10000000 --steps 30000
```
