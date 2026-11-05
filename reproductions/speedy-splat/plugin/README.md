# splatstudio-method-speedy-splat

Speedy-Splat — *Fast 3D Gaussian Splatting with Sparse Pixels and Sparse Primitives* (CVPR 2025,
arXiv 2412.00578). Native SplatStudio port of https://github.com/j-alex-hanson/speedy-splat
(commit 34c45c6). Paper fidelity unverified.

## Mechanism
- `ss_speedysplat/csrc/speedy/`: official rasterizer (speedy-splat-rasterizer b38abb6),
  **unmodified**: SnugBox + AccuTile tile assignment (`cuda_rasterizer/auxiliary.h`), the
  per-Gaussian score buffer `dL_dG2 += (o·dL/dα)²` (`backward.cu`), CUDA-event forward timer.
- `ss_speedysplat/csrc/stock/`: the same repository at 59f5f77 (graphdeco-inria
  diff-gaussian-rasterization as used by 3D-GS), unmodified: 3σ-square tiles. Used by the
  `speedy-splat-stock` control, the culling probe and `fps.py`.
- Both: Gaussian-Splatting licence (Inria/MPII, non-commercial research use,
  `csrc/*/LICENSE.md`); glm headers from upstream glm. JIT-built on first use with
  `torch.utils.cpp_extension.load` (run through `tools/ssenv`; ~2 min each).
- Renderer: official `gaussian_renderer.render`, black background.
- Density: official 3D-GS ADC (clone/split at 2e-4, every 100 in (500, 15000), opacity prune
  0.005, world-size prune 0.1·extent after 3000, screen-size prune no-op as in the Inria code,
  opacity reset 0.01 every 3000) + Speedy-Splat pruning: score = Σ_train views Σ_pixels
  (o·∂ΣI/∂α)² (train.py `prune`), prune `score ≤ sort[int(r(N-1))]`; soft r = 0.8 at
  6000/9000/12000 (after densify, before the opacity reset), hard r = 0.3 at 15000…27000.

## Methods
- `speedy-splat`: full method (paper row "+Hard Pruning").
- `speedy-splat-accutile`: culling rasterizer, no pruning (paper row "+AccuTile").
- `speedy-splat-stock`: stock rasterizer, no pruning (paper "Baseline"; control).

## Config defaults (official)
Renderer: `rasterizer` speedy|stock, `background` [0,0,0].
Density: densify_from 500, densify_until 15000, densify_every 100, grad_threshold 2e-4,
percent_dense 0.01, min_opacity 0.005, big_world 0.1 after 3000, reset_every 3000 /
reset_value 0.01, prune true, prune_from 6000, prune_until 30000, prune_every 3000,
soft_ratio 0.8, hard_ratio 0.3, score_background [0,0,0].

## Probes
- `speedy-culling-exact` (CUDA): culled image vs a float64 dense compositor that evaluates every
  Gaussian at every pixel (max abs err 1.2e-6; the stock 3σ rasterizer differs by 1.2e-2 on the
  same fixture); tile-Gaussian pairs 123 = independent exact ellipse∩tile count 123, vs 198
  for 3σ squares (= independent getRect count).
- `speedy-score-kernel` (CUDA): density score over 2 views vs Σ_p (o·∂ΣC/∂α)² from float64
  autograd (rel err 1.1e-6; summing before squaring would differ by 174×).
- `speedy-prune-schedule`, `speedy-adc-schedule` (CPU): schedule, percentile rule, ordering,
  optimizer alignment.

## Speed (secondary)
`tools/ssenv python -m ss_speedysplat.fps --run <run>` times the forward rasterizer (CUDA
events) of the test views with both rasterizers on the same checkpoint and GPU, and reports
tile-Gaussian pairs. Shared A6000s: only ratios are meaningful, not comparable to the paper's
A5000 FPS.

## Fidelity gaps
- Runtime applies the optimizer step before density control (official: densify / soft prune
  before the step, hard prune after it).
- View sampling with replacement, SSIM valid convolution, SH degree increments one step later
  (the score render uses the runtime's degree at that step).
- One seed (paper: mean of 3 runs).
- Every speedy forward synchronises for its CUDA-event timer (official behaviour).

## Commands
```bash
cd plugin && ../../../tools/ssenv pip install -e .
../../../tools/ssenv ss-verify speedy-splat --device cuda
../../../tools/ssenv ss-train speedy-splat --data ~/datasets/mipnerf360/w1600/garden \
  --output run --steps 30000 --recipe 3dgs --num-gaussians 10000000 --max-gaussians 12000000
```
