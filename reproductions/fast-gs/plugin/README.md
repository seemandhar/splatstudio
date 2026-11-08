# splatstudio-method-fast-gs

FastGS: Training 3D Gaussian Splatting in 100 Seconds (CVPR 2026, arXiv 2511.04283), native
SplatStudio port of the official code https://github.com/fastgs/FastGS @ 44e02a5.
Paper fidelity unverified.

Methods: `fast-gs` (main row, densify every 500), `fast-gs-big` (every 100), `-tdb` variants with
the T&T / Deep Blending Compact Box factor 0.7.

Mechanism (see `../paper.md`, `../plan.md`):
- renderer: the official FastGS CUDA rasterizer, vendored unmodified (`ss_fast_gs/csrc`,
  Inria licence) — Compact Box tile assignment (mult/beta), 4-column screen gradient (signed +
  AbsGS), per-Gaussian high-error-pixel counter. Black background as in the official code.
- density: VCD (clone/split gated by floor-mean high-error-pixel count over 10 views > 5),
  VCP (score-sampled pruning before 15k, o<0.1 or s_p>0.9 at 18k/21k/24k/27k), opacity clamp
  0.8, opacity reset 0.01 every 3000, 3DGS-accel optimizer schedule (`optim.py`), per-scene
  official arguments looked up by SfM point count (`scenes.py`).
- loss: L1 + 0.2 D-SSIM with same-padded SSIM (fused_ssim).

Fidelity gaps: runtime samples views with replacement; densification runs after the optimizer
step; score views come from a cache of views seen so far; evaluation metrics are the runtime's.

```bash
pip install -e .            # first CUDA use JIT-builds the rasterizer (~2-3 min)
ss-verify fast-gs --device cuda
ss-train fast-gs --data ~/datasets/mipnerf360/w1600/garden --output runs/x --recipe 3dgs \
  --num-gaussians 10000000 --max-gaussians 12000000
```
