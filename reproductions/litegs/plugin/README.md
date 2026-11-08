# splatstudio-method-litegs

Native SplatStudio port of the **algorithmic** parts of
*LiteGS: A High-performance Framework to Train 3DGS in Subminutes via System and Algorithm
Codesign* (arXiv [2503.01199](https://arxiv.org/abs/2503.01199)), following the official code
https://github.com/MooreThreads/LiteGS (Gaussian-Splatting licence, non-commercial research).
Paper fidelity unverified.

Method `litegs` = **LiteGS-quality**: per-scene target primitive count = 3DGS count (official
`full_eval.py`), 30K steps. Targets are looked up by the scene's SfM point count; set the density
config `target_primitives` for other scenes.

## Mechanism
- **Opacity-gradient-variance densification** (Sec. 3.3): CUDA kernel `csrc/litegs_stats.cu`
  re-traverses gsplat's tile lists and accumulates, per Gaussian, S = Σ(∂L/∂o)², M = Σ ∂L/∂o over
  all pixel fragments, the fragment count C and Σ T·α, during each densification epoch.
  Score = clamp(S/(C+1) − (M/(C+1))², 0) · C · σ(o)² (official `get_var`/`get_score`).
- **Budgeted growth** (official `DensityControllerTamingGS`): linear target growth from the SfM
  count to the target over epochs [3, until); `budget` candidates drawn without replacement ∝ score;
  clone if max scale ≤ 0.01·extent, else one split child (offset R·N(0,s), scale s/1.6, parent kept).
  Weight prune of primitives with Σ T·α = 0 in the statistics epoch.
- **Opacity decay** every 10 epochs: σ(o) ← max(σ(o)/2, 1/128), all Adam states cleared.
- **Epoch schedule**: densify every 5 epochs from epoch 3 until int(0.8·E/10)·10+1; SH degree
  min(epoch//5, 3).
- **Optimizer**: `LiteAdam` — visible-only, no bias correction, eps 1e-15; lrs means 1.6e-4·extent
  (log-decay to 1/100), sh0 2.5e-3, shN 2.5e-4, opacity 0.025, scale 5e-3, rotation 1e-3.
- Training image clamped to [0,1] before L1 + 0.2 D-SSIM.

## Not ported (speed only)
Warp-based fp16 rasteriser with scanline evaluation and Morton-sorted cluster-cull-compact
(paper Table 2: ≤0.05 dB effect). gsplat renders.

## Fidelity gaps
See `../plan.md` (per-primitive vs per-cluster Adam visibility; exact per-pixel squares vs the
official kernel's running-sum quirk; runtime view sampling with replacement, white background,
SSIM valid convolution; append truncation to multiples of 128 not ported).

## Commands
```bash
pip install -e .
ss-inspect litegs
ss-verify litegs --device cuda
ss-train litegs --data ~/datasets/mipnerf360/native/garden --output runs/x --steps 30000 \
  --recipe 3dgs --num-gaussians 10000000 --max-gaussians 12000000
```
