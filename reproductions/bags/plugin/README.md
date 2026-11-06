# splatstudio-method-bags

BAGS: Blur Agnostic Gaussian Splatting through Multi-Scale Kernel Modeling (Peng et al., ECCV
2024, arXiv 2403.04926), a native SplatStudio port of the official code
(github.com/snldmt/BAGS @ed04144, built on Mip-Splatting). No custom CUDA.

Mechanism
- `bags-raster`: gsplat Mip-Splatting render (3D filter, 2D Mip filter eps 0.1, black bg) of the
  sharp image C and the alpha-blended Euclidean-distance depth D. Training views are rendered at
  1/4 (step < 3000), 1/2 (< 6000), then full resolution; test views are always full-res sharp.
- `ss_bags.bank.BlurBank` (owned by `bags-density`, `ss_bags.bank.ACTIVE`): multi-scale Blur
  Proposal Network (view embedding 32, 2-D PE 16, RGB-D CNN 64/64/32, 1x1 MLP 64 no bias, heads
  K = 5/9/17 + masks; optional single-scale net for `use_another_mlp`), Adam lr 5e-4 eps 1e-15.
- `bags-objective`: C_out = m (h (*) C) + (1-m) C (Unfold, zero pad), 0.8 L1 + 0.2 D-SSIM vs the
  observed image resized to the render size; sharp-image loss while phase iteration <= 250.
  `bags-mask` 0.001 mean(m); `bags-depth-tv`, `bags-rgb-tv` per official script.
- `bags-density`: official two-phase clock (iteration restarts after ms_steps: xyz LR,
  densify 500..35000/100, reset /3000), init_dgt -> 2e-4, prune sigmoid(o) < 0.1, Mip 3D filter
  from the current-scale cameras, nothing applied at the final step.

Methods: `bags-<scene>` = official scripts/<blur|defocus><scene>.sh (20 Deblur-NeRF scenes,
`ss_bags/variants.py`); `bags` = official argument defaults; `bags-smoke` = compressed schedule
for smoke tests only. Run with `--steps` equal to the method's `total_steps`.

```bash
tools/ssenv ss-train bags-ball --data ~/datasets/deblurnerf/native-motion/ball --output RUN \
  --steps 49000 --recipe 3dgs --num-gaussians 10000000 --max-gaussians 12000000
tools/ssenv python -m ss_bags.evaluate --run RUN     # official Deblur-NeRF metrics -> RUN/test_official.json
```

Fidelity gaps: coarse targets resized from the stored 600x400 image (official: PIL from
2400x1600); runtime Adam step precedes densification; SH degree +1 step late; skimage SSIM
re-implemented (skimage not installed). See ../report.md.
