# splatstudio-method-aaa-gaussians

Native SplatStudio port of **AAA-Gaussians: Anti-Aliased and Artifact-Free 3D Gaussian
Rendering** (Steiner, Köhler et al., ICCV 2025, arXiv 2504.12811). It is paper fidelity unverified.

The renderer is the official CUDA rasterizer, github.com/DerThomy/AAA-Gaussians-Rasterization @ d03ce2d, vendored unmodified in
`ss_aaa_gaussians/csrc/` and JIT-built (about 75 s).
- It runs the StopThePop hierarchical sorter.
- Gaussians are evaluated in 3D through screen-space planes.
- It applies the adaptive 3D filter with perpendicular amplitude scaling.
- Bounds are view-space angle bounds.
- Culling is 3D frustum and per-tile culling.

Licences: the Gaussian-Splatting licence (non-commercial research use). The StopThePop headers are MIT.

## Components (method `aaa-gaussians`)

| component | what | config (default = official value) |
|---|---|---|
| `aaa-renderer` | vendored kernel, used for training and test | `splatting` = `configs/aaa.json` (HIER, PTD_MAX, queues 64/8/4, rect/tight/tile/4x4 culling, load balancing, proper_ewa_scaling, eval_3D, new_aabb); black background |
| `aaa-mcmc-density` | gsplat `MCMCStrategy` (relocate, add 5 %, noise) | cap from the `--data` basename (`full_eval.py` caps); refine for 500 < s < 25000 every 100 steps; noise_lr 5e5; init opacity 0.5; init scale ×0.1 |
| `photometric` | L1 + 0.2 D-SSIM | |
| `aaa-mcmc-reg` | 0.01·mean σ(o) + 0.01·mean exp(s) | |
| field `filter_3d` | training-view filter std | |

The `filter_3d` field stays 0 during training, so the kernel uses only the per-view filter, as the official code does with `filter_3D = None`. At the last step it is set to the value from `mip_filter.py`, so test renders use max(filter², k z²/f²) with k = 0.3.

## Verification

`ss-verify aaa-gaussians --device cuda` checks the following:
- Forward and backward match a float64 dense torch reference (`reference.py`) to about 1e-6. The volume-scaling, no-scaling and 2D-EWA variants are clearly rejected.
- Perpendicular scaling is checked against volume scaling on a thin disk.
- The adaptive filter and its training-view floor are checked.
- View-space bounding is checked on Gaussians whose mean lies behind the camera or outside the view.
- Frustum and tile culling are checked.
- The filter computation, the MCMC schedule and init, and the regulariser are checked.

## Commands
```bash
tools/ssenv ss-train aaa-gaussians --data ~/datasets/mipnerf360/native/garden --output runs/x \
  --steps 30000 --recipe 3dgs --num-gaussians 10000000 --max-gaussians 12000000
tools/ssenv ss-eval --run runs/x --split test
```

## Fidelity gaps

See `../plan.md`. In short:
- The runtime takes the optimizer step before density control.
- Views are sampled with replacement. SSIM uses valid convolution. LPIPS comes from the `lpips` package. The SH degree steps up one step late.
- The cap is looked up from the `--data` basename.
- The popping, FLIP, large-FOV and multi-resolution evaluations are not run.
