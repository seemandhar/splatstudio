# splatstudio-method-mvgs

Native SplatStudio port of **MVGS: Multi-view Regulated Gaussian Splatting for Novel View
Synthesis** (Du, Wang, Yu; [arXiv 2410.02103](https://arxiv.org/abs/2410.02103)), 3DGS variant,
matched to the official code https://github.com/xiaobiaodu/MVGS @ 36d073a (Inria 3DGS fork).
Paper fidelity unverified.

| method | recipe (official `run_360.py`) | steps |
|---|---|---|
| `mvgs` | indoor: CIG stages M=48 @ -r 8, 24 @ -r 4, 12 @ -r 2 (3000 it each), M=8 @ w1600 (30000 it) | 39000 |
| `mvgs-outdoor` | bicycle/garden/stump: M=3 @ w1600 | 30000 |

## Mechanism → code
- **Multi-view regulated learning** (Eq. 4): `mvgs-density` reloads the training images
  (ss-train `--data`) and samples M distinct views per step from a shuffled stack (official
  sampler); `mvgs-renderer` renders all of them (the runtime's single frame is ignored during
  training); `photometric` sums (not averages) the 3DGS losses (L1 + 0.2 D-SSIM, same padding).
- **Cross-intrinsic guidance** (Eq. 9): per-stage render size `round(W·ratio/r)` with K scaled
  (FoV kept), targets antialiased-bicubic resized; stage-local iteration counter restarts the
  densification window (500–15000), opacity resets (it%3000, it=500) and position LR, as the
  official checkpoint chain does. `full_width_ratio` = 3115/1600 (indoor Mip-NeRF 360).
- **Multi-view augmented densification** (Eq. 10): threshold 2e-4 → 1e-4 if any pair of the
  step's unit-normalised camera centres is > τ = 1 apart (code form; translation only).
- **Cross-ray densification** (supp. Alg. 1): max-mean L1 20×40 window per view, views sorted by
  that loss, 4 corner rays per window, pairwise closest-point midpoints → AABB cuboids; every
  Gaussian strictly inside a cuboid is cloned (plus the usual gradient clone), then split and
  prune. `box_index: official` keeps the official `boxes[j]` indexing quirk.
- Densification statistics: last rendered view only (official); `stats_views: all` for the ablation.

## Fidelity gaps
gsplat rasterizer (shared with baseline; saturated-alpha gradient drop); Adam step before density
control on densify steps (runtime order, shared); coarse targets resized from the w1600 images
instead of PIL from full-res (±1 px height); one continuous run instead of 4 processes (carried
state identical); no official T&T/DB config.

## Commands
```bash
pip install -e .
ss-verify mvgs --device cuda
ss-train mvgs --data ~/datasets/mipnerf360/w1600/counter --output runs/x --steps 39000 \
  --recipe 3dgs --num-gaussians 10000000 --max-gaussians 12000000
ss-train mvgs-outdoor --data ~/datasets/mipnerf360/w1600/garden --output runs/y --steps 30000 \
  --recipe 3dgs --num-gaussians 10000000 --max-gaussians 12000000
```
