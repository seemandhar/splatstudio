# DropGaussian reproduction report

**Paper:** *DropGaussian: Structural Regularization for Sparse-view Gaussian Splatting*, CVPR 2025, arXiv 2504.00773.

**Official code:** DCVL-3D/DropGaussian_release, read only (never run).

**Methods:**
- `dropgaussian`: the paper's method.
- `dropgaussian-nodrop`: the same pipeline with γ = 0. This is the first row of the paper's Table 5 and serves as the matched 3DGS baseline.
- The plugin is in `plugin/` (`ss_dropgaussian`).

**Outcome: matched after 1 iteration.**
- **Absolute PSNR:** the 8-scene mean on LLFF 3-view is **20.36 vs. the paper's 20.76 (−0.40 dB)**. SSIM is 0.705 vs. 0.713.
- **Gain from dropout:** smaller than the paper's. Ours is +0.52 dB over our no-drop control; the paper reports +1.54 dB over its 3DGS row. Our control is 0.62 dB stronger than the paper's 3DGS number (19.84 vs. 19.22).

## Results
Setup:
- 10k steps on `llff-fewview3-r8`, using the FSGS 3-view split with the official `-r 8` 504×378 images.
- Initialisation is the 3-view MVS fused cloud.
- Test views are every 8th image.
- The paper reports only averages, so the full 8-scene mean is the comparison.

| run | fern | flower | fortress | horns | leaves | orchids | room | trex | **mean PSNR** | SSIM | LPIPS |
|---|---|---|---|---|---|---|---|---|---|---|---|
| iter 1, dropgaussian | 22.73 | 20.82 | 22.38 | 18.78 | 17.56 | 16.03 | 22.31 | 22.24 | **20.36** | 0.705 | 0.232 |
| iter 1, nodrop | 21.92 | 20.11 | 22.07 | 18.78 | 17.07 | 15.37 | 21.45 | 21.95 | **19.84** | 0.682 | 0.241 |
| paper DropGaussian | | | | | | | | | **20.76** | 0.713 | 0.200 |
| paper 3DGS | | | | | | | | | **19.22** | 0.649 | 0.229 |
| iter 0, dropgaussian (kept) | 21.31 | 20.60 | 17.12 | 18.34 | 17.62 | 15.75 | 15.81 | 14.49 | 17.63 | 0.635 | 0.314 |
| iter 0, nodrop (kept) | 21.86 | 20.24 | 21.76 | 18.47 | 17.12 | 15.25 | 21.50 | 21.70 | 19.74 | 0.682 | 0.239 |

Runs: `runs/dropgaussian-llff-r8/` (iteration 0) and `runs/dropgaussian-i1-llff-r8/` (iteration 1).

**Reading the gain gap.**
- Our no-drop control already scores 19.84, close to the paper's DropGaussian. It shares the official training setup: τ = 5e-4, no size pruning, black background, clamped renders, and the MVS init.
- The paper's "3DGS" row may come from a different 3DGS configuration. The paper's own Tables 1 and 5 disagree by 0.2 dB for the default setting (plan gap 5).
- Because the absolute number matches within tolerance, the outcome is "matched". The smaller relative gain is recorded here rather than hidden.

## Iteration 1: what was wrong in iteration 0 and the fix
**Symptom.** In iteration 0, fortress, room and trex collapsed to 14–17 dB. Trex also collapsed with seeds 1 and 2 (14.4 and 15.2).

Short trex runs located the onset:

| steps | dropgaussian PSNR | nodrop PSNR |
|---|---|---|
| 1.5k–5.9k | 20.9–21.6 | — |
| 8000 | 15.2 | 22.6 |

The collapses followed the opacity resets at 6000 and 9000. Training views still fit (33 dB), and the error was large, streaky floaters seen from novel views. Culling large Gaussians and changing the near plane had no effect.

**Cause: a rasterizer difference.**
- gsplat's backward pass (`RasterizeToPixels3DGSBwd.cu:222`) computes opacity, conic and means2d gradients only where `opac·G ≤ 0.999`.
- The official Inria rasterizer clamps alpha only in the forward pass and always back-propagates.
- DropGaussian's compensation multiplies kept opacities by 1/(1−r), up to 1.25×, so many Gaussians saturate. Under gsplat they then receive no gradient at the pixels where they are most visible, and they turn into floaters after each opacity reset.

**Fix** (`renderer.py`, `st_alpha_clamp`, applied to both methods):
- The opacity passed to gsplat is capped at 0.999 with a straight-through gradient, which reproduces the official gradient flow.
- The forward pass is unchanged except on pixels where G lies in (0.999/o, 1).
- No hyperparameters changed.

**Effect on trex:** 14.49 → 22.24 with dropout, and 21.70 → 21.95 without.

**Honesty note.** The diagnosis used trex test-view PSNR, on one scene. The fix matches the behaviour of the official rasterizer and is not tuned. The same gsplat behaviour is recorded in `TODO-runtime.md` as a runtime-wide issue.

## Fidelity gaps (kept)
1. Views are sampled with replacement. The official code pops a shuffled stack.
2. The Adam step happens before densification. The official code densifies first.
3. Remaining rasterizer differences:
   - The alpha clamp is 0.999 here vs. 0.99 officially.
   - gsplat uses opacity-aware radii.
   - The straight-through cap imitates the Inria gradient but is not identical.
4. LPIPS uses the `lpips` package. The official code uses lpipsPyTorch.
5. One seed per scene. Few-view results vary noticeably between seeds.
6. Initialisation uses the COLMAP 3.9.1 MVS cloud. The authors' COLMAP version is not stated.

## Reproduce
```bash
python3 tools/bench.py launch --exp dropgaussian-i1-llff-r8 --methods dropgaussian dropgaussian-nodrop --dataset llff-fewview3-r8 --steps 10000 --per-gpu 2
```
