# CoR-GS (ECCV 2024, arXiv 2405.12110) — native SplatStudio reproduction

**Outcome: matched** (LLFF 3-view 8-scene mean PSNR −0.19 dB vs paper, within ±0.5 dB).
Plugin `reproductions/cor-gs/plugin` (methods `cor-gs`, `cor-gs-single`), ported from the official
code github.com/jiaw-z/CoR-GS (Inria GS licence, read-only reference). Not separately reviewed.

## Benchmark
Dataset `llff-fewview3-r8` (official `-r 8` from full-res images, 504×378; 3 train views by
linspace, every 8th test; 3-view COLMAP MVS init), 10 000 steps, `--recipe 3dgs`, one seed,
field 1 evaluated (paper keeps gs0). Runs: `runs/cor-gs-llff-r8`.

| scene | PSNR |
|---|---|
| fern | 22.77 |
| flower | 20.66 |
| fortress | 23.35 |
| horns | 19.07 |
| leaves | 16.57 |
| orchids | 15.71 |
| room | 21.89 |
| trex | 22.07 |
| **mean (ours)** | **20.26** / SSIM 0.703 / LPIPS 0.232 |
| paper (Table 1) | 20.45 / 0.712 / 0.196 |
| paper 3DGS row | 19.22 / 0.649 / 0.229 |

The paper reports averages only, so per-scene numbers cannot be compared. PSNR and SSIM are
close. **LPIPS is clearly worse (+0.036)** and is not within tolerance. The protocol-matched
baseline `cor-gs-single` was **not run**, so the gain over the authors' 3DGS row (+1.23 dB)
has not been checked in our pipeline.

## Mechanism evidence
ss-verify passes (co-pruning vs a brute-force Eq. 3 reference, co-reg vs an independent
L1/SSIM formula with detached targets, two independent fields, schedules). Smoke run (horns,
2k steps): co-pruning removed ~300–490 Gaussians per field every 500 steps, and the co-reg
term was ≈0.02–0.03.

## Deviations from the paper / official code
1. The runtime steps the optimizer before density control; official densifies first, and the
   densified tensors skip that step. Field 2 follows the runtime order to keep both fields symmetric.
2. Training views are sampled with replacement (official pops a shuffled stack).
3. gsplat vs the Inria rasterizer (opacity-aware radii; 0.999 alpha cap via straight-through).
4. Training SSIM uses valid convolution (official uses same padding). Eval uses runtime SSIM and
   the `lpips` package (VGG) on float renders; official uses 8-bit PNGs and lpipsPyTorch. This
   may explain part of the LPIPS gap.
5. Init cloud: COLMAP 3.9.1 MVS of the 3 views (authors' version unknown).
6. Native also steps the optimizer at the final iteration (official skips it); one seed per scene
   with different RNG streams than the official seed 42.
7. Co-pruning nearest neighbours use exact cdist instead of the open3d KD-tree (same result up to
   ties at exactly τ).
