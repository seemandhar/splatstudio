# Plan — 3DLS (linear kernels) as a native SplatStudio plugin

Method `3dls` (+ ablation `3dls-noags` = LK+DA), module `ss_linear_splatting`,
package `splatstudio-method-3dls` in `plugin/`. No official code; all from the paper.

## Components
| component | kind | factory | requires → provides | config (paper value) |
|---|---|---|---|---|
| linear-renderer | renderer | `ss_linear_splatting.renderer:create_renderer` | state.gaussian3d, camera.pinhole → render.rgb, render.depth.expected_z, gradient.screen.signed | lam 2.5 (C2), ags true (C3), tile_size 16, eps2d 0.3 |
| adaptive-density | density | `splatstudio.backends.gsplat:create_density` (wraps gsplat DefaultStrategy + 3DGS opacity-reset fix) | gradient.screen.signed | grow_grad2d 2e-4, grow_scale3d **0.006**, prune_scale3d **0.4**, prune_opa **0.025** (supp. Table 6) |
| l1-l2-ssim | loss | `ss_linear_splatting.losses:create_loss` (replaces `photometric`) | render.rgb | l1 0.6, l2 0.2, ssim 0.2 |

No new fields. Density unchanged apart from thresholds (reset value 2·prune_opa = 0.05 as gsplat).

## Renderer (custom CUDA, JIT `torch.utils.cpp_extension.load`)
Pipeline = gsplat 1.5.3 `rasterization` classic mode, step by step:
1. `fully_fused_projection(..., opacities=None)` (extend 3.33σ ⊃ linear support 2.5σ, so no
   Gaussian visible under the linear kernel is culled), eps2d 0.3, near 0.01, far 1e10.
2. **Bounding radius from λ**: support D<λ and α≥1/255 ⇒ D ≤ λ(1−(1/255)/o);
   radius_{x,y} = ceil(extend·sqrt(Σ2D_xx/yy)) with Σ2D = conic⁻¹; cull o<1/255 and boxes
   outside the image (gsplat's rule). Non-differentiable.
3. SH colours (gsplat `spherical_harmonics`, +0.5, clamp ≥0), cat depth for RGB+ED.
4. `isect_tiles` / `isect_offset_encode` (gsplat).
5. `csrc/linear_raster.cu`: forward/backward adapted from gsplat 1.5.3
   `RasterizeToPixels3DGSFwd.cu` / `Bwd.cu` (Apache-2.0, cited), only the falloff changed:
   sigma=½dᵀconic d, D=sqrt(2 sigma), k=1−D/λ, α=min(0.999, o·k), skip k≤0 or α<1/255,
   stop when T≤1e-4. Backward: v_o = k·v_α, v_sigma = −o·v_α/(λ·D) · ω, ω=exp(−D²) if AGS
   (Eq. 13–14), D clamped ≥1e-6; v_conic/v_xy from v_sigma exactly as gsplat; gsplat's
   `o·k ≤ 0.999` gate kept. CDIM 3 and 4 instantiated.
6. Output RGB over white (+(1−α)), depth = acc_z/α, info = gsplat-like meta (means2d with
   grad for DefaultStrategy, radii [1,N,2], width/height/n_cameras).

## Probes (each fails if its mechanism is removed)
| probe | contribution | measures |
|---|---|---|
| linear-forward-vs-dense | C1 | CUDA render (RGB, alpha) == dense float64 torch reference with the linear kernel; and differs from gsplat's Gaussian render by >X (fails if kernel reverts) |
| linear-backward-vs-dense | C1 | ags=False grads (means2d, conics, colors, opacities) == autograd of the dense reference |
| alignment-footprint | C2 | single Gaussian: rendered support radius = λ·σ (±1 px) for λ=2.5 and λ=1; config honoured; tile lists cover every reference pixel with α≥1/255 (no clipped footprint) |
| ags-gradient-scaling | C3 | ags=True geometry grads == reference with ∂k/∂D scaled by exp(−D²) (straight-through), opacity/colour grads == unscaled; means2d grads differ from ags=False |
| loss-weights | recipe | loss == 0.6·L1 + 0.2·MSE + 0.2·(1−SSIM), independent formula; zeroing a term detected |
| density-thresholds | recipe | built density's DefaultStrategy has 0.006/0.4/0.025/2e-4 |
| + base probes | | state-transitions, renderer-gradient (means2d grad present) |

## Fidelity gaps
1. **No official code**: kernel radius, AGS scope (geometry path only), saturate gating are
   our reading; expected effect small (radius only affects speed).
2. gsplat v1.3 vs 1.5.3 runtime (projection/densify near-identical; 1.5.3 per-axis radii).
3. Runtime protocol: white background (gsplat trainer: black), random-with-replacement view
   sampling, SSIM valid conv in loss/eval (gsplat uses fused-ssim "same"), our camera-extent
   scene scale. Calibration: our gsplat baseline garden 26.89 vs paper's gsplat-3DGS 27.41
   (−0.52), kitchen 31.34 vs 31.40 → compare improvements over baseline (paper: garden −0.01,
   kitchen +0.38).
4. 3DLS+AA variant not implemented (not the target).

## Benchmark
Per-scene numbers exist (supp. Table 7). mipnerf360: garden (outdoor, 27.40) + kitchen
(indoor, 31.78; also the ablation scene), 30k, vs `baseline3dgs-mipnerf360-30000`.

## Post-smoke amendment (2026-11-07)
Fig. 3b confirms the literal ω = exp(−D²) on the unaligned D (reviewer digitisation). Literal
AGS lowers the mean densification statistic 8.5× (evidence/ags_gradstat.json: 1.06e-5 vs 8.96e-5;
0.09 % vs 10.5 % of Gaussians above grow_grad2d), and the 2k smoke never densifies (139k → 101k,
20.93 dB vs 23.07 baseline). Densification is secondary, though: with an unscaled statistic
(910k Gaussians) or ω on conics only, literal ω still gives 21.1 dB, and it is already worse at
step 400, before densification starts (reviews/review-1.md §1, evidence/reviewer_scratch/). Literal ω
fails at any scope; `aags` is an inferred implementation detail, not a reading of the text. Added `3dls-aags` (ω = exp(−(D/λ)²), AGS on the DA-aligned
distance; N 417k vs LK+DA 836k at 2k = 0.50, paper ratio 0.57; 22.80 dB). Chosen from a training
statistic the paper reports, not from test images; labelled interpretation. Benchmark primary =
`3dls-aags`; `3dls` optional.
