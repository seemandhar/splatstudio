# Perceptual-GS — native SplatStudio plan

Method `perceptual-gs`, module `ss_perceptual_gs`, package `plugin/`.

## Components
| component | kind | factory | requires / provides | config (paper value) |
|---|---|---|---|---|
| pgs-renderer | renderer | `ss_perceptual_gs.renderer:create_renderer` | provides render.rgb, render.depth.expected_z, gradient.screen.signed, training.target_image, training.camera | stock gsplat classic; forwards the frame's camera + target image |
| pgs-loss | loss | `ss_perceptual_gs.loss:create_loss` | requires render.rgb, training.camera | lambda_s 0.1, bce_until 15000, ssim_weight 0.2, sidecar_root |
| pgs-density | density | `ss_perceptual_gs.density:create_density` | requires gradient.screen.signed, training.target_image | 3DGS defaults + Tab. 8 values, decision_step 600, od_k 1.2, num_depth 3.5M, reinit_every 5000 |

Field: `sensitivity` shape () (logit), lr 0.05, init logit(0.1) = −2.19722. Density ops use
`gsplat.strategy.ops` (generic fields are copied by duplicate/split/remove); full re-inits
rebuild every Parameter with `requires_grad=True` and fresh Adam state; a trainable-leaf check
runs after every population change (ResGS pitfall).

## Mapping
- C1 (Eqs. 5–7): `ss_perceptual_gs/sensitivity.py` (pure torch + PIL), CLI `ss-pgs-prep`
  builds a sidecar `/bigdata/sejain/perceptual-gs-sens/<data path relative to ~/datasets>/`
  (`viewmats.npy`, `names.json`, `sens_u8.npz`) from the **full-resolution** raw images, resized with
  PIL bicubic to the training size (identical to official `PILtoTorch`). The loss/density find the
  sidecar from ss-train's `--data` (or `PGS_DATA`) and match frames by viewmat.
- C2 (Eqs. 8–10): loss renders `colors=(1−σ(ε))` (1 channel, black bg) with the same gsplat
  settings; `L = 0.9·photo + 0.1·BCE(1−r, gt)` while step ≤ 15000 (BCE clamps log ≥ −100 like
  torch BCELoss), plain photometric afterwards.
- C3 (Eqs. 11–16): density, official `densify_and_prune` order (clone with OD → split → prune);
  per-Gaussian `trans = max_v Σ_pixels α T` from the Mini-Splatting stats kernel (custom CUDA
  re-traversing gsplat tile lists, vendored from `reproductions/mini-splatting`).
- C4 (Eqs. 17–18): γ/β decision at step 600; sparse scenes: depth reinit at 5000/10000 (kernel
  `ms_depth`, sampling with replacement + unique), no opacity reset, self-reinit at 15000.
- C5 (Eqs. 19–20): OD applied in place to clone-selected originals before duplication.

## Probes
| probe | checks (fails if mechanism removed) |
|---|---|
| pgs-extraction | map of a synthetic image equals an independent numpy Sobel/threshold/pool reference; differs if τ_e or pooling is removed |
| pgs-dual-branch | rendered sensitivity of one opaque Gaussian = σ(ε)·A + (1−A); BCE gradient pushes ε toward the target; loss at step 15001 = photometric |
| pgs-guided-densify | zero-gradient Gaussian with σ=0.95 and weight>25 splits at step 1000 (not 1100, not with σ=0.5 at 1000), medium σ=0.5 splits at 1500 |
| pgs-opacity-decline | clone opacity equals 1−sqrt(1−α^1.2) |
| pgs-scene-decision | β and γ match an independent computation; γ>0.55 sets sparse |
| pgs-stats-kernel | CUDA weight sums / argmax / T match brute-force blending |
| pgs-depth-reinit | reinit replaces population by depth points with GT colours, all fields trainable |

## Unchanged from 3DGS
Photometric loss (L1 + 0.2 D-SSIM), LR schedule (runtime), SH schedule, 3DGS clone/split/prune.

## Fidelity gaps
1. White background in RGB branch (runtime); sensitivity branch uses black as official. Small.
2. Runtime optimizer steps before density (official skips Adam on densify steps). Small.
3. Stats kernels use gsplat numerics (0.999 alpha cap, +0.5 pixel centres) vs official 0.99. Small.
4. View sampling with replacement vs official shuffled stack (affects which views seen by 5000; GT
   colour cache needs every view rendered before the first depth reinit — P(miss) ≈ 1e-10).
5. SH degree increments one step later (runtime). Negligible.
6. SSIM valid conv / LPIPS package (runtime-wide; compare deltas vs our baseline).
7. Benchmark on `mipnerf360-pilr` (matches official `-r`), baseline `baseline3dgs-mipnerf360-pilr-30000`.
