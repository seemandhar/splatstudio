# Plan — Deblurring 3DGS native port

Methods: `deblurring-3dgs-defocus` (real defocus config) and `deblurring-3dgs-motion` (real
camera-motion config). Module `ss_deblurring_3dgs`. Package
`splatstudio-method-deblurring-3dgs` in `plugin/`.

## Components
| component | kind | factory | requires → provides | config (paper/code values) |
|---|---|---|---|---|
| `deblur3dgs-renderer` | renderer | `renderer:create_renderer` | state.gaussian3d, camera.pinhole → render.rgb, render.depth.expected_z, gradient.screen.signed, render.blur_mlp | mode, lambda_s 0.01, lambda_p 0.01, max_clamp 1.1, background black |
| `deblur3dgs-density` | density | `density:create_density` | gradient.screen.signed | 3DGS clone/split (thr 2e-4 / 5e-4), depth-scaled prune (t_p 5e-3 / 1e-2, prune_range 3), from 500 / until 15000 / every 100, no opacity reset, no size prune, xyz lr 1.6e-3→1.6e-5 ×extent over 30000, extra points at 2500 (K 4, t_d 2, bound 10, rate 1.1, N_max 2e5), owns the blur bank (MLP lr 1e-3, width 64, hidden 3, PE 3/10, num_moments 4) |
| `deblur3dgs-photometric` | loss | `losses:create_photometric` | render.rgb | 0.8 L1 + 0.2 (1 − SSIM) with the Inria same-padding SSIM |

No new per-Gaussian fields. The MLP and its Adam live in `bank.BlurBank` (process-local
`bank.ACTIVE`, created by the density in `initialize`, stepped in `after_backward`,
exported every step into `density.state["bank"]` = {mlp state_dict, optimizer state_dict,
config} so the checkpoint carries it). `ss-eval` has no bank → sharp render (C5).

## Mapping
- Renderer training path (bank active, grad enabled): MLP(detach x, detach exp(s),
  detach normalize(q), camera centre) → δs, δr clamp(λs δ + 1 − λs, 1, 1.1); Σ =
  R(q̃) diag(s δs)² R(q̃)ᵀ with q̃ = normalize(q)·δr *unnormalised* (Inria computeCov3D), passed
  to gsplat `covars`. Defocus: 1 render. Motion: render 0 at x with the (M+1)-th factors, renders
  1..M at x + λp δx_i; mean of 5. Info list for densification. Black background.
- Eval path: plain gsplat render (quats/scales), black background.
- Density: stats per render (‖grad·(W/2,H/2)‖, denom += 1/n_renders), densify at step > 500,
  % 100, < 15000: clone → split (gsplat ops) → prune sigmoid(o) < t_p / (1 + 2 (z−zmin)/(zmax−zmin)).
  At 2500 (after densify): add_points (vectorised KNN, distance-proportional weights, t_d
  rejection), reinit all scales/quats/opacities, clear all Adam states, reset MLP Adam.
  xyz lr set every step to expon(step+1) over 30000.

## Probes
| probe | contribution | fails if |
|---|---|---|
| `mlp-architecture` | C1 | PE/inputs (91 ch), layer shapes, init gains or clamp differ from GTnet |
| `defocus-blur` (cuda) | C1, C5 | training render ≠ independent Σ(q̃) render; δ<1 possible; MLP gets no grad; eval ≠ plain gsplat |
| `motion-blur` (cuda) | C2 | training render ≠ mean of 5 independent renders (render 0 unshifted, others x+λp δx) |
| `extra-points` | C3 | N_p, rejection, colour interpolation, reinit or optimizer reset differ from brute-force reference |
| `depth-pruning` | C4 | far Gaussian with t_p/3 < o < t_p pruned or near one kept |
| `schedule` | all | opacity reset happens, thresholds/intervals/xyz lr differ |

## Fidelity gaps
1. gsplat vs Inria rasterizer numerics (same EWA, 0.3 dilation) — small.
2. Runtime samples train views with replacement (official: shuffled stack) — small.
3. Runtime steps the optimizers before density (official: densify before step; at 2500 the
   official step is skipped because the optimizer is recreated) — small.
4. SH degree increments one step later (runtime `(step-1)//1000`) — negligible.
5. ss-eval SSIM/LPIPS differ from Deblur-NeRF metrics (skimage SSIM, LPIPS-Alex); PSNR is
   the comparable number. Our run is a single seed.
6. Baseline: the existing `baseline-gsplat-deblurnerf-40000` uses the original Deblur-NeRF
   COLMAP points, 40k steps — not this protocol; a new gsplat baseline on `deblur3dgs-*` at
   20k is requested.

## Gap analysis (after benchmark)
- Official released models (weights.zip) evaluated with our renderer/split reproduce the paper
  (cake 26.89, caps 24.47, ball 28.24, girl 23.26) → data, split, background, eval are right.
- caps: our MLP was dead (all outputs clamped, Adam moments ~1e-45); official checkpoint's
  GTnet Adam moments are live at 20k. Cause: init draw. Fixed: official CUDA seed-0 xavier
  draw (bank.GTnet). 5k diagnostic caps 22.4-23.0 → 23.7.
- cake: MLP alive in both; residual ~2 dB unexplained (ours fits blurry train views 1.5 dB
  better with the sharp latent → less blur explained by the MLP). Candidates: runtime-level
  ordering (official skips Gaussian updates at densify iterations), with-replacement sampling,
  more Gaussians (327k vs 265k).
