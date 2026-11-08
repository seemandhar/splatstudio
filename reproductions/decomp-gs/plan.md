# DecompGS native plan

Method `decomp-gs`, module `ss_decomp_gs`, plugin in `plugin/`. No official code: written from
the paper (see `paper.md`); resolution plumbing (renderer-side downscale + antialiased target)
adapted from the DashGaussian port (`reproductions/dashgaussian`), which the paper cites as the
inspiration for its C2F scheme.

## Components
| component | kind | factory | requires / provides | config (paper value) |
|---|---|---|---|---|
| decomp-renderer | renderer | `ss_decomp_gs.renderer:create_renderer` | provides rgb, depth, `gradient.screen.signed`, `render.c2f_resolution` | – ; renders training views at floor(H/r)×floor(W/r) with scaled K, evaluation (no_grad) at full res |
| decomp-density | density | `ss_decomp_gs.density:create_density` | requires `gradient.screen.signed` | densify_until 25000, K 8, p 0.03, τ_u 0.05, g2l, c2f, adaptive_prune, 3DGS defaults (2e-4, 0.01, 0.1, 500/100, reset 3000→0.01) |
| photometric | loss | `ss_decomp_gs.losses:create_photometric` | requires rgb + `render.c2f_resolution` | L1 + 0.2 D-SSIM (same padding, 3DGS) against D_r(target) |

No new fields. Density ops via `gsplat.strategy.ops` (keep `requires_grad`, optimizer state
aligned — avoids the ResGS `torch.no_grad` Parameter pitfall; probe `decomp-g2l-phases` asserts
all fields remain trainable leaves bound to their optimizers).

## Mapping
- **C2 C2F (Eq. 1-3)**: `set_training_cameras` reloads the training images (same split, checks
  K/viewmat), `schedule.scale_energies` computes E_r = Σ_views ‖|FFT(D_r(I))|‖₁·r² for r=1..8,
  `end_iterations` T_r = round(E_r/E_1·25000); r(t) = max{r : T_r > t} (else 1). The density
  sets `current_scale` for the next step; the renderer reads it through `LINK`. Position LR:
  constant until T_2, then log-linear to 1 % at 30000 (`lr_mode`). Measured: truck T_8..T_2 =
  4525…13187, train 4516…13153, garden 3175…11044.
- **C1 G2L**: `densify_and_prune(step)`: step < T_2 → only split (max scale > 0.01·extent),
  clone candidates skipped (`global_mode=skip_clone`; `split_all` alternative); step ≥ T_2 →
  3DGS clone + split.
- **C3 adaptive pruning**: after growth, τ = min(sorted(σ(o))[⌊Np⌋], 0.05), prune σ(o) < τ, OR
  world-size (> 0.1·extent) after step 3000. Every densify step (500 < t < 25000, %100).
- Opacity reset every 3000 while densifying (3000…24000).

## Probes
| probe | measures | fails if removed |
|---|---|---|
| decomp-c2f-schedule | E_r vs numpy FFT + PIL resize; [T_{r+1},T_r) boundaries; flat image → T_r = T | c2f off renders r=1; wrong normalisation |
| decomp-render-scale | low-res render = gsplat with scaled K; eval full-res | renderer ignores schedule |
| decomp-target-downsample | target = PIL antialiased bilinear | aliased/no downsample |
| decomp-lr-schedule | LR constant until T_2, decay after | 3DGS decay |
| decomp-g2l-phases | clone=0 before T_2, 16 clones after / with g2l off | G2L removed |
| decomp-adaptive-prune | τ vs numpy quantile, τ_u cap, prune count | fixed 0.005 |
| decomp-cycle | runtime run: scale monotone to 1, no clones before T_2, pruning > 0 | any |
| decomp-renderer-gradient | screen gradients at r=2 reach stats | – |

## Benchmark choice
Paper reports **dataset averages only** (Table 2; no per-scene table, no supplementary).
Cheapest complete average: **Tanks & Temples (truck, train)**, 30K steps, `--recipe 3dgs`.
Compare Δ vs `baseline3dgs-tandt-30000` (gsplat 23.36) to the paper's Δ vs its 3DGS
(24.06 − 23.62 = +0.44 dB; vs 3DGS-accel +0.48). Absolute target 24.06 ± 0.5.

## Fidelity gaps
1. **3DGS-accel backbone** (Taming per-splat backprop + sparse Adam): dense Adam here; speed
   only in theory, sparse Adam changes moment updates of invisible Gaussians (paper: 3DGS-accel
   −0.04 dB vs 3DGS on T&T). Could add taming-3dgs's sparse Adam later.
2. **Global-phase rule for clone candidates** ambiguous (skip vs split); default skip (Fig. 2).
3. **LR reduction form** unstated; DashGaussian-style log-linear decay from T_2.
4. **Energy aggregation** over views unstated (sum over all training views).
5. Runtime-wide: white background, random-with-replacement view sampling, valid-conv eval SSIM,
   LPIPS-VGG; shared with the baseline. Inria resets opacity at step 500 with white background —
   not done (baseline doesn't either).
6. Timing: our wall-clock is not comparable to the paper's RTX 3090 minutes; only relative
   speed-up vs our gsplat baseline is meaningful.
