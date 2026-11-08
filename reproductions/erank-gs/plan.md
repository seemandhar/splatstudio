# Plan — erank-gs (plan id 130)

Method `erank-gs` (module `ss_erank_gs`) = paper row "+a+b" / "3DGS+e": vanilla 3DGS rendering +
fixed (GOF-style) ADC + effective-rank & thinness loss from step 7000. Variants for gap analysis:
`erank-gs-adc` ("+a": no erank loss) and `erank-gs-3dgs` (paper's 3DGS row: Inria ADC, prune
0.005, black background) so the DTU baseline is run under the identical protocol (the shared
gsplat baselines render on white and are M360-only).

## Components
| name | kind | factory | requires → provides | config (paper/code values) |
|---|---|---|---|---|
| erank-renderer | renderer | ss_erank_gs.renderer:create_renderer | state.gaussian3d, camera.pinhole → render.rgb, render.depth.expected_z, gradient.screen.signed, gradient.screen.absolute | background [0,0,0] (white_background=False), gsplat classic (0.3 dilation), absgrad=True |
| erank-density | density | ss_erank_gs.density:create_density | gradient.screen.absolute | thr 2e-4, 500–15000/100, abs_gradient True, clone_resample True, percent_dense 0.01, min_opacity 0.05, reset 3000 → 0.01, world prune 0.1·extent after 3000 |
| photometric | loss | splatstudio.losses:create_photometric | render.rgb | L1 + 0.2 D-SSIM (unchanged) |
| erank | loss | ss_erank_gs.losses:create_erank_loss | state.gaussian3d | lambda 0.01, thin_lambda 1, eps 1e-7, start 7000, end 30000 |

No new fields.

## Equations → code
- erank (Eq. 1–3): `q = s²/Σs²; erank = exp(-Σ q log q)` on `exp(log_scales)` (get_effective_rank).
- Eq. 4 (code form): `λ·mean(clamp(-log(erank-1+1e-7), 0)) + thin·mean(sort(s)[:, 2])`, active for
  start ≤ step ≤ end.
- ADC (App. A.4, GOF densify_and_prune_v2): gsplat `absgrad` gives per-component Σ_pixels |g|;
  in NDC units (×W/2, ×H/2) the L1 of that 2-vector equals the official kernel's Σ(|gx|+|gy|)
  exactly. Signed: L2 of the NDC 2D grad. Quantile matching, clone resampling, split (gsplat
  ops, s/1.6), prune — ported from reproductions/gof density (minus 3D filter).

## Probes
| probe | checks | fails if |
|---|---|---|
| erank-formula | erank of sphere=3, disk=2, needle→1; loss equals an independent formula; zero before step 7000; needle gradient increases small scales | loss removed / wrong ε / schedule |
| adc-abs-metric (CPU) | synthetic stats: a Gaussian with cancelling signed gradients but large abs gradient is cloned only with abs_gradient | abs path removed |
| absgrad-units (CUDA) | renderer's info supplies absgrad; per-Gaussian L1(absgrad·W/2,H/2) ≥ L2(grad·W/2,H/2) and a finite-difference-free check vs a hand-built single-Gaussian scene | renderer without absgrad |
| density-recipe (CPU) | clone resample, split /1.6, prune 0.05 / 0.1·extent, reset to 0.01 at 3000 | recipe changes |
| state-transitions | base runtime probe | — |

## Fidelity gaps
1. NVS split unspecified in paper — assumed 3DGS `--eval` llffhold8 (7/8 test views). Effect: absolute PSNR may shift ±0.5 dB; compare against our own `erank-gs-3dgs` baseline too.
2. Resolution 777×581 (2DGS `-r 2` on 1554×1162) vs paper's stated 800×600 — what the official loader actually produces.
3. Runtime: views sampled with replacement; optimizer step before densification; SH degree up at step 1001 vs 1000; SSIM valid conv. Small.
4. Opacity-mask BCE in official train.py (crashes as released, not in paper) — not ported.
5. gsplat rasterizer instead of Inria (same 0.3 dilation; numerics of absgrad identical in definition).
6. Mesh/CD evaluation not ported (NVS PSNR only).
