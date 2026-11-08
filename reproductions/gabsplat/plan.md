# Plan — 3DGabSplat native port

Method name **`gabsplat`**, module **`ss_gabsplat`**, package `splatstudio-method-gabsplat`.
Ablation method **`gabsplat-noreset`** (Gabor primitive, plain 3DGS density: no child / periodic
Gabor resets) for the C3 probe/diagnostics only.

## Components
| kind | name | factory | requires → provides | config (paper value) |
|---|---|---|---|---|
| renderer | gabor-renderer | `ss_gabsplat.renderer:create_renderer` | state.gaussian3d, camera.pinhole → render.rgb, render.depth.expected_z, gradient.screen.signed | `num_freqs 2`, `tile_size 16`, `eps2d 0.3` |
| density | gabor-density | `ss_gabsplat.density:create_density` | gradient.screen.signed | DefaultStrategy kwargs + `prune_opa 0.01` (reset value 2×=0.02), `opacity_lr 0.025`, `child_freq 0.001`, `child_weight 0.01`, `reset_weight 0.01`, `child_reset true`, `weight_reset true` |
| loss | photometric | splatstudio built-in | render.rgb | L1 + 0.2 D-SSIM (unchanged) |

## Fields
| field | shape | lr | init | density treatment |
|---|---|---|---|---|
| `gabor_freq` | (2,3) world cycles/unit | 0.02 (supp. B) | 0.05 (supp. B, Mip360) | copied by gsplat ops; new children reset to 0.001 |
| `gabor_weight` | (2,) logit | 0.01 (supp. B) | logit(0.01) | children reset to logit(0.01); all reset to logit(0.01) every 3000 (<15k) with opacity |

## Renderer (C1 + C2)
1. gsplat `fully_fused_projection` (EWA, eps2d 0.3) → means2d, conics, depths, radii (Gaussian
   3.33σ boxes contain the Gabor support since modulation ≤ 1).
2. Torch (autograd) 2D frequency: cam-space mean t, gsplat-clamped J (2×3), Σ = R S² Rᵀ,
   `a = J · Rcam · Σ · f` (N,F,2), `f2d = conic_mat · a` (Eq. 9-10 coordinate-free; dilated conic,
   ambiguity 7). Non-visible Gaussians use z=1 to keep gradients finite.
3. Custom CUDA rasteriser `csrc/gabor_raster.cu`, adapted from the validated
   `reproductions/linear-splatting` kernel (itself from gsplat 1.5.3
   RasterizeToPixels3DGS{Fwd,Bwd}.cu): per-Gaussian extra input `gab [N, 3F]`
   (f2d_x, f2d_y, ω). alpha = min(0.999, o·G·m), m = 1 − Σω + Σ ω cos(2π f2d·δ), δ = μ2d − p.
   Backward: v_opac = G m v_α; v_σ = −o G m v_α; v_δ += o G v_α Σ −ω sin(φ) 2π f; v_f = o G v_α (−ω sin φ 2π δ);
   v_ω = o G v_α (cos φ − 1). Gates identical to gsplat (skip α<1/255, 0.999 saturation gate).
4. SH, tiles, white background, RGB+ED depth exactly as `gsplat.rasterization`.

## Density (C3)
Subclass `splatstudio.backends.gsplat.Density`; re-implement `step_post_backward` with gsplat's
own `_update_state/_grow_gs/_prune_gs`: after `_grow_gs` the last `n_dupli + 2·n_split` rows are the
new children (gsplat appends duplicates, then split children) → reset their Gabor fields in place
(`.data` writes keep the Parameter / requires_grad) and zero their Adam moments; then prune.
Opacity reset (splatstudio fix schedule, value 0.02) + weight reset to 0.01 with zeroed Adam
moments of `gabor_weight`. Opacity lr set to 0.025 in `initialize`. Trainable-leaf check after
every population change.

## Probes
| probe | contribution | fails if |
|---|---|---|
| gabor-forward-vs-dense (CUDA) | C1/C2 | kernel ≠ float64 dense reference; or Gabor render == Gaussian render |
| gabor-backward-vs-dense (CUDA) | C2 | any of 6 gradients ≠ autograd of dense ref |
| frequency-projection (CPU) | C2 | renderer f2d ≠ paper Eq. 9-10 built from an explicit 3×3 ray-space Jacobian (two different completions), or ≠ numerical z-integration of the 3D Gabor |
| frequency-adaptive-density (CUDA) | C3 | children not reset / parents changed / weights not reset at 3000 / fields not trainable |
| recipe (CPU) | recipe | lrs/init/prune_opa ≠ supp. B |
+ base state-transitions, photometric-gradient, renderer-gradient.

## Files
`plugin/pyproject.toml, README.md, ss_gabsplat/{__init__,renderer,kernels,frequency,reference,density,probes}.py,
ss_gabsplat/csrc/{ext.cpp,gabor_raster.cu}, tests/test_gabsplat.py`

## Fidelity gaps
1. **No official code** — every numeric detail from the text; projection/integration
   re-derived. Effect unknown; closeable when code is released.
2. **F=2 symmetry** (ambiguity 2): literal constant init keeps both kernels identical. Expected
   effect: capacity of F=1. Could be closed with code.
3. **LR / init conflict** main vs supp.; supp. chosen. Possibly ±0.1 dB.
4. **Dilated conic in f2d** (ambiguity 7) — small-footprint Gaussians get lower 2D frequency.
5. Runtime protocol differences vs Inria 3DGS (white bg, random view sampling, SSIM valid conv;
   our gsplat baseline garden 26.89 vs paper 3DGS* 27.43; room 31.52 vs 31.46) → compare deltas.
6. Opacity lr 0.025 differs from our baseline (0.05); part of the paper's recipe, but it is
   not a Gabor effect — kept since the paper specifies it.
