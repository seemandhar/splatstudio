# Review 1 — confidence-densify (arXiv 2610.13541)

Reviewer: splat-reviewer (adversarial), round 1. Plugin code was not edited. The ablation
variants below live in a scratch module that wraps the plugin and were run with the plugin's
own runtime entry points.

**Verdict: NOT APPROVED.** 1 critical, 2 major, 5 minor.

## Smoke gap (garden, 2k it: CAD 21.38 vs MCMC control 22.67, −1.29 dB)

Our verdict: this is a **bug**, not a misreading of the paper's schedule. The confidence map
does not follow CoMe's numerics, and that breaks the training signal. Evidence comes from
2k-step garden runs. Each run used the same protocol as the smoke runs (recipe 3dgs, seed 0,
GPU 4) and was evaluated on the test split:

| variant | test PSNR |
|---|---|
| plugin as is (rerun; smoke-1 = 21.38) | 20.99 |
| CoMe-faithful confidence (normalised, gradient only to γ) — densification unchanged | **22.55** (+1.56) |
| plain L_rgb loss (no confidence), densification unchanged | 22.57 |
| smoke-control (gsplat MCMC, same budget) | 22.67 |

Fixing only the confidence numerics recovers 1.56 dB. This puts the method within 0.1 dB of
the MCMC control, and level with the plain-loss variant. So the whole smoke deficit comes from
CRITICAL-1.

The densification itself (sampling, relocation, colour init, kernel) is neutral at 2k steps.
That is expected: the paper's gains appear while relocation runs, up to 10k–25k. The rerun
spread is about ±0.4 dB (21.38 vs 20.99 for the same configuration), which is why the result
should be judged at ≥7k–30k steps.

Scratch scripts and logs:
/tmp/claude-1013/-home-sejain/6ad4135e-056e-4bd4-9844-6b8764bea041/scratchpad/cadchk/

## Issues

### CRITICAL-1 — Confidence map is unnormalised and back-propagates into geometry (renderer.py:41-63, losses.py:49-53)
- **Paper:** §3 Prelim, Eq. (confidence), "Ĉ(p) ∈ R+" blended from γ̄. Fig. confidence says
  "at iteration 500 the maps are nearly white because all confidences start at 1".
- **CoMe reference code** (`submodules/diff-gaussian-rasterization/cuda_rasterizer/stopthepop/hierarchical_render.cuh`):
  - :1231 has `confidence /= max(1 - T, 1e-3)`. The map is the **opacity-normalised** average
    of γ̄, so it is 1 everywhere at init.
  - :1513-1515 and :1643 show that dL/dĈ goes **only** to `dL_dconfidences`. It never enters
    `dL_dalpha`, `dL_dmean2D` or `dL_dconic`.
- **Code:** the plugin rasterises γ̄ as an extra colour channel through gsplat:
  - Ĉ = Σ b γ̄, which equals alpha at init.
  - Gradients of `pp·Ĉ − β log Ĉ` therefore flow into opacity, means, scales and quats.
- **Consequences:**
  - (a) Photometric gradients are scaled by alpha, so partially covered pixels are under-fit.
  - (b) The geometry receives a spurious force. `pp·Ĉ` pushes opacity **down** wherever the
    error is high. `−β/Ĉ` (up to 75 per pixel at the clamp) pushes alpha up at
    low-coverage pixels.
  - (c) The same spurious gradient contaminates `means2d.absgrad`, which drives the 80 %
    Eq. pdf sampling. paper.md §7 says "as in CoMe"; that claim is wrong.
- **Probe:** `confidence-render` asserts the wrong formula (`confidence == alpha` at γ=0,
  probes.py:103).
- **Fix:**
  - Render Ĉ = Σ b γ̄ / max(1−T, 1e-3) with geometry detached for the confidence channel,
    e.g. a second rasterisation with detached means/quats/scales/opacities, or a custom
    backward. Exclude it from absgrad.
  - Change the probe to assert Ĉ ≡ 1 at γ = 0 for any alpha, and that the confidence term
    gives zero gradient to opacity and means.

### MAJOR-1 — `confidence-render` / `renderer-gradient-cad` probes cannot detect CRITICAL-1 (probes.py:93-114, 356-372)
- `confidence-render` encodes the unnormalised formula.
- `renderer-gradient-cad` only checks that the gradients are finite and non-zero.
- **Fix:**
  - Add a probe that, with γ = 0, the confidence loss's gradient with respect to the
    opacities equals that of `mean(pp·sg(Ĉ))`, i.e. there is no path through Ĉ.
  - Add a probe that Ĉ = 1 at partial alpha.

### MAJOR-2 — Smoke control is not a like-for-like early-training comparison (control.py; bench_request)
- The control rasterises with the stock renderer (absgrad off), keeps MCMC noise, and keeps
  the regulariser to 30k. That matches the paper's baseline row.
- At 2k steps, however, only 3 relocations (800/1200/1600, w = 0 only) and ~14 growth steps
  have happened, and the confidence factor is at η = 0.4. The smoke tests mostly the loss,
  not the densification.
- **Fix:**
  - Gate the benchmark on a 7k/30k comparison after CRITICAL-1 is fixed.
  - Report the plain-L_rgb ablation (row above) next to it, so the effect of densification
    is separated from the effect of the loss.

### MINOR
1. **density.py:303-310** — growth-added primitives start with max_w = 0. Before 5k they are
   relocated at the next 400-step boundary if no random view has hit them in ≤ 300 it.
   - The paper exempts only relocated primitives, so this is literal.
   - The w window for a new primitive is shorter than 𝒱 for the others. Consider measuring w
     only over views since the primitive's birth, or label this as an ambiguity.
2. **density.py:251** — `refine_start < step` with 1-based steps puts the first densify at
   600 and the first relocation at 800. This matches the Inria `iteration > densify_from_iter`
   convention. OK, noted for completeness.
3. **density.py:184-207** — colour init uses `means2d` from the render *before* this step's
   optimizer update. The position drift is negligible.
   - The SH degree used is the current render degree. Higher bands are copied, as the paper
     says.
   - Correct per §3.3.
4. **density.py:226** — confidence ranks use exp(γ) per primitive. This is correct per
   Eq. pdf. After the CRITICAL-1 fix, γ is trained only by its own term, as in CoMe.
5. **paper.md §7, last bullet** — states that CoMe lets the confidence gradient reach ∇_abs.
   It does not (hierarchical_render.cuh:1643). Update it.

## Checked and found correct
- **Loss reduction:** per-pixel/channel `0.8|I−I*| + 0.2(1−SSIM_map)`, mean, clamp
  [1e-3, 5], β = 0.075. Matches CoMe train.py:203-217 and appearance_network.py:104.
- **Photometric loss:** correctly *replaced*, not summed.
- **Loss timing:** active from step 1. This matches paper §4 "throughout training"; CoMe's
  own default starts at 500.
- **λ_reg:** 0.01 on mean sigmoid(o) and mean exp(s), stopping at 25k, as in paper §4.
- **Eq. pdf:** min-rank ties, η ramp, 1.2 offset, 80/20 split. `draw_sources` excludes the
  primitives being relocated.
- **Relocation schedule:** w = 0 only and ≤ 10 % before 5k; 10 % for 5–10k; Gaussian
  fall-off from 10k; every 400 it; exemption of the last relocation's destinations.
- **`place`:**
  - MCMC `compute_relocation` with ratio = copies + 1; gsplat clamps ratio to n_max = 51.
  - Opacity clamp at 0.005.
  - The adjusted scale is used both for the source and for the N(μ, R diag(s')² Rᵀ) sample.
  - The source mean is kept. Rotation, SH and confidence are copied. No noise.
  - Moments of sources and destinations are zeroed.
  - New Parameters are built with `requires_grad=True`. The TODO-runtime p.clone() pitfall
    does not apply: `p.data` is used, then cat or in-place, then wrapped as a new Parameter.
- **Colour init:**
  - Bilinear GT at the projected mean; `align_corners=False` with +0.5 pixel centres matches
    gsplat's means2d convention.
  - DC = (c − 0.5 − SH_{l≥1})/C0.
  - Moments are reset.
  - Radius < 16 px, 400-it timeout.
- **csrc/cad_stats.cu:**
  - Alpha numerics match gsplat's fwd: min(0.999, ·), skip at σ < 0 or α < 1/255, stop
    exclusively at T ≤ 1e-4, +0.5 pixel centres.
  - atomicMax on the float bits is valid because w ≥ 0.
  - The per-view max is folded into the running max; the error sum is weighted by α·T.
  - Covered by the brute-force oracle probe.
- **Runtime order:** `after_backward` runs after `opt.step()` (runtime.py:205-211), so colour
  init and placement are not overwritten by the step that follows.

## Equation coverage
| eq | status | code |
|---|---|---|
| photoloss (per pixel) | exact | losses.py:per_pixel_rgb_loss |
| confidence (Ĉ blending) | **deviates** (unnormalised; gradient into geometry) | renderer.py:render |
| confidence (loss) | exact | losses.py:confidence_loss_value |
| reg | exact (code-mean form) | losses.py:create_reg |
| pdf | exact | density.py:sampling_pdf |
| w_i | exact | csrc/cad_stats.cu, density.py:_accumulate |
| relocation share | exact | density.py:relocation_share |
| position sampling / MCMC rule | exact | density.py:place, sample_positions |
| colour init | exact | density.py:_color_init, set_dc_to_color |
