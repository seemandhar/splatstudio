# Review 1 — vol3dgs (Vol3DGS, arXiv 2412.03378)

**Verdict: not approved** — 0 critical, 2 major, 4 minor.

The renderer (alpha formula, blending, backward) and recipe match the official code. Two
benchmark-relevant deviations remain: the density round trip uses the mathematically correct
inverse where the official code has an operator-precedence quirk (which changes densification),
and the background is white where the official code is black.

## Checked and matching the official code
- **Volumetric alpha (Eq. 13–19)**: `csrc/vol_raster.cu:49-70` vs `slang/utils.slang:574-614`
  (`evaluate_splat_volr`). They match term by term: den = max(eps, dᵀAd), num = μᵀAd,
  num ≤ 0 → 0, t = num/den, pdf = exp(-½ diffᵀA diff), σ_d = sqrt(1/(den+eps)),
  α = min(0.99, 1 − exp(−2.506 κ σ_d pdf)), eps = 1e-7.
- **Blending**: skip α < 1/255; stop when T(1−α) < 1e-4 without compositing that Gaussian
  (`alphablend_volr_shader.slang:91-105`, our `vol_raster.cu:127-139`).
- **Backward**: the envelope-theorem derivation is correct: dq/dμ = −2A·diff, dq/dA = diff·diffᵀ,
  dσ/dden = −½σ³ with zero gradient while den is clamped, and zero gradient through the 0.99
  clamp. Non-finite contributions are skipped, as in the slang code.
- **κ reparameterisation**: `reference.py:23` = `utils/general_utils.py:33-34`
  (−log(1−0.99θ)·mean(1/s)/2.506). The paper's Eq. 20 omits the 1/2.506 factor; following the code
  is correct, since the paper's numbers come from the code.
- **Tiles**: a 3σ radius with max(0.1, mid²−det), eps2d 0.3, near plane z > 0.2, and no
  scale3d_factor (`alphablend_volr_tiled_slang.py:34` sets it to ones).
- **Principal point**: the official code uses W/2 and H/2 (`alphablend_volr_shader.slang:63`); we
  use K's cx and cy. The two are the same for centred PINHOLE intrinsics.
- **Colour**: SH + 0.5, then softplus with β = 10 (`spherical_harmonics.slang:172-174`). The SH
  sign convention is the same as gsplat's.
- **SH degree**: rises at step % 2000 == 0, i.e. min(step//2000, 3). The runtime's min() with
  (s−1)//1000 never lowers it.
- **Densification**:
  - world-space ‖dL/dμ‖ (`train_volr.py:104-105`);
  - visible = radii > 0;
  - schedule: > 500, < 15000, every 200, N < 7e6;
  - clone ≥ 5e-5 with max s ≤ 0.01·ext;
  - split ≥ 1e-7 with max s > 0.01·ext; children s/1.6, θ halved using the parent's scale;
  - prune θ < 0.05, plus max s > 0.01·ext when step > 3000 (`size_threshold` is gated on
    `iteration > opacity_reset_interval` = 3000);
  - the screen-size prune never fires (postfix zeroes max_radii2D first) — confirmed;
  - extent ×2; no opacity reset.
- **LRs**: θ 0.03, sh0 0.003, shN 0.003/20, scales 0.005, quats 0.001, means 1.6e-4 → 1e-5
  × extent with exponential decay; these override the runtime decay every step. θ init 0.1 and
  3-NN scales match.
- **Resolution**: the official script passes no `-r`, so the loader resizes to width 1600
  (`camera_utils.py:25-32`). `w1600` with downscale 1 is correct for both bicycle and room.
- **requires_grad**: `means_in = means.view_as(means)` is a non-leaf view.
  - `retain_grad` runs only when `requires_grad` is set, so eval under no_grad is safe.
  - `zero_grad(set_to_none)` in the runtime does not clear a non-leaf's `.grad`, so the statistic
    survives until `after_backward`.
  - `density.py:150` asserts that every field is still a leaf with `requires_grad` after
    densification.
  - The `world-gradient` probe covers this path. No pitfall found.

## Issues

### MAJOR 1 — `density2alpha` does not reproduce the official operator precedence
- **Where**: `plugin/ss_vol3dgs/reference.py:29`, used by `density.py:117,127,132`.
- **Official code**: `utils/general_utils.py:56`
  `alpha = (1 - torch.exp(-density*2.506*1.0/((mean(1/s)))) / 0.99).clamp(0, 0.99999)`.
  The `/ 0.99` applies only to `exp(...)`, so the official code computes θ' = 1 − e^{−x}/0.99 and
  **not** (1 − e^{−x})/0.99. The plugin (and `paper.md:46`) assume the correct inverse.
- **Consequences in the official code** (these produced the paper's numbers):
  - (a) The round trip gives θ − 0.0101. The θ ≥ 0.99 split test
    (`gaussian_model_volr.py:450`) needs θ ≥ 1.0001, so **it never fires**. Our split fires for
    θ ≥ 0.99, i.e. every 200 steps for every saturated Gaussian.
  - (b) Halved θ for clones and split children is about 0.0102 lower than ours
    (θ = 0.1: 0.0412 official vs 0.0513 ours). With min θ = 0.05, the official code immediately
    prunes clones and children of any Gaussian with θ < 0.117 (ours: < 0.0975). Init θ = 0.1, so
    early clones of initial points die in the official code and survive in ours.
- **Measured** (2k-step bicycle w1600, black background, monkeypatched official precedence):
  - Gaussian count at step 600: 54.7k with the official precedence vs 78.8k with ours.
  - PSNR at 2k: −0.08 dB with the official precedence (see the table). The effect at 30k is unknown. Mirror the official code anyway, because it reproduces the paper's population dynamics.
- **Fix**: add an `official_d2a_precedence: true` default, or mirror the code literally with
  `1 - exp(-x)/0.99`. Keep the "correct" inverse only as an ablation. Update the
  `density-reparam` and `densify-rules` probes; they currently enforce the correct inverse and
  would reject the official behaviour.

### MAJOR 2 — background: the benchmark should use `background: black`
- **Where**: `renderer.py:145`.
- **Official code**: renders on black only (`inria_3dgs_volr.py:53` asserts it;
  `train_volr.py:43` uses black because `white_background` is false).
- **Why it matters more here**: density-based alpha leaves residual transparency, and Vol3DGS
  has no opacity reset to force saturation. Sky and edge pixels with T > 0 are biased toward white
  during training. This is the ConeGS situation, where matching black gained about 1 dB outdoors.
- **Recommendation**:
  - Add a registered variant (e.g. `vol3dgs-black`; spec config cannot be passed through
    `ss-train`) and benchmark that as the primary fidelity run.
  - Keep white as the runtime-convention row.
  - Report the gain against the white gsplat baseline, and note the background difference
    explicitly.

### MINOR
1. **Low-risk float differences**: `--use_fast_math` with `__expf` (`kernels.py:424`), and the
   NaN skip also skips inf. Fine.
2. **Step order**: the optimizer steps before densification (runtime-wide). In the official code
   the step comes after densify, so new Gaussians receive one step of zero moments. Negligible.
3. **Tile assignment**: gsplat's box culling (`base_radii > 0`, `renderer.py:163`) runs before
   our box. Both use the same formula; the projection is EWA with clamped Jacobian limits (the
   official slang code also clamps). Fine.
4. **2k smoke gap (bicycle 19.90 test / 20.03 vs gsplat 21.11)**: plausible at 2k and not a sign
   of a kernel bug. The forward and backward probes against the float64 dense reference pass.
   - At 2k the baseline has had no opacity reset yet either.
   - Vol3DGS starts from θ = 0.1, and its effective alpha is 0.099 at the centre, falling off along
     the ray.
   - Halved-density children add little coverage early.
   - The white background penalises a low-coverage model.
   - The paper itself reports Vol3DGS −0.22 dB below 3DGS-Slang on average.
   - Watch the 7k checkpoint. A gap of more than 1 dB at 7k would point to a real problem.

## 2k diagnostic (bicycle w1600, seed 0, this review; ≤2000 steps)
| variant | test PSNR | #G @2k |
|---|---|---|
| plugin as-is (white, smoke-1) | 19.90 | 436k |
| black background | 20.28 (+0.38) | 371k |
| black + official d2a precedence | 20.20 (−0.08 vs black) | 334k |

Script: `/tmp/claude-1013/-home-sejain/6ad4135e-056e-4bd4-9844-6b8764bea041/scratchpad/v3/run.py`
(spec override and monkeypatch only; no plugin edits).

Black recovers 0.38 dB of the 1.2 dB gap at 2k. The remaining ~0.9 dB is consistent with Vol3DGS's slower early growth (371k vs 502k Gaussians).

## Launch (after the fixes)
```
mkdir -p /bigdata/sejain/runs/vol3dgs-mipnerf360-1600-30000 && ln -s /bigdata/sejain/runs/vol3dgs-mipnerf360-1600-30000 runs/
python3 tools/bench.py launch --exp vol3dgs-mipnerf360-1600-30000 --methods vol3dgs-black vol3dgs \
  --dataset mipnerf360-1600 --scenes bicycle room --steps 30000 --recipe 3dgs --min-free-gb 24 \
  --extra "--num-gaussians 10000000 --max-gaussians 12000000"
```
(Put the run on /bigdata per CLAUDE.md. Compare against `baseline3dgs-mipnerf360-1600-30000`
(bicycle) and `-1600b-30000` (room).)
