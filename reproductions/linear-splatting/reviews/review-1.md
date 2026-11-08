# Review 1 — 3DLS (arXiv 2411.12440), plugin `ss_linear_splatting`

**Verdict: approved for benchmark with `3dls-aags` as primary (labelled interpretation).**
Critical 0 · Major 1 (documentation/claim, not code) · Minor 4.
No plugin code was edited. Reviewer scratch experiments live in
`/tmp/claude-1013/-home-sejain/6ad4135e-056e-4bd4-9844-6b8764bea041/scratchpad/lsrev/`.

## 1. The AGS question

### What the paper says (main.tex)
- Eq. 11 (l.355–358): `D'_M = sqrt((x'-μ')ᵀ Σ_2D⁻¹ (x'-μ'))`. This is the **unsquared 2-D Mahalanobis distance in σ units**,
  computed from pixel offsets and the pixel-space Σ_2D. It is defined **before** DA, which comes later (Eq. 12, l.380:
  `L = max(0, 1 − D_M/λ)`).
- Eq. 13 (l.430): `ω(D'_M) = exp(−D'_M²)`, "the Mahalanobis distance from the kernel center in 2D space". It is not divided by λ.
- Eq. 14 (l.436): `Δθ ∝ −∇_θ L'(x') · ω(D'_M)`. ω multiplies the per-pixel **kernel** derivative, and the paper describes it as a
  rule for the *parameter update*. The paper does not say whether the densification statistic sees ω.
  Sec. 4.1 (l.466) says AGS is implemented inside the custom CUDA backward on gsplat v1.3. In gsplat v1.3 the single
  `v_means2d` buffer feeds both Adam and `DefaultStrategy`, so the most natural implementation **does** scale the statistic.
- **Fig. 3b (`figures/gradient_scaling.pdf`) settles the form of ω.** I digitised it at 300 dpi:
  - The native gradient is ±0.400 = 1/λ over |x| < 2.50, so the x-axis is the *unaligned* D and support ends at λ.
  - The scaled curve at x = −2, −1.5, −1, −0.5 reads 0.0074, 0.042, 0.145, 0.306.
  - `0.4·exp(−D²)` predicts 0.0073, 0.042, 0.147, 0.312, which matches.
  - `exp(−D²/2)` predicts 0.054, 0.130, 0.243, 0.353, and `exp(−(D/λ)²)` predicts 0.211, 0.279, 0.341, 0.384. Both are ruled out.

  **So the plugin's literal ω (`linear_raster.cu:219-221`, c = 1, unaligned D, applied to dk/dD only) is exactly what the paper
  writes and plots.** Our reading of D is not the source of the failure.
- **The caller's premise about 0.006 is wrong.** In supp. Table 6 / main `tableParams`, 0.006 is τ_G3D (`grow_scale3d`, the
  clone/split boundary), not the gradient threshold. The gradient threshold τ_G∇ stays **0.0002**, the same as 3DGS.
  The "3DGS" row of that table is exactly gsplat's `DefaultStrategy` defaults. So the paper's threshold settings do not
  compensate for ω, and the normalisation (|∇means2d|·W/2, H/2) is gsplat's, unchanged in v1.3 (checked against the v1.3.0
  source; v1.3 also resets opacity correctly to 2·prune_opa = 0.05, which the runtime fix reproduces).

### Does the failure come from AGS feeding densification? No: I tested it.
These are 2k-step garden smokes (recipe 3dgs, same seed and init). The scratch renderers reuse the plugin's CUDA kernel twice
with straight-through combination; the forward pass is identical.

| variant | ω on | densify statistic | N @2k | loss @400 (pre-densify) | loss @2k | test PSNR |
|---|---|---|---|---|---|---|
| `3dls` literal (builder) | means2d + conic | scaled | 101k | 0.164 | 0.135 | 20.93 |
| **literal ω, unscaled statistic** (reviewer) | means2d + conic | **unscaled** | 910k | 0.164 | 0.117 | **21.09** |
| **literal ω on conic only** (reviewer) | conic | unscaled | 786k | 0.163 | 0.127 | **21.10** |
| `3dls-aags` ω = exp(−(D/λ)²) | means2d + conic | scaled | 417k | 0.145 | 0.100 | 22.80 |
| `3dls-noags` (LK+DA) | – | – | 836k | 0.145 | 0.099 | 22.84 |
| gsplat baseline | – | – | 840k | 0.155 | 0.104 | 23.07 |

- Restoring densification (910k Gaussians, more than noags) recovers only +0.16 dB.
- The literal ω is already worse at step 400, before any densification. Adam is invariant to the gradient's global scale,
  so the damage comes from the *spatial* reweighting: ω < 0.02 for D > 2, so the outer ring (1 < D < 2.5, 84 % of the
  footprint) gives almost no geometry signal, while opacity and colour still learn from the full 2.5σ support.
- **Conclusion:** the literal Eq. 13 cannot be made to work just by changing its scope with respect to densification. It is
  incompatible with a λ = 2.5 support, whatever "most literal" scope one picks.
- The paper's Table 2 says AGS gives "+0.03 dB kitchen, +0.42 train, N × 0.57". The most plausible explanation is that their
  code evaluates ω on the distance the kernel actually uses after DA (e.g. conic pre-scaled by 1/λ²). `3dls-aags` is that
  variant: its N ratio is 0.50 against the paper's 0.57, and it does not degrade the optimisation. This is an inference, not
  something the paper states, and Fig. 3b contradicts it.

**Recommendation:** primary = `3dls-aags`, labelled "interpretation: ω on λ-aligned D (Fig. 3b plots unaligned)".
Also benchmark `3dls-noags`, the paper's LK+DA row, which is the cleanest comparison: at 2k AGS gives no PSNR gain over it.
Report `3dls` literal as a failed literal row (one scene is enough).

## 2. Issues

**Major M1 — `__init__.py:12-17`, `plan.md` post-smoke amendment, `report.md` (to be written).**
- The justification blames densification ("suppresses the densification gradient ~100×, which cannot give Table 2 counts").
  My scope-ablation shows densification is a secondary effect. The primary failure is the optimisation itself (pre-densify
  loss gap, and 21.1 dB even with 910k Gaussians).
- The docstring and report should say three things:
  1. Fig. 3b confirms ω = exp(−D²) on unaligned D.
  2. Literal ω fails whether or not it feeds densification (evidence: the table above).
  3. `aags` is therefore an inferred implementation detail, not a reading of the text.
- Also correct "0.09 % vs 10.5 %": the *mean* statistic ratio is 8.5× (ags_gradstat.json 1.06e-5 vs 8.96e-5). The 100× is
  the fraction above threshold, not "the gradient".
- Fix: amend the text and copy the reviewer table into `evidence/`. No code change.

**Minor**
- m1 `__init__.py:39-40`: `grow_scale2d`/`prune_scale2d` are set but inert, because `refine_scale2d_stop_iter = 0`. This matches
  gsplat v1.3 defaults and therefore the paper. Note in the report that τ_G2D/τ_P2D are inactive in both.
- m2 `renderer.py:169-176`: the radius `λ(1 − 1/(255 o))·sqrt(Σ_xx)` is correctly derived (o·(1−D/λ) ≥ 1/255 ⇔ D ≤ λ(1−1/(255o));
  the axis-aligned extent of the ellipse D ≤ e is e·sqrt(Σ_ii)). The 3.33σ pre-cull from `fully_fused_projection` is a
  superset for λ ≤ 3.33, which is guarded at `renderer.py:231`. No issue; noted for the record.
- m3 `linear_raster.cu:217`: the gate `o·k ≤ 0.999` is gsplat's saturation gate, kept consistently with gsplat v1.3, which the
  paper used. It drops geometry and opacity gradients only for o > 0.999 near the centre (k ≈ 1). This is a smaller exposure
  than for Gaussians at equal opacity. Acceptable and faithful to the paper's base; mention it in the report (TODO-runtime item).
- m4 SSIM in the loss uses valid convolution (runtime gap, already listed). The loss weights `losses.py:29-33` = supp. Eq. 1–4
  (0.6·mean|·| + 0.2·mean(·)² + 0.2·(1−SSIM)) are exact.

## 3. Equation coverage
| eq | status | code |
|---|---|---|
| (9)/(10) D_M, Gaussian form | exact | `linear_raster.cu:35-38` (σ = ½dᵀQd, D = √(2σ)) |
| (11) L' = max(0, 1−D') | exact (λ = 1 limit) | `linear_alpha` |
| (12) DA, D/λ, λ = 2.5 | exact | `linear_raster.cu:39`, config `lam` |
| (13) ω = exp(−D'²) | exact in `3dls`; deviates (c = 1/λ²) in `3dls-aags`, labelled | `linear_raster.cu:220`, `__init__.py:127` |
| (14) scale kernel derivative | exact (geometry path: means2d + conic; opacity/colour unscaled) | `linear_raster.cu:217-227` |
| supp. Eq. 1–4 loss | exact | `losses.py` |
| supp. Table 6 thresholds | exact (0.0002/0.006/0.4/0.025) | `__init__.py:34-41` |

Backward check, operator by operator: dα/dσ = −o/(λD), and `v_xy = v_σ·Q·δ`, `v_conic = (½v_σδx², v_σδxδy, ½v_σδy²)` with
δ = μ − p, as in gsplat. The D → 0 clamp is harmless because δ → 0. `reference.py` is an independent dense float64
autograd reference.

## 4. Probes
- `linear-forward-vs-dense` / `linear-backward-vs-dense`: these compare against a dense float64 autograd reference and also
  require a difference from the Gaussian render, so they would fail if the kernel reverted. **Detect removal: yes.**
- `alignment-footprint`: checks support radius = λσ for λ = 2.5 and λ = 1, plus no tile clipping. **Yes.**
- `ags-gradient-scaling`: compares against the reference with the configured c, and checks that opacity/colour gradients are
  unscaled. **Yes.** It verifies c as configured, not which c is "right"; that judgement is covered by §1.
- `loss-weights` (independent formula plus a dropped-term check) and `density-thresholds`: **yes.**
- Unit tests: 5 passed (re-run by the reviewer).

## 5. Other checks
- **requires_grad pitfall:** not applicable. There are no custom density ops; the plugin uses the runtime gsplat density with
  `gsplat.strategy.ops`.
- **Baseline unchanged:** projection, SH, tiling and densification are gsplat's. The only intended changes are the kernel,
  λ-radius, loss and thresholds. White background, as the runtime requires.
- **Packaging:** three entry points with names matching `MethodSpec.name`; verify passed for all three (evidence).
