# FreGS implementation plan (native SplatStudio)

- Method name: **`fregs`**
- Module: **`ss_fregs`** (pip package in `reproductions/fregs/plugin/`)
- Paper: FreGS: 3D Gaussian Splatting with Progressive Frequency Regularization (arXiv 2403.06908, CVPR 2024)
- Official code: none. Every value the paper omits is marked **assumed** (a written default) or
  **calibrated** (set by a procedure that uses only training views or Gaussian counts, never test images).
- `feasible_natively: true`. FreGS changes only the loss and one density hyper-parameter. The one schedule the runtime
  cannot express (the 1/4-resolution warm-up) is missing from the baseline too (gap G3).

## 1. Contributions → components

| Ci | Paper | Kind / component name | Factory | requires | provides | Config (value, status) |
|---|---|---|---|---|---|---|
| C1 Frequency regularization (Sec 3.2, Eq. 2–6) | amplitude + phase discrepancy in Fourier space | loss `fregs-frequency` | `ss_fregs.frequency:create_frequency_loss` | `("render.rgb",)` | `()` | see §3 |
| C2 Frequency annealing (Sec 3.3) | fixed low-pass D0 + dynamic high-pass band D0<r≤D_t, linear T0→T; L_f switches at T0 | same loss component, `anneal=True` | same | same | same | `d0_frac`, `t0`, `t_end`, `w_h` |
| C3 Count-matched densification (Sec 4.2, Table 1/2 captions) | raised view-space gradient threshold so #Gaussians ≈ 3DGS | density `fregs-density` | `splatstudio.backends.gsplat:create_density` (wraps gsplat `DefaultStrategy`) | `("gradient.screen.signed",)` | `()` | `grow_grad2d` = 0.00035 (calibrated, `evidence/threshold_calibration.json`; 0.0003 was the pre-calibration default), `absgrad: False` |
| C4 Frequency loss ends with densification (Sec 4.1) | L_f = 0 for t > 15000 | same loss component | same | | | `stop_iter = 15000` (**paper**) |

All other parts stay the **3DGS / gsplat baseline**: renderer `gsplat` (`splatstudio.backends.gsplat:create_renderer`,
config `{absgrad: False, antialiased: False}`, capabilities = `render.rgb, render.depth.expected_z, gradient.screen.signed`),
loss `photometric` (L1 + 0.2 D-SSIM for the whole run, as the paper states), DefaultStrategy defaults other than `grow_grad2d`
(refine 500–15000 every 100, opacity reset every 3000, prune_opa 0.005, grow_scale3d 0.01), Adam with the base learning rates,
means LR exponential decay, SH degree +1 every 1000 steps, 30000 steps, `--recipe 3dgs`.

Construction: `base = make_method("fregs")`, then replace `components[1]` (density) with `fregs-density` and append
`fregs-frequency` after `photometric`. Keep the base probes and contributions and add the FreGS ones (§5).

### Variants (separate entry points; `ss-train` has no config override flag)
| entry point | purpose | difference from `fregs` |
|---|---|---|
| `fregs` | the method (Base+FR+FA, raised threshold) | — |
| `fregs-fr` | ablation Base+FR (Table 2) | `anneal=False`: both bands from step 1 to 15000 with fregs' per-band weights, `w_l(d_la+d_lp) + w_h(d_a+d_p over r>D0)`. This deconfounds the ablation (review-1), so annealing is the only difference; the paper's single weight is the case w_l = w_h. Same `grow_grad2d` as `fregs` (the caption says Base+FR and Base+FR+FA use the same threshold) |
| `fregs-thr0002` | diagnostic: FreGS with the unchanged 3DGS threshold | `grow_grad2d=0.0002` (shows how much of the gain comes from the extra Gaussians) |
| `fregs-calib-gXXXX` | threshold calibration only (§6.2), XXXX ∈ {0003,0004,0005,0006} | `grow_grad2d` only |

Implement these with one helper `_make(name, **overrides)` in `ss_fregs/__init__.py` and expose each through a module-level function.
Every override is stored in the `ComponentSpec.config`, so it is part of the specification hash and run provenance.

## 2. State / fields

No new `FieldSpec`. FreGS adds no per-Gaussian attribute. gsplat's DefaultStrategy keeps the optimizer state aligned.
`fields: []`.

## 3. Loss: equations → code (`ss_fregs/frequency.py`)

Inputs: `rgb = outputs["rgb"]` (HWC, unclamped, white-composited, which is the image the photometric loss sees) and `target` (HWC).
Transpose to `[C,H,W]` and compute `F = torch.fft.fftshift(torch.fft.fft2(x, norm="backward"), dim=(-2,-1))` per channel.
`norm="backward"` is the unnormalized DFT of Eq. 2 and is torch's default.

Radius grid (built once per (H,W,device) and cached): `u = arange(H) - H//2`, `v = arange(W) - W//2`, `r = sqrt(u² + v²)`
in spectrum-index units, with the centre (H/2,W/2) as origin as the paper states. `D = r.max()` (the corner, i.e. "max range of the spectrum").
`D0 = d0_frac · D`.

Masks (ideal, hard, circular):
- `H_l = (r <= D0)`
- `H_h(t) = (r > D0) & (r <= D_t)`, with `D_t = D0 + (t - T0)(D - D0)/(T - T0)` clamped to [D0, D] for T0 < t. This takes the paper's inequality bound as an equality.

Discrepancy on a masked spectrum M (sum over the masked bins and over channels):
- `d_a(M) = (1/sqrt(HW)) · Σ_c Σ_{(u,v)∈M} | |F̂| − |F| |` (Eq. 5)
- `d_p(M) = (1/sqrt(HW)) · Σ_c Σ_{(u,v)∈M} | ∠F̂ − ∠F |` with `∠ = atan2(Im, Re)` (Eq. 6, raw difference with no wrapping)

Implementation: gather the masked bins with boolean indexing *before* taking `abs`/`atan2`. Bins outside the filter are exactly 0
for both images, so they contribute 0 to the loss (paper ambiguity 3). Gathering first also keeps their undefined `atan2(0,0)`
gradients out of the graph. Phase safeguard: in the phase term, exclude bins whose rendered amplitude is below
`phase_eps · sqrt(HW)` (default `phase_eps=1e-6`). Torch's atan2 gradient is 1/|F|, which is unbounded at zero amplitude.
The amplitude term keeps all masked bins. For the amplitude, compute `|F̂| = sqrt(Re² + Im² + 1e-20)` so its gradient is finite at 0.

Schedule (`t = outputs["step"]`):
```
if t > stop_iter:                L_f = 0 * rgb.sum()          # stays attached to the graph (check_loss requires_grad)
elif not anneal:                 L_f = w_l * (d_a(all) + d_p(all))                       # fregs-fr
elif t <= t0:                    L_f = w_l * (d_a(H_l) + d_p(H_l))
else:                            L_f = w_l * (d_a(H_l) + d_p(H_l)) + w_h * (d_a(H_h(t)) + d_p(H_h(t)))
```
The total objective is the runtime's sum: photometric + L_f. The paper does not say how L_f is added, so a plain weighted sum is assumed.

Config keys (all validated; unknown keys raise):

| key | default | status | justification |
|---|---|---|---|
| `w_l` | calibrated (§6.1); placeholder 2e-6 | calibrated (train views) | Not given. Fig. 2 (Bicycle) shows that at iteration 0, where only the low band is active, the mean pixel gradient with FR is ≈1.7× that without FR (0.43 vs 0.25 over-reconstructed; 0.405 vs 0.248 well-reconstructed). Choose `w_l` so that mean\|∂(L_photo+L_f)/∂rgb\| = 1.7 × mean\|∂L_photo/∂rgb\| on training views at the step-1 state. With the literal 1/sqrt(HW) sum, the per-pixel gradient is ~10⁵–10⁶× that of the L1 term, so an O(1) weight would make L_f dominate. The placeholder follows from this order-of-magnitude estimate. |
| `w_h` | calibrated (§6.1); placeholder 2e-6 | calibrated (train views) | Not given. Per-band unit-gradient equalization (review-2): `w_h = w_l · M_low/M_high`, where M_b is the pooled mean ‖∂(d_ba+d_bp)/∂means2d‖ at unit weight on the 15k train-view gsplat reference at t=T. The resulting whole-image t=T ratio is checked against Fig. 2's 12k–15k envelope (≈1.15 well to ≈2.3 over). A miss is a recorded deviation and is not retuned. Whole-image targets 1.15/1.5/2.0 are diagnostics only. |
| `d0_frac` | 0.10 | assumed | Not given. The low band should carry only "global patterns and structures" (Sec 3.3). r ≤ 0.1·D keeps ≈0.8% of the bins, i.e. structures larger than ~15–20 px at the benchmark resolutions. This conventional ideal-low-pass cut-off leaves most of the spectrum to the annealed band. |
| `t0` | 3000 | assumed | Not given. Must be > 500 (densification start) so that the low-frequency-only phase actually drives early densification ("coarse-to-fine densification"). 3000 is the first opacity reset and the step at which the SH degree reaches 3. Coarse geometry is set by then. |
| `t_end` (T) | 15000 | assumed (bounded by paper) | FR ends at 15000 (paper), so T ≤ 15000. T = 15000 spreads the low-to-high progression over the entire densification window. The full spectrum is reached exactly when densification and FR stop. |
| `stop_iter` | 15000 | **paper** (Sec 4.1) | "frequency regularization terminates once Gaussian densification ends". |
| `anneal` | True | paper (FA on) | False gives the Base+FR ablation. |
| `mask` | `"ideal_circular"` | assumed | "range" and "max range of the spectrum" with the centre as origin suggest a radial distance. Ideal (hard) masks, because no filter profile is given. |
| `radius_units` | `"index"` | assumed | Literal origin (H/2,W/2) in spectrum-index coordinates. The alternative `"normalized"` (u/(H/2), v/(W/2)) is available for sensitivity checks only. |
| `channels` | `"per_channel_sum"` | assumed | I ∈ R^{H×W×C}, so per-channel FFT summed over C. Mean vs sum only rescales and is absorbed by the w calibration. |
| `phase` | `"wrapped"` (iteration 1; i0 used `"raw"`) | assumed | Eq. 6 does not say whether to wrap. i0 used the plain atan2 difference (`"raw"`). i1 uses angle(F_hat conj F) in (-pi, pi] (`"wrapped"`) to remove 2*pi jumps (G5). `"arctan"` exists for sensitivity only. |
| `phase_eps` | 1e-6 | assumed (numerical) | Not in the paper. Needed because the phase gradient diverges at zero amplitude. |
| `normalization` | `"paper"` (1/sqrt(HW)·Σ, backward FFT) | paper | Literal Eq. 5/6. |

## 4. Density: `fregs-density`

`ComponentSpec("fregs-density", "density", "splatstudio.backends.gsplat:create_density", ("gradient.screen.signed",), (),
{"absgrad": False, "grow_grad2d": <calibrated>})`. Built-in `Density` wraps `DefaultStrategy(**config)`. Its `requires`
({gradient.screen.signed}) equals the spec's `requires`. Only the threshold changes, which is a method change (C3) marked
**calibrated**. Default before calibration: 0.0003 (assumed ≈1.5× the 3DGS 0.0002, which is the average with-FR/without-FR
gradient ratio across both regions in Fig. 2). No custom density code is needed. The paper describes the densification
improvement as coming only through the gradients of L_f.

## 5. Probes (`ss_fregs/probes.py`)

| probe | contribution | what it measures / why it fails if Ci is removed | cuda |
|---|---|---|---|
| `freq-oracle` | C1 | On a random 24×40×3 pair, compare the component's L_f value and ∂L_f/∂rgb (autograd) to an **independent** implementation. The oracle uses an explicit DFT matrix (Eq. 2 via `exp(-2πi(ux/H+vy/W))` einsum in float64, no torch.fft), separately built masks (python loops over (u,v)), and per-bin amplitude/phase. Checks at a step ≤ T0 and a step in (T0,T). Also checks identical images ⇒ L_f=0 and zero gradient, and that L_f>0 and gradient ≠ 0 for a different pair. Removing or replacing the loss (e.g. by a spatial L1) breaks the value/gradient match. | no |
| `freq-annealing` | C2, C4 | Known frequency content: target = base image + a sinusoid at radius r* (D0 < r* < D) and rendered = base. Asserts (a) the high-frequency discrepancy is 0 for t ≤ T0 and for t with D_t < r*, and > 0 once D_t ≥ r*; (b) the passed-bin count of H_h(t) matches an independently counted annulus at t ∈ {T0+1, (T0+T)/2, T}; (c) a low-frequency sinusoid (r < D0) gives low term > 0 at t=1; (d) L_f == 0 with a graph attached at t = 15001. Fails for Base+FR (which sees r* at t=1), with no stop, or with a wrong band formula. | no |
| `fregs-threshold` | C3 | Builds the density component from the spec. Asserts `strategy.grow_grad2d` equals the calibrated value and is > 0.0002, `refine_stop_iter == 15000`, `absgrad False`. Functional part on a 64-Gaussian fixture with allocated Adam state: set `state["grad2d"]/state["count"]` so that the average gradient lies in (0.0002, grow_grad2d), call `step_post_backward` at a refine step, and assert no growth. The same fixture with a base `DefaultStrategy()` must grow (self-check). | yes (gsplat ops on cuda) |
| `fregs-gradient-gain` | C1 mechanism (Fig. 2) | Tiny synthetic scene (~2k random Gaussians, 64×64 camera, target rendered from perturbed Gaussians). Computes mean \|∂L/∂rgb\| and mean ‖∂L/∂means2d‖ with photometric only and with photometric + L_f (calibrated weights, t=1). Asserts both ratios exceed 1.2. Fails if L_f is removed or its weight is 0. | yes |
| base: `state-transitions`, `photometric-gradient`, `renderer-gradient` | baseline | unchanged | |

Contributions: `frequency-regularization` → (`freq-oracle`, `fregs-gradient-gain`); `frequency-annealing` → (`freq-annealing`);
`count-matched-densification` → (`fregs-threshold`); plus the base contributions.

## 6. Calibration (no test images; results go to `evidence/`)

### 6.1 Loss weights (`ss_fregs/calibrate.py weights`, output `evidence/weight_calibration.json`)
For each of the 9 Mip-NeRF 360 scenes, use `splatstudio.data.load_dataset` and only `splits["train"]`:
1. `w_l`: build the step-1 state exactly like `runtime.train(recipe="3dgs")` (initial_points with 10M cap, knn_log_scales, SH degree 0).
   Render 16 evenly spaced training views. Bisect `w_l` so that mean\|∂(L_photo + w_l(d_la+d_lp))/∂rgb\| / mean\|∂L_photo/∂rgb\| = 1.7.
2. `w_h` (revised, review-2): use a gsplat reference trained on training views for 15000 steps of a 30000-step schedule (state ≈ t=T;
   cached in `runs/fregs-calib-reference/`). Render the same training views and set `w_h = w_l · M_low/M_high` from the unit-weight
   per-band screen-gradient magnitudes at t=T. Record the whole-image ratio, Fig. 2 envelope membership, and diagnostic solves at 1.15/1.5/2.0.
   `solve()` raises (recorded as 'unidentifiable') instead of returning None.
3. Take the **median over the 9 scenes** for each weight and use it as a single global value (the paper presumably used one value per dataset).
   Write it into the `fregs` spec defaults and record per-scene values and the ratio targets with their Fig. 2 provenance.
   Label: "calibrated from Fig. 2 gradient ratios on training views".

### 6.2 Densification threshold (count matching, `evidence/threshold_calibration.json`)
Densification stops at 15000, so N(15000) = final N. Run `gsplat` and every candidate (`fregs-calib-g0003…g0006`) in **one**
sweep with `--steps 15000` on {bicycle, garden, room}, so steps, means-LR schedule, seed and recipe are the same for all
(revised per review-1: do not compare with the 30k baseline's step-15000 count). Read **only** `gaussians` at step 15000 from
`training.jsonl`. **Blinding is non-destructive (review-1):** nothing is deleted, because bench needs test.json to mark a job done.
The script never opens validation.json/test.json/.eval.json. It records their sha256 in a `BLINDED.json` marker in each run dir and in
`runs/<exp>/`, and in the report. `bench summarize` must not be run on this experiment. Pick the zero of the mean log-ratio
N_fregs/N_gsplat, interpolating linearly between grid points to 1e-5, and confirm it on the 3 scenes. Label: "calibrated (Gaussian-count
matched, no image metrics)". Then fix it in all variants except `fregs-thr0002`.
(6.1 revised per review-1 and review-2: ratios are measured on the per-Gaussian screen-space gradient norm. w_l comes from the 1.7 target at
step 1; w_h from per-band equalization on the 15k train-view reference. See plugin/README.md. `calibrate_threshold.py` refuses to run until
CALIBRATED w_l/w_h are True, checks that every candidate run was trained with those weights (from its config.json), and records them.)
Order: run 6.1 before 6.2, because the threshold depends on the loss weights.

## 7. Benchmark protocol
- `tools/bench.py launch --exp fregs-mipnerf360-30000 --methods fregs --dataset mipnerf360` on **all 9 scenes**, 30000 steps, `--recipe 3dgs`.
  Compare with the existing `baseline-mipnerf360-30000` (gsplat), using the same seed and protocol.
- Report the dataset average only over all 9 scenes. The primary target is ΔPSNR vs our gsplat baseline compared with the paper's
  Δ = +0.64 dB (27.85 − 27.21), ΔSSIM +0.011, ΔLPIPS −0.005, plus absolute numbers. Also report #Gaussians vs baseline.
- Ablations (if budget allows, all 9 scenes): `fregs-fr` (paper Δ vs Base: +0.42 dB) and `fregs-thr0002`.
- Optional extras: tandt (truck, train; paper Δ +0.82) and db (drjohnson, playroom; Δ +0.52).
- Sensitivity checks (`d0_frac`, `t0`, `phase`), only if the gap analyst requests them: evaluate on a **train-split hold-out**
  (e.g. every 8th *training* image held out), never on the test split. Label any change "tuned (train hold-out)".

## 8. Plugin file list (`reproductions/fregs/plugin/`)
- `pyproject.toml`: package `splatstudio-fregs`, dependency `splatstudio>=0.1,<0.2`. Entry points `fregs`, `fregs-fr`, `fregs-thr0002`,
  `fregs-calib-g0003`, `-g0004`, `-g0005`, `-g0006` under `splatstudio.methods`.
- `ss_fregs/__init__.py`: `method()`, the variant functions, `_make(name, **overrides)`, and the ProbeSpecs/ContributionSpecs. Citation: arXiv 2403.06908.
- `ss_fregs/frequency.py`: `create_frequency_loss(spec)`, `radius_grid(H,W,device)` (cached), `band_radius(t, cfg, D)`, `discrepancy(F̂, F, mask, eps)`.
- `ss_fregs/probes.py`: the 4 probes in §5 (the oracle is self-contained and must not import `frequency.py` helpers).
- `ss_fregs/calibrate.py`: CLI `python -m ss_fregs.calibrate weights|threshold ...` (§6). It uses only train splits and training logs.
- `tests/test_frequency.py`: pytest wrappers around the CPU probes, plus a shape/odd-size (H or W odd) case.

## 9. Implementation notes / pitfalls
1. Odd H or W: use `fftshift` with `u = arange(H) - H//2`, which matches torch's fftshift centre. Test odd sizes in the oracle.
2. Do not clamp `rgb` before the FFT (the photometric loss also uses unclamped rgb).
3. Cost: one FFT of 3×~1M px for each of the two images per step. Cache the masks per (H,W,step-band), recomputing H_h only when ⌊D_t⌋ changes.
4. Use float32 FFT in training. Check that the probe tolerance (rtol 1e-4 vs the float64 oracle) holds.
5. After `stop_iter`, return `rgb.sum()*0` so that `check_loss` (requires_grad) passes and logging stays stable.
6. The loss logs as its own key in `training.jsonl` (`losses["fregs-frequency"]`). The coder should confirm that its magnitude after calibration is not orders above the photometric loss in gradient terms (the probe already checks this).

## 10. Fidelity gaps

| id | what | why | expected effect | closable later? |
|---|---|---|---|---|
| G1 | `w_l`, `w_h` unknown; calibrated from Fig. 2 gradient ratios | paper and arXiv source give no values and there is no supplementary or code | **High**: the frequency-to-photometric balance controls both quality and densification. A wrong scale can erase the gain or create artifacts | Yes, if authors release code/values. A train-hold-out sweep (labelled tuned) could refine it |
| G2 | `d0_frac=0.1`, `t0=3000`, `t_end=15000` assumed | not given | Medium: affects the coarse-to-fine schedule (FA was worth +0.22 dB on Mip-NeRF 360 in the ablation) | Same as G1 |
| G3 | no 1/4-resolution warm-up (first 500 iterations) | native runtime trains at full resolution from step 1 and has no resolution schedule. A loss component cannot change render resolution. To our knowledge the official 3DGS code doesn't implement this warm-up either, even though the paper text describes it | Small. It affects **baseline and method equally** (our gsplat baseline doesn't warm up either), so the Δ comparison is unaffected. Densification starts at 500, so early densification sees full-res gradients in both | Needs a runtime extension (resolution schedule in the trainer) |
| G4 | densify threshold calibrated by count matching on 3 scenes under a **15000-step schedule**, then transferred to 30000-step runs | value not given; paper only says "similar number of Gaussians". A like-for-like gsplat baseline needs the same schedule | Medium: more Gaussians usually means higher PSNR. The transfer assumes N_fregs/N_gsplat is schedule-invariant, which is untested; the 30k runs have a higher means LR during densification | Partly. Check N_fregs/N_gsplat in the 30k benchmark runs (training.jsonl only), or calibrate on all 9 scenes |
| G5 | phase definition: raw atan2 difference with no wrap, amplitude-floor mask `phase_eps` | Eq. 3 writes arctan(Im/Re). No handling of wrap or of the zero-amplitude singularity is described | Medium: a raw difference can be ~2π for nearly identical complex values near the ±π cut, so the phase term is noisy | Sensitivity variant (`phase="wrapped"` (the i1 default; i0 used `"raw"`)) on a train hold-out only |
| G6 | mask shape: ideal circular mask in index units; D = corner radius | unstated | Low–medium | Sensitivity only |
| G7 | channel handling: per-channel FFT summed | unstated | Low (absorbed by the weight calibration) | — |
| G8 | paper's baseline row is the 3DGS paper's numbers, not retrained | Table 1/2 | Absolute comparison is confounded. Compare Δ vs our gsplat baseline to the paper's Δ | — |
| G9 | native protocol differences (white background, random-with-replacement view sampling, valid-conv SSIM, LPIPS-VGG via `lpips`, gsplat vs the original 3DGS CUDA rasterizer) | runtime | Affects baseline and method equally. Small absolute offsets | See `reproductions/_calibration/` (not present yet) |
| G10 | single seed; paper seeds unstated | budget | ±0.05–0.1 dB per scene noise on a +0.64 dB effect | Add seeds |
| G11 | L_f added as a plain sum | paper never writes the total loss | Low (the weights absorb it) | — |
| G12 | only dataset averages exist, so no per-scene comparison is possible | paper | Diagnosis at scene level is impossible. All 9 scenes are mandatory for the average | — |

## Evidence log
- 2026-11-02, probe re-registration after weight calibration. The original bars were registered
  before calibration on the synthetic probe scene: `fregs-high-band-gain` required the high band's
  increment to be >= 5% of the low band's, and `fregs-gradient-gain` required a t=1 means2d ratio
  > 1.2. With the calibrated weights (w_l = 3.400926541615607e-06, w_h = 5.568719487276853e-07,
  `evidence/weight_calib_v2/merged.json`) both **failed** on the synthetic scene: share 3.48%
  (0.00534 vs 0.1536) and ratio 1.154. The synthetic scene's band ratio M_low/M_high is about 2.5,
  against about 0.31 on the real scenes. The two bars were replaced by (a) ablation-contrast
  mechanism probes on the synthetic scene: the high-band gain over w_h = 0 must exceed 10x the
  float noise floor, and the t=1 ratio must be > 1.05 while the no-op ablation gives 1.000; and
  (b) a real-data probe `fregs-calibration-ratio` (garden, 4 train views, cached 15k reference):
  t=1 ratio 1.7 +- 0.15 and t=T ratio in [1.15, 2.3]. It reports unverified when the files are
  absent. The failed results are kept here and are not hidden.
- 2026-11-02, threshold calibration: grow_grad2d = 0.00035 (`evidence/threshold_calibration.json`; count-matched to gsplat at step 15000
  in the same 15k sweep on bicycle/garden/room; mean log-ratios +0.2125/-0.240/-0.617/-0.925 at 0.0003/0.0004/0.0005/0.0006,
  linearly interpolated). Written into DEFAULTS with CALIBRATED["grow_grad2d"] = True. The plugin is frozen for the benchmark.

- 2026-11-03, iteration 1 (fix iteration allowed by the user). (1) Runtime: SplatStudio's gsplat
  `Density` now performs the 3DGS opacity reset (`opacity_reset`, default True). gsplat 1.5.3's
  DefaultStrategy never reset opacity because of an operator-precedence bug. fregs-density
  inherits the reset; the new shared baseline `baseline3dgs-mipnerf360-30000` has a mean of 27.28.
  (2) Plugin: `phase` defaults to `"wrapped"`, i.e. Eq. 6 with the difference wrapped to (-pi, pi]
  as angle(F_hat conj F), with bins skipped when |F| or |F_hat| < phase_eps sqrt(HW). Grounding:
  Eq. 6 is ambiguous about wrapping, and raw-difference 2*pi jumps were the main numerical risk
  (G5, i0 hypothesis 6). `"raw"` (the i0 behaviour) and `"arctan"` remain available.
  `freq-oracle` wraps independently and tests fail on a raw/wrapped mix-up.
  **Not recalibrated:** w_l, w_h and grow_grad2d keep their i0 values. The threshold 0.00035 was
  count-matched against the no-reset gsplat in the 15k sweep (gsplat 1.5.3 without opacity reset),
  so the count match may not hold with the reset.
