# Review 1 — gabsplat (3DGabSplat, arXiv 2508.05343)

**Verdict: approved for benchmark** (0 critical, 0 major code defects, 5 minor / fidelity notes).
The implementation follows the most literal reading of the paper. The F=2 symmetry is a real fidelity risk, but it comes from the paper's own text, not from a coding error.

Checks run: read paper main + supp. (LaTeX), all plugin code and gsplat 1.5.3 `DefaultStrategy`; analysed `evidence/smoke-1/checkpoint.pt` (garden, 2000 steps, 774k Gaussians). No training runs were started.

## 1. Identical-init symmetry (F=2 ≈ F=1)

- Smoke checkpoint at step 2000: **86.3 %** of rows have `gabor_freq[:,0] == gabor_freq[:,1]` exactly, and 90.1 % have identical weights. In the remaining rows the kernels drift apart by up to |Δf| = 4.96. The only source of that asymmetry is float noise from the order of `gpuAtomicAdd`/warp reductions.
- With the two kernels identical, the modulation is m = 1 − 2ω(1 − cos φ). That is exactly F=1 with the weight range widened to [0, 2).
- **The paper does not specify any symmetry breaking.** Main text Sec. 3.4.1 (tex l.312) and Sec. 4.1 (l.516) say "initialize **all** frequencies and their corresponding weighting coefficients to 0.001 and 0.01". Supp. B (Supplementary.tex l.662) gives one scalar per dataset (0.05 for Mip-NeRF 360). There is no mention of random values, orthogonal axes, per-kernel values or per-component signs.
- Two passages *imply* the kernels differ: "F **distinct** Gabor kernels" (l.261) and "assigned distinct frequencies" (l.263). The F-ablation (Tab. 4: F=1 27.739, F=2 27.845, memory +32 MB) also suggests the kernels end up different. Neither passage gives a mechanism.
- The paper's own resets **make the symmetry worse**:
  - The child reset (supp. l.663; `density.py:451-452`) sets both kernels of every new child to f = 0.001. Most of the final population is children.
  - The weight reset every 3000 steps (`density.py:467`) sets both weights equal again.
  - Nothing in the paper breaks the symmetry.
- **Most literal reading = the current constant init.** Keep it for the headline run. Record the asymmetric-row fraction at 30k in report.md.
- Recommended diagnostics (labelled, not the headline):
  - (a) `gabsplat-symbreak`: per-kernel init f₀·s, where s is a fixed sign pattern per component (e.g. kernel 1 = (+,+,+), kernel 2 = (+,−,+)), applied to both the initial and the child-reset values.
  - (b) F=1.
  - Caution: flipping the sign of the whole vector does **not** break the symmetry, because cos is even, so ±f evolve as mirror images.

## 2. Main text vs supplementary conflicts

- **Initial frequency: use supp. B (0.05 for Mip-NeRF 360).** Supp. B is dataset-specific, and the main text itself defers to it ("A more detailed parameter configuration can be found in the supplementary material", l.516). Little rides on this choice: Adam with lr 0.02 moves f by up to 0.02 per step, and the smoke run already has median |f| = 1.95 at 2k.
- **Learning rates: unresolved.** The freq/weight lrs are an outright swap (main 0.01/0.02, supp. 0.02/0.01) and both texts call their values "default". The plugin's supp. choice (`__init__.py:33-34`) is defensible. Suggest one labelled garden run with lr_f 0.01 / lr_w 0.02.
- Plugin choices that match supp. B: opacity lr 0.025, prune threshold 0.01, opacity reset value 0.02, child reset values 0.001/0.01.

## 3. Projection and kernel math

- **Frequency projection (`frequency.py:173-179`) is exact.** Partition S = Σ''⁻¹ and let v = Σ''f''. Then f''_xy − S_xz/S_zz·f''_z = (S_aa − S_ab S_ba/S_bb)·v_a = (Σ''_2×2)⁻¹·(J₂Σf). This does not depend on the third row of the Jacobian, which is why the two completions agree in the probe (error 7e-18). The einsum `nij,nkj,nfk->nfi` computes M Mᵀ f = Σf correctly, and J₂ = J_gsplat·R_cam is correct (translation does not affect frequencies).
- **Minor — dilated conic (`frequency.py:177-179`; conic from `renderer.py:232`).** The code uses (Σ2d + 0.3 I)⁻¹, while Eq. 10 takes S from the *undilated* 3×3 Σ''⁻¹. The literal value is Σ2d⁻¹·J₂Σf. The difference factor is σ²/(σ² + 0.3), which is 0.77 at σ = 1 px. The dilated form is arguably the physically consistent one (convolving a Gabor with a 0.3 px² Gaussian gives exactly this frequency). Keep it as gap 4, or switch to the undilated conic for literalness: Σ2d⁻¹ = (J₂ΣJ₂ᵀ)⁻¹, computed in torch.
- **Depth-axis amplitude factor: dropping it is literal.** Paper Eq. 13 and supp. Eq. (final ĝ) contain no factor; the supp. sets C₁C₂ aside. Supp. C₂ = exp(+(2πf_z)²/2S₂₂) also has a sign error: the correct factor is exp(−…), i.e. a damping term.
- **Kernel (`gabor_raster.cu:39-52, 235-255`) is correct.**
  - Gaussian × [(1 − Σω) + Σω cos φ] is multiplicative on the envelope, as in Eq. 12/13.
  - φ = 2π f2d·(μ − p); the sign of δ does not matter because cos is even.
  - Weights are sigmoid-activated in `renderer.py:239`.
  - Backward terms: dm/dω = cos − 1, dm/dφ = −ω sin φ, v_δ += Σ dφ·f, v_f = dφ·δ.
  - Backward matches the dense float64 autograd reference to about 4e-7.
- **Minor — Σω is not bounded by 1.** With F=2, sigmoid allows Σω up to 2; 4.3 % of rows already have Σω > 1 at 2k. Then 1 − Σω < 0 and m can reach 1 − 2Σω < 0. Negative-alpha pixels are dropped by the 1/255 gate (`cu:52`), so they get no gradient. The paper claims the coefficients "sum to 1" (l.261), but its sigmoid does not enforce that either. This is literal behaviour; just note it.
- **Minor — saturation gate `opac·G·m ≤ 0.999` (`cu:235`).** This is gsplat behaviour, not Inria's, the same as our baseline (TODO-runtime). Since o·G·m ≤ 1 there is no need for a straight-through cap.

## 4. Density

- **`after_backward` (`density.py:439-469`) matches gsplat 1.5.3 `step_post_backward` line for line**, with these additions:
  - The child rows are taken as the last n_dupli + 2·n_split rows. This is correct: gsplat duplicates first, then `split` removes the parents and appends 2·n_split children, giving the order [kept originals, duplicates, split children].
  - The children are reset **before** pruning, so the row indices are still valid.
  - The Adam moments of the reset rows are zeroed.
  - Opacity is clamped with `reset_opa` to ≤ 0.02 at steps 3000…12000 (`resets_opacity`, < 15000).
  - All weights are *set* to 0.01 at the reset (literal, supp. l.663: "reset to 0.01"). Frequencies are not reset periodically, which follows supp. B. The ablation label "w/o periodic freq reset" (Tab. 5) is loose naming.
- **requires_grad pitfall: avoided.** Writes go through `.data` in place (`density.py:405`), gsplat ops keep `requires_grad`, and `check_trainable` (`density.py:412-418`) runs after every population change.
- The opacity lr of 0.025 is applied in `initialize`, after the runtime builds the optimizers (`runtime.py:154`). No later code overrides it.
- **Minor — smoke result is confounded.** The smoke run (23.03 vs baseline 23.07 at 2k) differs from the baseline in opacity lr (0.025 vs 0.05), so it says nothing about the Gabor contribution. Compare at 30k against the paper's +0.44 dB delta (garden 27.66 vs 3DGS* 27.43, room).

## Probes

- **Strong:** forward and backward vs the dense float64 reference, Eq. 10 checked against numerical z-integration with two Jacobian completions, and the density probe, which has a no-reset ablation check. The density probe fails if the child reset is removed.
- **Weak:** `recipe` only compares constants.
- **Missing:** no probe or log reports whether the two kernels have diverged. Add the identical-row fraction to the diagnostics.
