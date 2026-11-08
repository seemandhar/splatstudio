# Report — Denoising-GS (arXiv 2605.14880), plan #115

**Outcome: partial.** Counter is within the ±0.5 dB target. Bicycle misses it, and the gap comes from our 3DGS-MCMC backbone, not from the Denoising-GS additions.

| scene | ours | paper | Δ |
|---|---|---|---|
| counter | 29.36 | 29.57 | −0.21 |
| bicycle | 25.43 | 26.17 | −0.74 |

Our gain over our own MCMC control is −0.03 dB on counter and −0.04 dB on bicycle. The paper's gain over its MCMC row is +0.06 and +0.02 dB.

Setup: 35k steps (paper protocol for Mip-NeRF 360), images_4 / images_2, llffhold 8, single seed. The budget is the paper's own MCMC primitive count per scene (Table 2): bicycle 5.9M, counter 1.2M.

## Results (test PSNR / SSIM / LPIPS, Gaussians)

| run | bicycle | counter |
|---|---|---|
| **denoising-gs v2** (`runs/denoising-gs-v2-35k`) | 25.43 / 0.788 / 0.202, 5.32M | 29.36 / 0.916 / 0.245, 1.08M |
| denoising-gs v1 (`runs/denoising-gs-35k`) | 25.39 / 0.788 / 0.201, 5.32M | crashed at 32k (see fixes) |
| denoising-gs-mcmc control (`runs/denoising-gs-35k`) | 25.46 / 0.788 / 0.201, 5.90M | 29.39 / 0.916 / 0.245, 1.20M |
| our gsplat 3DGS baseline (30k) | 24.92 / 0.757 / 0.243 | 28.85 / 0.907 / 0.260 |
| paper Denoising-GS (Table 2) | 26.17, 5.32M | 29.57, 1.08M |
| paper 3DGS-MCMC row | 26.15, 5.90M | 29.51, 1.20M |
| paper 3DGS row | 25.64 | 29.08 |

Gains (dB):

| comparison | bicycle | counter |
|---|---|---|
| ours − our MCMC control | −0.04 | −0.03 |
| paper ours − paper MCMC | +0.02 | +0.06 |
| our control − paper MCMC | **−0.69** | −0.12 |
| our control − our 3DGS baseline | +0.54 | +0.54 |
| paper MCMC − paper 3DGS | +0.51 | +0.43 |

Primitive counts match the paper exactly: 5.32M on bicycle and 1.08M on counter. That is 0.9 × the cap plus the small growth from the five refinement splits.

The method's own effect is within noise in both the paper (+0.02 / +0.06) and our runs (−0.04 / −0.03). The remaining bicycle gap is all in the backbone: our MCMC control is −0.69 dB below the paper's MCMC row.

Some of the paper-side difference may also come from step counts. The paper's MCMC row is the original 3DGS-MCMC paper's numbers at 30k steps, while Denoising-GS ran 35k. Our control also runs 35k at the same cap.

LPIPS is higher than typical paper values throughout. This is the known LPIPS offset of our runtime, and the control shows it too.

## What was built
Native plugin at `reproductions/denoising-gs/plugin` (`ss_denoising_gs`), written from the LaTeX source. There is no official code, and the paper's Appendix (its "detailed implementation") is not in the arXiv source.

- **Backbone:** 3DGS-MCMC using gsplat ops (relocation, +5 % growth to the cap from 500 to 25k, SGLD noise with noise_lr 5e5), the λ = 0.01 opacity/scale regulariser, and initial opacity 0.5.
- **C1, momentum drift:** −α·λ_t·m̂/(√v̂+ε), with α 0.05 and β1 0.9.
- **C2, local-frame correction:** +β2·R·Δμ with β2 0.5. Δμ_j = −|∇S_j|·sign(∇M^G_j), applied only when the dominant axes agree.
- **C3, Fisher pruning:** u = trace of Σ_v g gᵀ, using per-view L2 gradients on means and log-scales. The 10 % with the lowest u are removed at step 30k.
- **C4, spatial coherence refinement:** split the 0.05 % of primitives with the largest mean squared distance to their 3 nearest neighbours, per Eqs. p6/p7 with N = 2. This runs five times, at 30.5k–32.5k.
- **Control:** `denoising-gs-mcmc`, the same backbone with C1–C4 turned off.

Checks:
- ss-inspect clean, 6 unit tests pass, ruff clean.
- ss-verify returns `declared_checks_passed`.
- 6 mechanism probes check against independent references: an explicit 6×6 Fisher matrix with SVD, a numpy rotation loop, brute-force kNN, hand-written Eqs. p6/p7, and a split-then-train finiteness test.

## Fixes during the reproduction
1. **Review 1, C2 sign.** Eq. m4 as printed points up the gradient; we measured this for 100 % of aligned Gaussians. Fig. 2 and the §4.3 text instead move it with the update. The default now follows the figure; the printed sign remains available as the `literal` option.
2. **Review 1, C1 scale.** With the literal raw-gradient EMA, the drift is about 2.5e-10 per step against an Adam step of about 1.5e-5, so it does nothing. That contradicts the α sensitivity in the paper's Table 3(a). The default is now Adam-normalised; the literal form remains as `momentum_mode: raw`.
3. **Review 1, minor fixes:**
   - The schedule is checked against `--steps`.
   - The Fisher pass uses the current SH degree.
   - A `dmu_magnitude: adam_step` option was added.
4. **v1 counter crash** ("scales: nonfinite" at 32k, right after a refinement split).
   - Cause: gsplat's `compute_relocation` computes 1−√(1−o) directly. In float32 this is 0 when o < ~1e-7, so the Eq. p7 scale becomes o/0 = inf.
   - Fix: a cancellation-free closed form for N = 2 (`core.split_in_two`), plus the regression probe `split-finite`, which fails on the old code path.
   - The v1 bicycle run (25.39) used the old split; v2 (25.43) is the reported number.
   - The control never runs the split, so its results stand.

## Deviations / fidelity gaps
- **Missing Appendix: every schedule detail is guessed.**
  - Fisher pruning at total − 5000 (30k), modelled on PUP-3DGS's prune-then-fine-tune for 5k.
  - Refinement splits five times, every 500 steps after the prune.
  - Fisher pass over all training views at training resolution.
  - Noise temperature τ set to the MCMC noise_lr.
  - These guesses are consistent with Table 2's counts (pruning happens after growth stops, with no regrowth) and with §6 (refinement comes after pruning). Nothing in the paper decides between pruning at 25k and at 30k.
- **Interpretations:**
  - Adam-normalised momentum drift (C1).
  - C2 sign taken from Fig. 2 rather than Eq. m4.
  - ∇S is the log-scale gradient.
  - Δμ is not multiplied by a learning rate.
- **Split numerics:** the stable closed form replaces gsplat's relocation formula for the refinement split. It is mathematically identical; the probe checks agreement within 1e-4.
- **Input points:** the paper says it used "denser and noisier input SfM point clouds" on Mip-NeRF 360, without saying how. We use the standard SfM points.
- **Runtime conventions:**
  - gsplat rasteriser and gsplat MCMC instead of the official 3dgs-mcmc.
  - White background.
  - Views sampled at random with replacement.
  - LR decays over 35k steps, versus 30k in official MCMC.
  - The noise uses the learning rate after that step's decay.
- **Caps:** taken from the paper's MCMC row. T&T / DB caps are unknown and were not benchmarked.
- **Coverage:** only 2 scenes were benchmarked (per-scene protocol), so no dataset averages are claimed.
