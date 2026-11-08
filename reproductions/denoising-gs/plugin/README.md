# splatstudio-method-denoising-gs

Native SplatStudio port of **Denoising-GS: Gaussian Splatting with Spatial-aware Denoising**
(arXiv 2605.14880). No official code; written from the LaTeX source. The paper's Appendix
(implementation details) is not in the arXiv source, so schedules are guesses (see below).
Paper fidelity unverified.

Methods: `denoising-gs` (paper), `denoising-gs-mcmc` (3DGS-MCMC control, same budget, C1–C4
off), `denoising-gs-quick` (prune/SCR compressed into 2000 steps, smoke test only).

## Mechanism → code
- Backbone 3DGS-MCMC: gsplat `ops.relocate` / `ops.sample_add` (refine 500–25000 / 100,
  +5 % up to the cap, min opacity 0.005), SGLD noise `inject_noise_to_position(λ_t·5e5)`,
  init opacity 0.5, λ_o = λ_s = 0.01 reg (`losses.py`). Cap = the paper's MCMC row (Table 2)
  for the `--data` scene (`core.SCENE_CAPS`).
- C1 (Eqs. m2/msgld): default `means -= α λ_t m̂/(√v̂+ε)` (Adam-normalised; `momentum_mode: raw`
  = literal `α λ_t m`, a numerical no-op), `m ← β1 m + (1−β1)∇M` (α 0.05, β1 0.9).
- C2 (Eqs. m3–m5): `means += β2 R Δμ`, Δμ_j = −|∇S_j| sign((Rᵀ∇M)_j) if the dominant axes agree
  (β2 0.5; along the update per Fig. 2; `denoise_sign: literal` = Eq. m4 as printed;
  `dmu_magnitude: adam_step` = applied scale step).
- C3 (Eqs. f2–f3): u = Σ_v ||∇_(M,logS) ||I_v − Î_v||²||², prune lowest 10 % at total_steps − 5000.
- C4 (Eq. 4.1 + p6/p7): split the 0.05 % largest mean-squared-3NN-distance primitives (N = 2),
  5 times every 500 steps after pruning.

Gradients are captured with tensor hooks (the runtime zeroes `.grad` before `after_backward`).
Training images for the Fisher pass are reloaded from `--data` (pattern from dashgaussian).

## Guesses (paper silent)
τ / noise = MCMC noise_lr 5e5; prune at 30k of 35k (PUP-3DGS-like 5k fine-tune), SCR at
30.5k…32.5k; Fisher over all training views at training resolution; ∇S = log-scale gradient;
Δμ sign from Fig. 2 (descent); ss-train --steps must equal total_steps (asserted); per-scene cap from the paper's MCMC primitive counts.

## Commands
```
tools/ssenv ss-train denoising-gs --data ~/datasets/mipnerf360/native/bicycle --steps 35000 \
  --recipe 3dgs --num-gaussians 10000000 --max-gaussians 12000000 --output runs/...
```
