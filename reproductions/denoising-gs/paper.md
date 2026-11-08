# Denoising-GS: Gaussian Splatting with Spatial-aware Denoising — implementation spec

arXiv 2605.14880 (Zhou et al., Fudan / CUHK / NTU, 2026). Sources: `source/latex/` (arXiv e-print,
main `arxiv.tex` + `Section/*.tex`, `Tables/*.tex`), `source/paper.pdf`, `source/paper.txt`.
**Official code: none** ("Source code and models will be made publicly available"; re-checked
2026-11-07: no code_url in plan, no link in the paper). **The Appendix referenced for "detailed
implementation" of every module is NOT in the arXiv source** — all scheduling details below are
missing from the paper and are our guesses (§7).

## 1. Method summary
- Backbone is **3DGS-MCMC** (Preliminaries §3 explains MCMC's SGLD update Eq. p5 and relocation
  Eqs. p6/p7; the "detailed implementation" says the noise is opacity-aware and covariance-shaped,
  i.e. exactly MCMC's noise; Table 2 primitive counts equal 0.9 × the MCMC row: bicycle
  5.90→5.32, bonsai 1.30→1.17, counter 1.20→1.08).
- C1 **Momentum-biased stochastic exploration** (§4.2): MCMC's SGLD mean update plus a momentum
  drift −αλ_t m_t, m_t an EMA (β1) of past mean gradients; independent of opacity/covariance.
- C2 **Spatial gradient-based denoising** (§4.3): in each Gaussian's local frame, if the dominant
  axis of the mean gradient equals the dominant axis of the scaling gradient, add
  β2 R Δμ with Δμ_j = |∇S_j| sign(∇M^G_j) to the mean.
- C3 **Uncertainty-based denoising** (§4.4): Fisher information from per-view L2 loss gradients
  w.r.t. (means, scales) per primitive; u_i = Σ singular values (= trace for PSD F_i); prune the
  10 % with the smallest u (highest uncertainty).
- C4 **Spatial coherence refinement** (§4.5): d_i = mean squared distance to k=3 nearest neighbours;
  split the 0.05 % sparsest primitives with MCMC's Eqs. p6/p7 (N=2); done 5 times.

## 2. Equations (verbatim)
SGLD (Eq. m1): `\mathcal{M}_{t+1} = \mathcal{M}_t - \lambda_{t} \nabla \mathcal{M}_t L_{\mathrm{total}} + \sqrt{2 \lambda_{t} \tau} \, \boldsymbol{\xi}_t`
MCMC simplification (Eq. p5): `\mathcal{M} \leftarrow \mathcal{M} - \lambda_{\mathrm{lr}} \nabla\mathcal{M} L_{\mathrm{total}} + \lambda_{\mathrm{noise}}\lambda_{\mathrm{lr}} \boldsymbol{\xi}`
Momentum (Eq. m2): `\mathbf{m}_{t} = \beta_1 \mathbf{m}_{t-1} + (1 - \beta_1)\nabla \mathcal{M}_{t-1}`
Update (Eq. msgld): `\mathcal{M}_{t+1} = \mathcal{M}_t - \lambda_{t} \nabla \mathcal{M}_t L_{\mathrm{total}} + \sqrt{2 \lambda_{t} \tau} \, \boldsymbol{\xi}_t -\alpha\lambda_{t}\mathbf{m}_{t}`
Local gradient (Eq. m3): `\nabla \mathcal{M}^{G} = \mathbf{R}^\top \nabla \mathcal{M}`
Denoise term (Eq. m4), only if argmax-axis(∇M^G) == argmax-axis(∇S) = j:
`\Delta \mu=|\nabla \mathbf{S}_j| \cdot \mathrm{sign}(\nabla \mathcal{M}_{j}^{G})`
Final mean update (Eq. m5): `\mathcal{M}_{t+1} = \mathcal{M}_t - \lambda_{t} \nabla \mathcal{M}_t L_{\mathrm{total}} +\Delta D +\beta_2\mathbf{R}\Delta \mu`,
`\Delta D = \sqrt{2 \lambda_{t} \tau} \, \boldsymbol{\xi}_t -a\lambda_{t}\mathbf{m}_{t}` (a ≡ α).
Fisher (Eq. f2): `F_i = \sum_{v \in \mathcal{V}} \nabla{\theta_i} \mathcal{L}_v \, (\nabla{\theta_i} \mathcal{L}_v)^\top`,
`\mathcal{L}_v = \| I_v - \hat{I}_v(\theta) \|_2^2`, θ_i = (M_i ∈ R³, S_i ∈ R³), V = all camera views.
Uncertainty (Eq. f3): `u_i = \sum_{j=1}^{6} \sigma_j(F_i)`; smaller u ⇒ more uncertain ⇒ pruned.
Sparsity (Eq. 4.1): `d_i = \frac{1}{k} \sum_{j=1}^{k} \| x_i - x_{i,j} \|_2^2`; larger d ⇒ densified via Eqs. p6/p7:
`o^{new}_{1..N} = 1 - \sqrt[N]{1 - o^{old}_N}`, `\Sigma^{new} = (o^{old})^2 (\sum_{i=1}^{N}\sum_{k=0}^{i-1}\binom{i-1}{k}\frac{(-1)^k (o^{new})^{k+1}}{\sqrt{k+1}})^{-2}\Sigma^{old}`.

## 3. Hyper-parameters
| name | value | source |
|---|---|---|
| α (momentum bias) | 0.05 | §5.1, Table 3(a) |
| β1 (EMA) | 0.9 | §5.1 |
| β2 (denoise term) | 0.5 | §5.1 |
| k (kNN) | 3 | §5.1 |
| prune fraction | 10 % most uncertain | §5.1, Table 3(b) |
| densify fraction / times | 0.05 % sparsest, 5 times | §5.1 |
| iterations | 35k Mip-NeRF 360, 30k T&T / DB | §5.1 |
| τ / noise | not given → MCMC noise_lr 5e5 × λ_t, opacity-aware covariance shaping | guess (MCMC, "detailed implementation in Appendix") |
| λ_t | not given → 3DGS means lr 1.6e-4·extent, exp decay to 0.01× | guess (3DGS/MCMC) |
| MCMC cap | not given → paper's own MCMC row counts (Table 2): bicycle 5.90M, bonsai 1.30M, counter 1.20M, garden 5.20M, kitchen 1.80M, room 1.50M, stump 4.75M | inferred |
| MCMC λ_opacity, λ_scale | 0.01, 0.01 | MCMC code (not stated) |
| MCMC refine | 500–25000 every 100, +5 %/refine, min_opacity 0.005, init opacity 0.5 | MCMC code (not stated) |

## 4. Schedules (all missing from the paper)
- Prune step, Fisher view set, SCR timing: not given. Guess: Fisher prune once at
  `total_steps − 5000` (30k on M360 = MCMC's standard 30k, leaving 5k like PUP-3DGS's
  prune-then-fine-tune 30k+5k; explains the odd 35k for M360), all training views, full
  training resolution; SCR five times at prune+500, +1000, …, +2500.

## 5. What it builds on
3DGS-MCMC (Kheradmand et al. 2024): relocation + growth to a cap, SGLD noise, opacity/scale L1
reg, L1+0.2 D-SSIM. 3DGS rasteriser, SH degree schedule and LRs unchanged. Fisher idea from
FisherRF / PUP-3DGS (cited).

## 6. Protocol
Mip-NeRF 360 7 scenes (no flowers/treehill, following MCMC), T&T truck/train, DB drjohnson/playroom.
Downsampling not stated → MCMC/3DGS convention images_4 outdoor / images_2 indoor; llffhold 8.
"We explore the use of denser and noisier input SfM point clouds on Mip-NeRF 360" — unspecified
(how it was densified is unknown; we use the standard SfM). Metrics SSIM/PSNR/LPIPS (net not
stated). H100, PyTorch 2.5.1. Single run presumably.
Table 1 baseline rows (3DGS, MCMC) are the original papers' numbers (MCMC 29.89/24.29/29.67).

## 7. Ambiguities and our choices (revised after review-1)
1. **Δμ sign**: Eq. m4/m5 as printed (+β2RΔμ, sign(∇M^G_j)) points *up* the gradient (measured
   100 % ascent). Fig. 2 and the §4.3 text ("same directional sense as the original gradient
   update" −λ∇M) say it moves along the update. Default `denoise_sign: descent`
   (Δμ_j = −|∇S_j| sign(∇M^G_j)); `literal` keeps the printed sign.
2. ∇S = stored log-scale gradient (axis selection and default magnitude). Fig. 2 writes `s + ∇s_j`
   for the applied update, so `dmu_magnitude: adam_step` (|exp(s')−exp(s)|_j, the Adam scale step
   in world units) is offered as an option; default stays |∇S_j| (raw grad, p99 ≈ 5e-6).
3. **Momentum scale**: the literal −αλ_t m_t with a raw-gradient EMA is ~2.5e-10 per step vs an
   Adam step ~1.5e-5, i.e. a no-op, contradicting Table 3(a) (α changes DB PSNR by 0.4 dB). Default
   `momentum_mode: adam`: −αλ_t m̂_t/(√v̂_t+ε), m an own lagged EMA (β1, bias-corrected), v̂ the
   means Adam second moment. `raw` = literal. This is an interpretation (fidelity gap).
4. Δμ is not multiplied by a learning rate (Eq. m5 literal).
5. Fisher θ = (means, log-scales) by default; u_i via trace (= Σσ for PSD, checked by probe);
   Fisher pass uses the runtime's current SH degree.
6. Noise uses λ_t after the step's LR decay (gsplat MCMC convention).
7. Momentum and Δμ are applied every step to all Gaussians throughout training.
8. Schedule (prune 30k / SCR 30.5k–32.5k) is consistent with Table 2 counts (0.9× MCMC cap + 5
   tiny splits ⇒ prune after growth stops, no regrowth) and §6 (SCR after pruning); 25k vs 30k is
   not pinned. §6's 13.9K for 0.1 % fits 5 applications; the 5.6K for 0.05 % does not (7.0K) —
   likely a paper typo. The schedule is tied to `total_steps`; a mismatching `--steps` raises.
9. **Comparability**: the paper's MCMC row (26.15 / 29.51) is the original MCMC paper at 30k steps;
   Denoising-GS ran 35k, so part of its +0.02 / +0.06 dB may be the extra 5k steps. Our control
   runs the same cap at 35k (LR decays over 35k, official MCMC over 30k).
