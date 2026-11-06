# A Step to Decouple Optimization in 3DGS (AdamW-GS) — implementation spec

- arXiv 2601.16736 (ICLR 2026), Ding et al. Project https://eliottdjay.github.io/adamwgs/
- Official code: https://github.com/EliottDJay/3DGS_AdamWGS (MIT), cloned to `source/code/`
  (built on the graphdeco 3DGS code + `diff-gauss` rasterizer with a fused CUDA Adam kernel).
- LaTeX in `source/latex/iclr2026_conference.tex` (main + appendix with per-scene tables).

## 1. Method summary

- Adam in 3DGS couples (i) the *update step* (synchronous updates move invisible primitives via
  historical moments, "implicit update") and (ii) *gradients* (regularisation gradients enter the
  same moments as photometric gradients).
- AdamW-GS recouples three pieces: **Sparse Adam** (only visible primitives update; per-primitive
  asynchronous step counter for bias correction), **Re-State Regularisation (RSR)** (every 100
  iterations a uniform random subset of primitives has its moments multiplied by α1, α2), and
  **Decoupled Attribute Regularisation (DAR)** (opacity / scale L1 regularisation removed from the
  loss and applied in the update as `min(λ ∇R / N_I / (sqrt(v̂)+ε), C)`; moments see only ∇ℓ).
- Applied to vanilla 3DGS: plus MCMC-style position noise (outdoor only), no opacity reset in the
  released configs, opacity correction on clone ("Opacity Correction is adopted for cloning").
- Result: ~-50 % active primitives with slightly higher PSNR than 3DGS; faster training.

Contributions
- C1 Sparse Adam with asynchronous per-primitive step (Sec. 4.1, Eq. 4; code `diff_gauss.SparseGaussianAdam`, `adam.cu`).
- C2 Re-State Regularisation with milestone StSS (Sec. 4.2, Eq. 5; App. "StSS in RSR"; code `GaussianModel.restate_regularization`).
- C3 Decoupled Attribute Regularisation for opacity and scale (Sec. 4.3, Eq. 8; App. Eqs. 13–16; code `adam.cu` reg branches).
- C4 AdamW-GS recoupling into vanilla 3DGS incl. exploration noise and clone opacity correction (Sec. 4.4; App. Eq. 17–18; code `train.py` noise block, `opacity_correction`).

## 2. Equations

Eq. 1 (loss, 3DGS-MCMC form): `\mathcal{L} = (1-\lambda_1)\mathcal{L}_1 + \lambda_1 \mathcal{L}_{DSSIM} + \lambda_o |o|_1 + \lambda_s |s|_1`
— for vanilla 3DGS + AdamW-GS only the photometric part is a loss (λ1 = 0.2); the regularisers go through DAR.

Eq. 2/3 (Adam): `m_t = β1 m_{t-1} + (1-β1) g_t`, `v_t = β2 v_{t-1} + (1-β2) g_t^2`,
`θ_{t+1} = θ_t − η m̂_t/(sqrt(v̂_t)+ε)`.

Eq. 4 (Sparse Adam): `β' = β × V + (1 − V)` (V = visibility filter, radii > 0) → invisible
primitives keep moments and parameters unchanged. Bias correction uses a per-primitive step
`t_i` (code `asystep`, incremented by V each iteration): `m̂ = m/(1−β1^{t_i})`, `v̂ = v/(1−β2^{t_i})`.

Eq. 5 (RSR): `m^{new} = α1 m^{old}`, `v^{new} = α2 v^{old}`, `0 ≤ α < 1`, on primitives sampled uniformly
(without replacement) by StSS every 100 iterations; the per-primitive `t_i` is *not* reset
(`er_asy_step_reset: False`).

Eq. 7 (decoupled moments): `m'_t = β'1 m'_{t-1} + (1−β'1) ∇ℓ/N_I`, `v'_t = β'2 v'_{t-1} + (1−β'2)(∇ℓ/N_I)^2`.

Eq. 8 (DAR): `θ_{t+1} = θ_t − η [ m̂'/(sqrt(v̂')+ε) + min(λ_θ (∇R(θ)/N_I)/(sqrt(v̂')+ε), C_t) ]`.

App. Eqs. 13–16: opacity `o = σ(τ)`, `∇R = σ(τ)(1−σ(τ))` (L1 on positive o has unit gradient);
scale `s = exp(κ)`, `∇R = exp(κ)` (per component).

App. position noise: `R_μ = η_μ σ(−λ_μ(o − λ_t)) Σ γ`, γ~N(0,I) — code: `noise = randn * op_sigmoid(1−o) * noise_lr * xyz_lr`,
`op_sigmoid(x)=1/(1+exp(−100(x−0.995)))`, `noise ← Σ noise` with `Σ = (RS)(RS)^T` (full covariance,
not its square root, as in 3DGS-MCMC), added to μ after the optimizer step.

Clone opacity correction (code `opacity_correction`, applied to the *source* before copying, so
source and clone both get it): `o' = clamp(1 − sqrt(1 − o), 0.005, 1−eps)`.

Exact kernel numerics (code `submodules/diff-gauss/cuda_rasterizer/adam.cu`), per element of a visible primitive:
```
bc1 = 1/(1-0.9^t_i); bc2 = 1/(1-0.999^t_i)
m = 0.9 m + 0.1 g ; v = 0.999 v + 0.001 g^2
denom = sqrt(v*bc2) + eps                     # eps = 1e-15
reg = min(reg_scale * o(1-o) / denom, 10)     # opacity field, inside opacity window
reg = min(reg_scale * exp(κ) / denom, 10)     # scale field, inside scale window
θ = θ - lr * (m*bc1/denom + reg)
```
`reg_scale = 0.001 / img_scale` (opacity) and `1e-5 / img_scale` (scale), hard-coded in
`rasterize_points.cu:adamUpdate`; `img_scale = (first two digits of N_I) × 10^(len(N_I)−3)` ≈ N_I/10,
N_I = W×H of the first training camera (`SparseGaussianAdam._img_scale`).

## 3. Hyper-parameters

| name | value | source |
|---|---|---|
| β1, β2, ε | 0.9, 0.999, 1e-15 | adam.cu call (`_C.adamUpdate(..., 0.9, 0.999, eps)`), get_optimizer |
| λ_o / λ_s | 1e-3 / 1e-5 (÷ N_I/10) | paper App. "DAR"; code rasterize_points.cu:256-262 |
| C_t (opacity, scale) | 10, 10 | paper App. "DAR"; adam.cu `fminf(...,10.0f)` |
| opacity DAR window | [3000, 15000] (inclusive) | config `opacity_penalty_begin/end` (T&T/DB: end 14000) |
| scale DAR window | [1000, 15000] | config `scaling_penalty_begin/end` (T&T/DB: end 14000) |
| RSR interval, window | 100, [1000, 15000] | config `restate_reg` (T&T/DB end 14000; paper figure shows 14000) |
| StSS GS1 (indoor, `mipnerf_adamwgs`) | ratio 0.1 from 1000, 0.3 from 7000 | config ratio_list/ratio_shift; Fig. 7c |
| StSS GS2 (outdoor, `mipnerf_adamwgsp`) | 0.1@1000, 0.3@7000, 0.5@10000 | config; Fig. 7c |
| StSS GS3 (`ttdb_adamwgs` drjohnson only; truck/train/playroom use 0.1/0.3) | 0.1@1000, 0.15@7000 | config |
| StSS GS5 (outdoor GS7 rows) | 0.5@1000, 0.7@3000 | Fig. 7c only (no released config) |
| α1 (exp_avg fading) | 0.2 | paper App. "Scaling in RSR"; config `rsr_exeavg_fading` |
| α2 (exp_avg_sq fading) | **paper 0.04; code-effective 0.01** | config key `rsr_exeavgsq_fading: 0.04` is never read — code reads `rsr_exeabg_sq_fading` (default 0.01), gaussian_model.py:121 |
| noise | Mip-NeRF 360 outdoor, truck, drjohnson, playroom: on, [1000, 14000]; Mip-NeRF 360 indoor and T&T `train`: off | config `noise_transport` (ttdb_adamwgs/sub12_train: False), `noise_begin/end` |
| noise_lr | 5e5 (× current xyz lr) | arguments/__init__.py:138 (3DGS-MCMC default) |
| opacity reset | off (interval 300000) in all released AdamW-GS configs | config; the paper's GS8 rows use reset (see §7) |
| opacity correction on clone | on | config `use_opacity_correction: True` |
| densification | 3DGS defaults: every 100 from 500 (exclusive) to 15000, grad 2e-4, percent_dense 0.01, split /1.6 (divide_ratio 0.8 × N=2), prune o<0.005 | config |
| big-primitive prune | never in released configs (needs `iteration > opacity_reset_interval`); with reset (GS8) world-space only, screen test dead | train.py size_threshold, gaussian_model.py prune |
| init opacity | 0.1 | gaussian_model.py opacity_initial |
| lrs | 3DGS defaults (pos 1.6e-4→1.6e-6 × extent, f_dc 2.5e-3, f_rest/20, opacity 0.05, scale 5e-3, rot 1e-3) | config |
| iterations | 30000 | config |

## 4. Schedules
- 1000: scale DAR, RSR (ratio 0.1), noise (where on) start. 3000: opacity DAR starts.
- 7000: RSR ratio 0.3 (DB 0.15). 10000: RSR ratio 0.5 (outdoor GS2 only).
- 14000: noise stops. 15000: DAR, RSR and densification stop (P-Op: plain Sparse Adam).
- SH degree +1 every 1000; position lr exponential decay to 30000.

## 5. Builds on
Vanilla 3DGS (graphdeco code) — renderer, ADC, losses unchanged except: optimizer replaced by the
fused Sparse-Adam/DAR kernel, RSR, position noise, clone opacity correction, no opacity reset.
(3DGS-MCMC + AdamW-GS is the other host in the paper; not targeted here.)

## 6. Protocol
Mip-NeRF 360 (9 scenes; images_4 outdoor / images_2 indoor, forced in train.py), T&T (truck,
train), DB (drjohnson, playroom); llffhold 8; black background; 30k iterations; PSNR/SSIM/LPIPS-VGG
(3DGS metrics.py). Single A6000. Per-scene tables for all 9 Mip-NeRF 360 scenes in App. "All Results";
T&T and DB only as dataset averages (Table 3).

## 7. Ambiguities / code-vs-paper
1. α2: paper 0.04, released code effectively 0.01 (config-key typo). Port default follows code; config exposes it.
2. N_I scaling: paper says "most significant digit ... /10"; code keeps two significant digits (`_img_scale`).
3. Which config produced which table row: released outdoor config (`mipnerf_adamwgsp`: noise, no reset, StSS GS2)
   equals per-scene row **GS9**; released indoor config (`mipnerf_adamwgs`: no noise, no reset, StSS GS1) equals
   row **GS7/GS8** of the indoor tables. The main-table "GS8" outdoor rows use opacity reset (column "reset ✓") with
   StSS GS2 and "GS7" uses StSS GS5 without reset — neither has a released config.
4. With opacity reset (GS8), train.py:219 sets size_threshold=20 after 3000. Only the world-space test (> 0.1 extent) actually fires: max_radii2D is re-zeroed by densification_postfix (called in clone and split) before prune(), so the 20-px test never triggers.
5. In the official loop, on densification iterations all `.grad` are None after densify, so the Adam kernel skips every
   field that iteration (only `asystep` is incremented, with the visibility extended to new primitives).
6. RSR samples are drawn over all current primitives (including fresh zero-moment ones) — `torch.multinomial(ones, k)` without replacement.
7. Paper Fig. 7 shows StSS/DAR ending at 14000; Mip-NeRF 360 configs end at 15000.
