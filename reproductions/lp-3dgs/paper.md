# LP-3DGS: Learning to Prune 3D Gaussian Splatting — implementation spec

arXiv 2405.18784 (v1, only version), NeurIPS 2024. Zhang, Song, Lee, Yang, Peng, Chellappa, Fan.
Official code: https://github.com/ASU-ESIC-FAN-Lab/LP-3DGS @ b828063 (the paper's
`dexgfsdfdsg/LP-3DGS` link redirects there). Inria 3DGS licence (non-commercial research).
Cloned to `source/code/` (read only). **Code defaults override the paper where they differ.**

## 1. Method summary
- Standard 3DGS training (densification to 15k, opacity reset every 3k) is unchanged.
- At step 19,500 a real-valued trainable mask `m_i` (init 1) is attached to every Gaussian.
- For 500 steps the rendered opacity is `o_i * gs(m_i * S_i)` where `gs` is a Gumbel-sigmoid
  (τ = 0.5) and `S_i` is a (detached) importance score recomputed every 20 steps over all
  training views; an L1/sparsity loss on the mask pushes masks to 0.
- At step 20,000 Gaussians whose (sampled) mask is < 0.5 are removed once; training then
  continues as plain 3DGS to 30,000 (pruning ratio is learned, not preset).
- Two score variants: RadSplat max-contribution `S_i = max_{views,rays} α_i T_i` (headline,
  better) and Mini-Splatting cumulative weight `S_i = Σ α_i T_i`.

Contributions
- C1 (§3.3 "Gumbel-Sigmoid based trainable mask", Eq. 7–8): Gumbel-sigmoid mask applied to the
  importance score and multiplied into opacity; gradients reach the mask (vs STE).
- C2 (§3.3 "Sparsity regularization", Eq. 9–10): mask L1 regulariser, λ_m = 5e-4.
- C3 (§3.3 last ¶, §4.1): 500-step mask-learning window (19,500–20,000), score refresh every 20
  steps, one-time hard pruning, fine-tune to 30k (learned pruning ratio).
- Importance score (§3.2, Eq. 3): RadSplat max contribution (re-implemented by the authors in
  their rasterizer `mw_score`). Not a contribution of this paper but required.

## 2. Equations (verbatim from LaTeX)
Eq. 3 (RadSplat score): `\bf{S}_i = \max_{I_f\in \mathcal{I}_f, r\in I_f} \alpha_i^r \cdot T_i^r`
Eq. 5 (Mini-Splatting score): `\bf{S}_i = \sum_{j=1}^{K} \omega_{ij}`
Eq. 7 (Gumbel-sigmoid):
`gs(m) = \frac{exp((\log(m) + g_0)/\tau)}{exp((\log(m)+g_0)/T)+exp(g_1/\tau)} = \frac{1}{1+exp(-(\log(m) + g_0 - g_1)/\tau)}`
Eq. 8 (apply mask): `o_{im} = o_i * gs(m_i*S_i)`
Eq. 9 (sparsity): `R_{mask} = \frac{1}{N} \sum_{i=1}^{N} \left| m_i \right|`
Eq. 10: `L = (1-\lambda_{ssim})*L_{L1} + \lambda_{ssim}*L_{ssim} + \lambda_m*R_{mask}`

Official code (mask.py, gaussian_model_rad_splat.py) — what is actually run:
- `gs(x) = sigmoid((x + n)/τ)`, `n = -log(log(u1+1e-10)/log(u2+1e-10) + 1e-10)`, u1,u2 ~ U(0,1)
  (difference of two Gumbels = logistic noise). Input is `x = m * S` (**not** `log(m*S)` as in
  Eq. 7). Fresh noise on every opacity evaluation (each render). `hard=False`.
- Masked opacity: `sigmoid(o) * gs(m*S)` (Eq. 8), S detached (computed under no_grad).
- Mask loss: `0.0005 * mean(sigmoid(m))` (**not** `mean|m|` as in Eq. 9).
- Final pruning (step 20,000): remove `gs(m) < 0.5` where `gs(m) = sigmoid((m + n)/τ)` with fresh
  noise and **without** the score — i.e. remove iff `m + n < 0` (stochastic).
- Mask optimizer: `AdamW([m], lr=0.05, eps=1e-15)` (default betas, weight_decay = 0.01), created
  fresh at 19,500; masks initialised to ones.
- RadSplat score (`mw_score` CUDA kernel): per pixel front-to-back, Inria numerics
  (α = min(0.99, o·G), skip α < 1/255, stop when T(1-α) < 1e-4), `atomicMax(score_i, α·T)`;
  max over all training views; opacity used is the *current* (masked, once active) opacity.

## 3. Hyper-parameters
| name | value | source |
|---|---|---|
| iterations | 30,000 | §4.1; arguments `iterations` |
| mask start (`prune_iterations`) | 19,500 | §4.1; arguments |
| mask iterations (`train_mask_iters`) | 500 | §3.3/§4.1; arguments |
| score refresh | every 20 steps | §4.1; train.py `(iteration-19500)%20==0` |
| τ (`gumbel_temp`) | 0.5 | §4.1; arguments |
| λ_m | 5e-4 | §4.1; `addtional_loss` hard-coded 0.0005 |
| mask lr | 0.05 (AdamW, wd 0.01 default, eps 1e-15) | mask.py `set_mask` |
| mask init | 1.0 | mask.py |
| prune threshold | gs(m) < 0.5 | gaussian_model_rad_splat.py |
| 3DGS defaults | lr, densify 500–15k every 100, grad 2e-4, reset 3k, λ_ssim 0.2 | arguments |

## 4. Schedules (official train.py loop, `it` 1-based)
- `it == 19500` (after backward): `set_mask` → m = 1, new AdamW. Gaussian optimiser step as usual.
- `19500 < it < 20000`: mask loss added; mask optimiser stepped together with the Gaussians.
- `19500 < it < 20000 and (it-19500)%20==0` (19520, …, 19980), **before** the render: score
  recomputed over all train views; opacity activation switched to masked. So 19501–19519 render
  unmasked (only the mask loss moves m: Adam ≈ −lr/step → m ≈ 0.05 at 19520).
- `it == 20000`: render still masked, mask not stepped; after backward prune `m + n < 0`,
  restore plain sigmoid opacity.
- 20000–30000: plain 3DGS fine-tune (densification already stopped at 15k).

## 5. Builds on
Inria 3DGS (code base is the official 3DGS repo). Everything else unchanged. SH degree +1 every
1000 steps (RadSplat model; the Mini-Splatting model variant delays SH increase until after 15k).

## 6. Protocol
Mip-NeRF 360 (9 scenes, images_4 outdoor / images_2 indoor — 3DGS convention), T&T truck/train,
NeRF-Synthetic. `--eval` llffhold 8. 30k iterations. Black background (3DGS default). Metrics:
3DGS `metrics.py` (PSNR, SSIM, LPIPS-VGG via lpipsPyTorch). Two A6000 GPUs. Baseline row = the
authors' own 3DGS run. No seeds stated.

## 7. Ambiguities / code-vs-paper differences
1. Eq. 7 uses log(m); code feeds `m*S` directly as the logit. Code followed.
2. Eq. 9 is mean|m|; code uses mean(sigmoid(m)). Code followed.
3. Final prune uses gs(m) without S and with fresh noise (stochastic); paper says "prune points
   with mask value 0". Code followed.
4. AdamW weight decay 0.01 on the mask (default, implicit).
5. Score render uses masked opacity after the first refresh (opacity activation swapped).
6. Table 4's LP-3DGS LPIPS row duplicates Table 1's baseline LPIPS row (copy error).
