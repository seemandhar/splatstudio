# SteepGS — Steepest Descent Density Control for Compact 3D Gaussian Splatting

Wang, Wang, Wang, Mohan, Fan, Wu, Cai, Yeh, Wang, Liu, Ranjan. CVPR 2025, arXiv 2505.05587.
Official code: https://github.com/facebookresearch/SteepGS (commit 63eee45, cloned to
`source/code/`; Gaussian-Splatting licence, non-commercial research). LaTeX in `source/latex/`.

## 1. Method summary
- Built on the official 3DGS code; only density control (and the rasterizer backward, which
  additionally outputs per-Gaussian splitting-matrix statistics) changes.
- Densification is cast as splitting each Gaussian into offspring; a 2nd-order expansion
  (Thm. 1) shows splitting lowers the loss only if the per-Gaussian **splitting matrix**
  S^(i) has a negative eigenvalue (saddle escape).
- Optimal split (Thm. 2, "SDC"): 2 offspring, displaced ±v_min(S^(i)), opacity × 1/2.
- In practice (official code): a Gaussian is densified iff the 3DGS screen-gradient criterion
  holds **and** λ_min(S^(i)) < −1e-6; offspring are placed along ±v_min.

Contributions
- C1 (§3.2, §3.4, App. A.1): splitting matrix S^(i) accumulated on the fly from the rasterizer
  backward (custom CUDA), averaged over the views since the last densification.
- C2 (§3.2, Thm. 2 (i)): saddle/eigenvalue split condition λ_min(S) < ε_split (= −1e-6), AND-ed
  with the ADC gradient condition ("uncertain" filter).
- C3 (§3.3, Thm. 2 (iv)): offspring displaced along ±v_min(S) (steepest descent directions).
- C4 (§3.3, Thm. 2 (iii)): offspring opacity halving — **paper only; disabled by default in the
  official code** (see Ambiguities).

## 2. Equations
Theorem 1 (sec/3_method.tex):
$$\mathcal{L}(\vartheta,w)=\mathcal{L}(\theta)+\nabla\mathcal{L}^\top\mu+\tfrac12\mu^\top\nabla^2\mathcal{L}\mu+\tfrac12\sum_i\sum_j w^{(i)}_j\delta^{(i)\top}_j S^{(i)}(\theta)\delta^{(i)}_j+O(\epsilon^3)$$
$$S^{(i)}(\theta)=\mathbb{E}_{(\Pi,x)\sim\mathcal{D}}\Big[\frac{\partial\ell}{\partial\sigma_\Pi(x;\theta^{(i)})}\nabla^2_{\theta^{(i)}}\sigma_\Pi(x;\theta^{(i)})\Big]$$
Theorem 2: if λ_min(S^(i)) ≥ 0 no split; else m=2, w_1=w_2=1/2, δ_1=v_min, δ_2=−v_min.
Hessian approximation (§3.4): ∇²σ_Π ≈ σ ΥΥ^T − σ P^T Π(Σ)^{-1} P, Υ = P^TΠ(Σ)^{-1}(x−Π(p)).
Algorithm 1 (App. A.1): accumulate G^(i) += ∇L, S^(i) += ∂_σℓ ∇²σ every step; every T_split
steps divide by count, λ ← λ_min(S), δ ← v_min(S); if (condition on G) and λ < ε_split,
replace by (θ+εδ, o/2), (θ−εδ, o/2).

**Estimator actually used (official code, `--S_estimator inv_cov`, README default):**
rasterizer backward (`backward.cu` renderCUDA) accumulates per Gaussian and view
c_v = Σ_pixels dL/dG, with dL/dG = o·dL/dα (Inria blending, α = min(0.99, oG), all pixels with
α ≥ 1/255); `add_densification_stats` then does
S^(i) += c_v · Σ_i^{-1}, Σ^{-1} = R diag(1/(s+1e-8))² R^T (3D inverse covariance, current
params), for visible Gaussians (radii>0). Densification: S̄ = S/denom (nan→0),
`eigh` ascending, λ_min, v_min. Note S̄ = c̄ Σ^{-1} (if Σ fixed): λ_min<0 ⇔ c̄<0 and then
v_min is the **shortest** principal axis. (Sign is opposite to the −σP^TΠ(Σ)^{-1}P term of the
paper approximation, and the ΥΥ^T term and σ factor are dropped — code is authoritative.)

## 3. Hyper-parameters
| name | value | source |
|---|---|---|
| densify_strategy | `steepest` | README command |
| S_estimator | `inv_cov` | README ("by default inv_cov"); arg default string `inv_covar` is a typo that would KeyError |
| densify_S_threshold ε_split | −1e-6 | paper §4.1, arguments/__init__.py |
| densify_grad_threshold (uncertain filter) | 0.0002 on mean ‖∂L/∂p_ndc‖ | arguments, gaussian_model.py:545 |
| min_grad (stationary, only with `stationary` option) | 1e-4 | gaussian_model.py:486 |
| N offspring | 2 | gaussian_model.py:533 |
| split length | ‖n‖, n~N(0, diag(s)) per offspring | gaussian_model.py:579-587 |
| clone offset | v_min · lr_xyz(current) for both copies | gaussian_model.py:594-603 |
| split scale | s / (0.8·2) | gaussian_model.py:619 |
| offspring opacity | copied (halved only with `no_div_opacity` option) | gaussian_model.py:608-627 |
| percent_dense | 0.01 | 3DGS default |
| densify from/until/interval | >500, <15000, 100 | 3DGS default, paper §4.1 |
| opacity reset | every 3000 (<15000), to min(o, 0.01), Adam moments zeroed | train.py:193, gaussian_model.py:247 |
| prune | o < 0.005; after it 3000: max scale > 0.1·extent (screen-size test inert) | train.py:189-190, gaussian_model.py:507 |
| LRs, λ_dssim 0.2, 30k its, SH +1/1000 | 3DGS defaults | arguments |

## 4. Schedules
Stats accumulate from iteration 1 for visible Gaussians; every densification resets all stats
(`densification_postfix`). Densify at 600, 700, …, 14900; opacity reset at 3000, 6000, 9000,
12000. Clone offset uses the means LR of the current iteration (exponential 1.6e-4→1.6e-6 ×
extent).

## 5. Builds on
Official 3DGS (Inria rasterizer + `GaussianModel`). Unchanged: loss, LRs, SH schedule, opacity
reset, pruning, initialization, black background, densification cadence.

## 6. Protocol
Mip-NeRF 360 (7 scenes: bicycle garden stump / bonsai counter kitchen room; images_4 outdoor,
images_2 indoor via `full_eval.py -i`), T&T (train, truck), Deep Blending (drjohnson, playroom),
`--eval` (llffhold 8), 30k iterations, black background, LPIPS-VGG (3DGS metrics.py). V100.
Per-scene PSNR/SSIM/LPIPS and point counts in supp. Tables 6-9.

## 7. Ambiguities / code-vs-paper
1. Opacity halving (paper Thm. 2 (iii), Alg. 1) is **off** in the official default: the option
   named `no_div_opacity` *enables* halving. We follow code (copy opacity); config flag kept.
2. Split offspring signs: `offsets[0::2] *= +len`, `offsets[1::2] *= −len` on a `repeat(N,1)`
   array (rows i and K+i belong to Gaussian i). For an even number K of split Gaussians both
   offspring of a Gaussian get the same sign; for odd K, opposite signs. Replicated exactly.
3. Clone branch: both the original and the copy are moved by +lr·v_min (same position).
4. "Condition on G" in Alg. 1 = the ADC viewspace-gradient threshold (option list default
   `['steepest']` → filters = uncertain ∧ λ_min<ε). `stationary` variant off.
5. Screen-size prune `max_radii2D > 20` is inert (postfix zeroes `max_radii2D` before the test).
6. Main Table 1 MipNeRF360 averages (7 scenes) differ slightly from the means of the per-scene
   supp. tables (SteepGS 28.734 vs 28.749; 3DGS 29.037 vs 29.121) — compare per scene.
   Supp. SSIM row for stump/bonsai/counter looks shifted (stump 0.742) — PSNR is the target.
