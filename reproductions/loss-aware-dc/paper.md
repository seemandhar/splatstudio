# Unified Loss-Aware Density Control for 3D Gaussian Splatting

Kim, Chung, Heo (Sungkyunkwan University), NeurIPS 2026, arXiv 2611.02044. LaTeX source in
`source/latex/` (sections/4_method.tex, sections/7_appendix.tex).
**Official code:** none. `github.com/vanmeruso/Unified-Loss-Aware-Density-Control-for-3D-Gaussian-Splatting`
(first author) only has a README that says "Code will be soon" (commit 267f5ef, 2026-11-03), cloned
to `source/code/`. The NeurIPS checklist says the code was attached to the supplementary, which we do
not have. Everything below comes from the paper text.

## 1. Method summary
- The base is SteepGS (CVPR 2025, splitting matrix and saddle-escape split) on top of the 3DGS
  training configuration.
- All of 3DGS's heuristic ADC operations (clone, split, opacity reset) are replaced by per-Gaussian
  loss-change surrogates. These are accumulated during training and evaluated at every
  density-control interval.
- At each interval the operations run in a fixed priority, **Split → Clone → Decay**, on disjoint
  sets. Low-opacity Gaussians are then pruned and all accumulators are reset (Alg. 1).

Contributions:
- **C1 (§4.2, App. A.4) Joint position–scale splitting.**
  - The SteepGS splitting matrix is computed in φ = [p; log s] ∈ R^6.
  - It is normalised by J = blockdiag(L, I/√2), with L = R diag(s).
  - The split direction is the min-eigenpair of the normalised matrix.
  - Children are placed at p ± L v_p and log s ± v_s/√2, each with opacity o/2.
- **C2 (§4.2, Eq. 9) Split selection.** A Gaussian splits only if it is stationary
  (‖ḡ_pos‖ ≤ τ_pos and ‖ḡ_scale‖ ≤ τ_scale) and λ_min < τ_split.
- **C3 (§4.3) Surrogate-guided cloning.**
  - The quadratic surrogate in the duplicate opacity η_dup has terms g_dup and h_dup.
  - η_dup = −g/h. A clone is added only if g < 0 and τ_clone < η ≤ 1e-2.
  - The child's opacity is η·o.
- **C4 (§4.4) Surrogate-guided selective opacity decay.**
  - This replaces the global opacity reset.
  - η_dec = g_dec/h_dec, applied when g > 0 and τ_decay < η ≤ 1e-2, giving o ← (1−η)o.
- **C5 (App. A.5) Proxy curvature ω.** ω is a diagonal mix of Charbonnier and SSIM curvature,
  used in h_dup and h_dec.

## 2. Equations (verbatim from LaTeX, then definitions)
Joint split matrix (Eq. 5):
$$\phi^{(i)}=\begin{bmatrix}p^{(i)}\\ \tilde s^{(i)}\end{bmatrix},\ \tilde s=\log s,\qquad
\mathcal S_\phi^{(i)}=\mathbb E_{(\varphi,x)}\Big[\frac{\partial\mathcal L_{\varphi,x}}{\partial\alpha^{(i)}(x)}\nabla^2_{\phi^{(i)}}\alpha^{(i)}(x)\Big]$$
Computed via the screen variable z = [u_x,u_y,q_xx,q_xy,q_yy] (u = projected mean, q = conic),
α = o·exp(−½ dᵀQd), with d = u − x (App. Eq. app_screen_z…app_projected_alpha):
$$\mathcal S_\phi=\mathbb E\Big[\mathcal J_z^\top\mathcal H_z\mathcal J_z+\sum_{m=1}^5 g_{z_m}\nabla^2_\phi z_m\Big],\quad
g_z=\tfrac{\partial\mathcal L}{\partial\alpha}\nabla_z\alpha,\ \mathcal H_z=\tfrac{\partial\mathcal L}{\partial\alpha}\nabla_z^2\alpha$$
The blocks of the second-order chain term are as follows:
- S_pp keeps Σ g_u ∇²_p u + Σ g_q ∇²_p q (Eq. app_Spp).
- S_ss keeps Σ g_q ∇²_s q (Eq. app_Sss).
- **S_ps omits the mixed term Σ g_q ∇²_{ps} q** ("more stable first-order pullback form",
  Eq. app_Sps_impl).
- S is symmetrised before eigendecomposition.

Normalisation (Eq. 6):
$$L=R\,\mathrm{diag}(s),\quad J=\begin{bmatrix}L&0\\0&\tfrac{1}{\sqrt2}\mathbf I\end{bmatrix},\quad \tilde{\mathcal S}=J^\top\bar{\mathcal S}J$$
Eigen (Eq. 7): (λ_min, v_min) = eigmin(S̃).

Split set (Eq. 8):
$$\mathcal I_{split}=\{i\mid \|\bar g_{pos}\|_2\le\tau_{pos}\wedge\|\bar g_{scale}\|_2\le\tau_{scale}\wedge\lambda_{min}<\tau_{split}\}$$
Children (Eq. 9): p_{1,2} = p ± δ_p, s̃_{1,2} = s̃ ± δ_s̃, o_{1,2} = o/2, with [δ_p; δ_s̃] = J v_min.
The children inherit rotation and appearance.

Clone (Eqs. 10–13):
- Δ(x) = T α (c − b^{(i+1)}(x)). Here b^{(i+1)} is the background-inclusive colour composited
  behind i.
- Δ_dup = (1−α)Δ.
- g_dup = Σ_{x,c} ∂L/∂I_c · Δ_dup,c and h_dup = Σ ω_c Δ_dup,c².
- η_dup = −g_dup/h_dup.
- The clone set is I_clone = {i ∉ I_split | g_dup < 0 ∧ τ_clone < η_dup ≤ 1e-2}, and each clone is
  appended as (p, Σ, η_dup·o, c).

Decay (Eqs. 14–17):
- g_dec = Σ ∂L/∂I_c · Δ_c and h_dec = Σ ω_c Δ_c².
- η_dec = g_dec/h_dec.
- The decay set is I_decay = {i ∉ split ∪ clone | g_dec > 0 ∧ τ_decay < η_dec ≤ 1e-2}, and each
  member gets o ← (1−η_dec)o.

Proxy curvature (App. A.5):
- ω_charb = ε²/(r²+ε²)^{3/2}, with r = I − Î and ε = 1e-3.
- μ = G∗Î and ν = G∗Î² − μ² (11×11 Gaussian window, σ = 1.5).
- D_μ⁻¹ = 1/(2μ²+C1+ε_ssim) and D_ν⁻¹ = 1/(2ν+C2+ε_ssim).
- ω_ssim = 2(G²∗D_μ⁻¹) + 2[(G∗D_ν⁻¹) − (G²∗D_ν⁻¹)].
- ω = (1−λ)ω_charb + λ γ_ssim ω_ssim, with λ = 0.2 and γ_ssim = 1e-4.
- ω is computed once per rendered image and reused for every Gaussian.

## 3. Hyper-parameters
| name | value | source |
|---|---|---|
| τ_pos = τ_scale | 1e-4 | App. A.1 |
| τ_split | −1e-8 | App. A.1, Tab. split_hyper |
| τ_clone | 1e-6 | App. A.1 |
| τ_decay | 1e-4 | App. A.1 |
| η upper bound (clone, decay) | 1e-2 | Eqs. 13, 17 |
| Charbonnier ε | 1e-3 | App. A.5 |
| γ_ssim | 1e-4 | App. A.5 |
| SSIM window | 11×11, σ=1.5; C1=0.01², C2=0.03² (standard) | App. A.5 |
| ε_ssim | **unspecified** (we use 1e-8) | — |
| densify from / every / until | 500 / 100 / 15000 | App. A.1 |
| prune opacity | 0.005 (3DGS) | App. A.1 |
| LRs, λ_dssim=0.2, 30k its, SH schedule | 3DGS defaults | App. A.1 |

## 4. Schedules
- Statistics accumulate every step until 15000.
- At every step > 500 with step % 100 == 0 and step < 15000, the method runs Split → Clone → Decay,
  then prunes and resets the accumulators.
- There is no global opacity reset; it is replaced by the decay.

## 5. Builds on
- The code base is SteepGS: the statistics are "computed within the rasterizer" as in SteepGS.
- The 3DGS training is unchanged: losses, LRs, init, SH, and the 30k iterations.
- 3DGS's world-space big-Gaussian prune is not mentioned; we keep the 3DGS / SteepGS
  default prune.

## 6. Protocol
- Datasets: Mip-NeRF 360 (9 scenes; images_4 outdoor, images_2 indoor), T&T (train, truck),
  DB (drjohnson, playroom).
- Train/test split: llffhold 8, 3DGS evaluation.
- Runs: 5 runs per scene on an A6000, with exposure compensation disabled.
- Per-scene PSNR, SSIM and LPIPS are in App. B.1.
- LPIPS: VGG presumably (3DGS metrics).

## 7. Ambiguities (and our choice)
1. **"Unresolved view-space variation".** The text says a split Gaussian "is nearly stationary,
   has unresolved view-space variation, and exhibits sufficiently negative curvature", but Eq. 8
   lists only stationarity and curvature. SteepGS's default ANDs an "uncertain" filter
   (mean ‖∇_{2D}‖ ≥ 2e-4). **Choice: include the SteepGS uncertain filter by default (config
   `require_uncertain`).** This is the largest open question; it can be ablated.
2. **ḡ_pos and ḡ_scale.** We take the norm of the view-averaged dL/dp and dL/d log s, as in the
   SteepGS "stationary" option (mean over the views where the Gaussian is visible).
3. **Expectation normalisation of S.** Pixels are summed per view and averaged over visible views
   (SteepGS `denom`). This matters only for τ_split.
4. **Loss normalisation.** ∂L/∂I is the gradient of the per-image *mean* loss (3DGS), while ω is the
   un-normalised per-pixel curvature. This is consistent with the small thresholds (η ≤ 1e-2,
   τ_clone 1e-6).
5. **Clone children** have opacity ≤ 0.01·o and are subject to the same-interval prune (o < 0.005),
   following Alg. 1 literally.
6. **Split.** Children use no 1/1.6 scale shrink and no percent_dense size branch (Eq. 9 literally).
   Their opacity is o/2.
7. **Optimizer state.** Children's Adam moments are zero (3DGS postfix). The decayed opacity keeps
   its Adam moments, which the paper does not say.
8. **Background.** The paper uses the 3DGS default (black). The runtime and baseline use white.
9. **Pruning.** We prune o < 0.005, plus the world-size prune max s > 0.1·extent after step 3000
   (3DGS / SteepGS `max_screen_size` branch). The paper only says "the opacity threshold is kept".
