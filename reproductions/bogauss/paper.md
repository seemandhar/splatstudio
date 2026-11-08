# BOGausS: Better Optimized Gaussian Splatting — implementation spec

arXiv 2504.01844 (Pateux, Gendrin, Morin, Ladune, Jiang; Orange / INSA Rennes, Apr 2025).
Sources: `source/latex/` (e-print, sections 3_preliminary.tex = "Precision" + "Improved gradient
descent", 4_methods.tex = "Improved Densification", 5_results.tex). **No official code**: none
linked in the paper or the plan, none found by web search (GitHub API search was rate-limited;
web search and paper aggregators list no repository). Everything below is from the paper text.

## 1. Method summary
- Budgeted training (target final count per scene), densification every 500 iterations
  (as in Taming 3DGS), split-only densification (no cloning), no opacity reset.
- Per-Gaussian positional precision Δ_i from a multi-view triangulation-noise model; used as
  minimal size and as a per-Gaussian step scale for means and (linearly parameterized) sizes.
- Modified (unbiased, sparse) Adam with per-Gaussian lifespan counter t; optimizer states are
  partially inherited by split children.
- Pruning by effective opacity (max over cameras of mean blend weight over footprint), capped
  at 1 % of the Gaussians per densification.
- Density-preserving split along the longest axis (±0.3σ), children size/opacity from an
  offline-fitted `SplitAdaptation`; split priority = relative contributed squared error P_SNR.

Contributions (intro list + sections):
- C1 Precision/confidence of the 3D reconstruction → Δ_i and minimal size (§"Precision matrix derivation").
- C2 Unbiased sparse Adam with precision-aware (Δ-scaled) updates and linear size parameterization (§"Improved gradient descent", 3 subsections).
- C3 Optimizer-state inheritance after splitting (§"States inheritance after Gaussian splitting").
- C4 Effective-opacity pruning replacing opacity reset (§"Effective opacity-based Gaussian pruning").
- C5 Density-preserving split, no clone (§"Density-preserving Gaussian splitting").
- C6 Distortion-based split prioritisation P_SNR (§"Distortion-based prioritization").

## 2. Equations (verbatim from LaTeX)
Eq. (precision):
```
{\Sigma^{(XYZ)}_i}^{-1} = \sum_c \left( {z^{(c)}_i}^2 {\sigma^{(uv)}}^2 W^T_c \mathrm{diag}(1,1,0) W_c \right)^{-1}
\Delta_i = \alpha \, \mathrm{tr}({\Sigma_i^{(XYZ)}}^{-1})^{-0.5}, \alpha = 2 ;  \Sigma^{(\sigma)}_i = \Sigma^{(XYZ)}_i
```
W_c = world→camera rotation of camera c, z_i^(c) = camera-space depth. The inner matrix is rank 2;
interpreted as the pseudo-inverse: per camera the lateral precision is 1/(z² σ_uv²) on the two
image axes. With σ_uv in pixels and focal f: tr = Σ_c (f_x² + f_y²)/(z_c² σ_uv²).

Eq. (adamNewState):
```
\alpha_{s,t} = \max(1/t, 1-\beta_s^\star);  \hat g_0 = 0
\hat g_t   = \hat g_{t-1}   + \alpha_{1,t} (\nabla_\theta L_t - \hat g_{t-1})
\hat g^2_t = \hat g^2_{t-1} + \alpha_{2,t} ((\nabla_\theta L_t)^2 - \hat g^2_{t-1})
\theta_{t+1} = \theta_t + \eta_\theta \Delta_\theta \frac{\hat g_t}{\hat g^2_t + \epsilon}
```
β* as in Adam (0.9, 0.999). t is per Gaussian, incremented only when the Gaussian is updated
(sparse: only Gaussians contributing to the current image). Δ_θ = Δ_i for positions and sizes,
1 otherwise. Sizes σ_a are linearly parameterized (not log). Two evident typos: sign (+ should
be − for descent) and missing square root on ĝ²_t (standard Adam); implemented as
θ ← θ − η Δ ĝ / (sqrt(ĝ²) + ε).

Inheritance:
```
t_child = \alpha_t t_mother;  \hat g_child = \alpha_g \hat g_mother;  \hat g^2_child = \alpha_g^2 \hat g^2_mother
(\alpha_t, \alpha_g) = (1, 0.2)
```

Pruning:
```
P^{(c)}_i = \frac{\sum_p w^{(c)}_i(p)}{S^{(c)}_i};  P_prune(i) = \max_c P^{(c)}_i ;  prune if P_prune < \tau_prune = 0.02,
at most 1 % of the current number of Gaussians per pruning.
```
w_i^(c)(p) = blending weight T·α of Gaussian i at pixel p; S_i^(c) = surface covered in image c.

Split: along the longest principal axis, means μ ± 0.3 σ_long · axis;
`(σ_child, o_child) = SplitAdaptation(σ_mother, o_mother)` "learned offline" so that the sum of
the children's density approximates the mother's (function not given).

Split priority:
```
SE_{c,i} = \sum_{p \in I_c} w_i^{(c)} (R^{(c)}(p) - I^{(c)}(p))^2
\Delta PSNR_{c,i} \simeq \frac{10}{\ln 10} \frac{(1-\alpha_{split}) SE_{c,i}}{SE_c}
P_{SNR}(i) = \sum_c \frac{SE_{c,i}}{\sum_j SE_{c,j}}
```

## 3. Hyper-parameters
| name | value | source |
|---|---|---|
| α (precision scaling) | 2 | §Precision |
| σ_uv | not given → 1 pixel | ambiguity |
| β1, β2 | 0.9, 0.999 ("as in traditional Adam") | §Proposed optimizer |
| ε | not given → 1e-15 (3DGS) | ambiguity |
| η for Δ-scaled means / sizes | not given | ambiguity (see 7) |
| η other params | not given → 3DGS defaults | ambiguity |
| α_t, α_g | 1, 0.2 | §States inheritance |
| τ_prune | 0.02 | §Effective opacity pruning |
| prune cap | 1 % of current count | same |
| split offset α | 0.3 | §Density-preserving splitting |
| densification interval | 500 | §Experimental setup |
| iterations | 30k (table file names "30k") | tables |
| budget ("Big") | per-scene vanilla 3DGS count (= Taming "big" budgets) | §Results, #G column |

## 4. Schedules
Densification every 500 its; start/stop not given (Taming budget mode: 500 → 15000). Growth curve
to the budget not given (Taming's parabolic count array assumed, since budgets, interval and
Sparse Adam all come from Taming). SH degree schedule, 30k iterations, means LR decay: 3DGS.

## 5. Builds on
3DGS training (L1 + 0.2 D-SSIM loss, SH schedule, SfM init) on what appears to be the Taming 3DGS
code base ("fast" = Taming's fast mode with Sparse Adam; budgets equal Taming's). Replaced:
optimizer, densification (no clone, new split, no gradient threshold), pruning, opacity reset.

## 6. Protocol
Mip-NeRF 360 (9), T&T (truck, train), Deep Blending (drjohnson, playroom); 3DGS splits
(llffhold 8); resolution as 3DGS (images_4 outdoor / images_2 indoor implied by "same as
3DGS"). SSIM/PSNR/LPIPS. Results reported as **dataset averages only**; no per-scene table.
Exposure-compensated rows exist (not reproduced). Background, LPIPS net, seeds: not stated.

## 7. Ambiguities and choices
1. η for Δ-scaled updates: not given. Choice: dimensionless η calibrated once at start so that
   the median initial step equals 3DGS's (means: 1.6e-4·extent / median Δ, decayed ×0.01;
   sizes: 5e-3·median(σ/Δ), i.e. the log-lr 0.005 relative step for the median Gaussian).
2. σ_uv unit / value: 1 pixel. Cameras in the sum: training cameras whose frustum (1.3× FoV
   margin, z > 0.2 like 3DGS culling) contains the Gaussian centre; Gaussians in no frustum get
   the largest finite Δ.
3. Minimal size: σ_a ≥ Δ_i (all axes), applied in the update.
4. S_i^(c): number of pixels where the Gaussian contributes (α ≥ 1/255 before termination).
5. Statistics window: accumulated over the training iterations between densifications
   (each iteration's camera, current residual); max over seen cameras for P_prune.
6. SplitAdaptation: offline numeric fit of (σ_child/σ_mother, o_child) per mother opacity
   minimizing the L2 error between the mother's alpha profile and the alpha-composited children
   along the split axis; only the split-axis size changes.
7. Number of splits per densification: grow to the next count-array target after pruning;
   top-k by P_SNR (no gradient threshold).
8. Densification window 500..15000 (Taming); pruning only inside the window.
