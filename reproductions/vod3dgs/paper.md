# VoD-3DGS: View-opacity-Dependent 3D Gaussian Splatting — implementation spec

arXiv 2501.17978 (Nowak, Jarosz, Chin; Dartmouth). Source: `source/latex/main.tex` (single file,
no supplementary). **No official code** (searched GitHub API for `vod3dgs`, `vod-3dgs`,
`view-opacity-dependent`, arXiv page, paperswithcode: nothing, 2026-11-07). Built "upon the
original codebase of 3DGS" (§5.1).

## 1. Summary and contributions
- Opacity becomes view dependent: logit = γ_i + ωᵀ Ŝ_i ω with a learnable per-Gaussian symmetric
  3×3 matrix Ŝ_i (6 params), SGGX-inspired (§4.1).
- C1 view-dependent opacity (§4.1, Eq. 9).
- C2 density control: opacity prune uses the *maximum* opacity over the training views (§4.2, Eq. 10).
- C3 enhanced opacity reset: at every 3DGS opacity reset Ŝ_i is replaced by its rank-1 eigen
  projection keeping the largest ([L]) or the smallest ([S]) eigenvalue (§4.3, Eq. 11).
- C4 view-consistency loss L_vc between the current view and a second view sampled ∝ keypoint
  matches (LightGlue in the paper; "could be collected during SfM") (§4.4, Eq. 12-13).
- C5 learning rates: γ and all six Ŝ components use lr_γ/4 (§5.1).

## 2. Equations (verbatim)
Eq. 8 (3DGS): $\alpha_i = \sigma(\gamma_i)$.
Eq. 9: $\hat{\alpha_i}(\omega_{i,j}) = \sigma(\gamma_{i,j} + \omega_{i,j}^T \hat{S}_i \omega_{i,j})$,
"ω_{i,j} defines the view vector between the center of the Gaussian i and the center of the camera j".
(γ_{i,j} is a typo for γ_i; quadratic form is sign-invariant in ω, so direction convention is moot;
ω is taken unit-norm — "view vector", SGGX convention.)
Ŝ (Eq. 5 layout): $\begin{pmatrix} S_{xx}&S_{xy}&S_{xz}\\S_{xy}&S_{yy}&S_{yz}\\S_{xz}&S_{yz}&S_{zz}\end{pmatrix}$, six free params (no PSD constraint stated).
Eq. 10: prune if $\hat{\alpha}(\omega_{i,j^*}) < \tau$, $j^* = \arg\max_{j\in N}\hat{\alpha_i}(\omega_{i,j})$, N = training set.
Eq. 11: $\hat S_i = Q\,\mathrm{diag}(\lambda_0,0,0)\,Q^{-1}$ (keep highest → [L]) or $Q\,\mathrm{diag}(0,0,\lambda_2)\,Q^{-1}$ (keep lowest → [S]).
Eq. 12: $L_{vc} = \frac{1}{|G|}\sum_{g \in G}\left[\max(\cos\theta,0)(\hat{\alpha}(\omega_{g,i})-\hat{\alpha}(\omega_{g,j}))^2\right]$, θ = angle between ω_{g,i}, ω_{g,j}.
Eq. 13: $L = (1-\lambda)L_1 + \lambda L_{D-SSIM} + L_{vc}$ (L_vc weight 1).

## 3. Hyper-parameters
| name | value | source |
|---|---|---|
| lr γ (opacity logit) | 0.05/4 = 0.0125 | §5.1 (3DGS lr_γ = 0.05) |
| lr Ŝ (each of 6) | 0.0125 | §5.1 |
| Ŝ init | 0 (not stated; makes start identical to 3DGS) | assumption |
| τ (prune) | 0.005 | 3DGS default (unchanged, §5.1) |
| L_vc weight | 1 | Eq. 13 |
| λ (D-SSIM) | 0.2 | 3DGS |
| all other 3DGS params | unchanged | §5.1 |
| iterations | 30000 | 3DGS protocol (§5.3) |

## 4. Schedules
3DGS: densify 500–15000 every 100, opacity reset every 3000 (Ŝ reset at the same steps, C3),
SH degree +1 per 1000. L_vc active from step 1 (no warm-up stated).

## 5. Builds on
Original 3DGS (Inria code): photometric loss, ADC, SH, schedules unchanged except the above.

## 6. Protocol
Mip-NeRF 360 (all 9), T&T (truck, train), Deep Blending (drjohnson, playroom), NeRF-Synthetic;
same training/eval as 3DGS (llffhold8, images_4 outdoor/images_2 indoor). Only per-dataset
**averages** reported (Table 1); 3DGS† = authors' recomputation. Single L40 GPU. No seeds.

## 7. Ambiguities (choices)
- γ reset: paper says Ŝ is reset "to facilitate a similar approach"; we keep the 3DGS γ reset
  (min(σ(γ), 0.01)) and add the Ŝ reset ("we do not modify any existing parameters").
- Eq. 11 wording ("λ0 highest, λ2 lowest"; L/S "respectively") → [L] keeps the max eigenvalue,
  [S] the min (consistent with [L] using more memory: max eigenvalue ≥ min, so more Gaussians survive).
- Adam moments of Ŝ at reset: zeroed (as 3DGS does for opacity at reset).
- L_vc gradient paths: formula involves γ, Ŝ and means (through ω); we let gradients reach γ and Ŝ
  and detach the means (view vector treated as geometry input). Over all Gaussians (|G| = N).
- View sampling: p(j|i) ∝ matches(i,j), j≠i. We use COLMAP covisibility (number of shared
  triangulated SfM points) instead of LightGlue matches (paper's own suggested alternative).
- Eq. 10 max over all training cameras (no visibility/frustum test stated).
- No clamp of the quadratic form (older Nerfstudio port clamped to ±10; not in paper).
