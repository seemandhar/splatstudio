# MaskGaussian: Adaptive 3D Gaussian Representation from Probabilistic Masks (CVPR 2025)

arXiv 2412.20522 · project https://maskgaussian.github.io/ · official code
github.com/kaikai23/MaskGaussian @ 2f581d0 (vanilla Inria 3DGS host) + rasterizer
github.com/kaikai23/mask-diff-gaussian-rasterization @ 540dc11 (Inria rasterizer + masks).
Licence: Gaussian-Splatting licence (Inria/MPII, non-commercial research) for both.
Sources: `source/latex/` (sec/3_method.tex, sec/4_experiments.tex, sec/X_suppl.tex, tables/),
`source/code/`, `source/maskrast/`.

## 1. Method summary
- Every Gaussian owns 2 learnable mask scores; each iteration a binary existence mask
  M_i ∈ {0,1} is drawn with hard Gumbel-Softmax (straight-through) from them.
- **Masked rasterization**: all Gaussians are splatted and α-filtered normally; the mask enters
  only the α-blending (colour and transmittance). Masked Gaussians do not change the image but
  still receive dL/dM_i = α_i T_i dL/dc · (c_i − b_{i+1}) (+ background term), so unsampled
  Gaussians learn whether they should come back.
- Squared mask loss L_m = (mean_i M_i)^2 with weight λ_m on a step window.
- Pruning: sample each Gaussian 10 times; remove those never sampled. Done at every
  densification step and every 1000 iterations after densification. The saved model is one
  final mask sample (save_ply_default keeps only Gaussians with M=1), rendered with the stock
  3DGS rasterizer.

Contributions
- C1 Masked rasterization forward/backward (§3.3 Eq. 3–4, §3.4 Eq. 6–7, supp. A).
- C2 Probabilistic masks via Gumbel-Softmax sampling (§3.3; ablation Tab. "Gumbel vs STE").
- C3 Squared mask loss + schedule (§3.5 Eq. 8–9; three settings α/β/γ, §4.1).
- C4 Mask-sampling pruning schedule (§3.5 last paragraph; code gaussian_model.py).

## 2. Equations (verbatim)
Eq. render3 / render4 (forward):
\textbf{c}(\textbf{x}) = \sum^N_{i=1} \mathcal{M}_i \cdot \textbf{c}_i \cdot \alpha_i \cdot T_i,
T_{i+1} = \mathcal{M}_i \cdot (1-\alpha_i) \cdot T_i + (1-\mathcal{M}_i) \cdot T_i.
Eq. b_color: \textbf{b}_{i+1} = \sum^N_{j=i+1} \mathcal{M}_j \cdot \textbf{c}_{j} \cdot \alpha_j \cdot T_j
(normalised by T_{i+1} in the supplementary recursion b_i = M_i α_i c_i + (1 − M_i α_i) b_{i+1}).
Eq. gradient: \frac{\partial L}{\partial\mathcal{M}_i} = \alpha_i \cdot T_i \cdot \frac{\partial L}{\partial \textbf{c(x)}} \cdot (\textbf{c}_i - \textbf{b}_{i+1})
supp. full form adds  \frac{-\alpha_i T_{N+1}}{1-\alpha_i \mathcal{M}_i} \frac{\partial L}{\partial c(x)} c_{bg}.
Eq. Lm: L_m = \left(\frac{1}{N} \sum_{i=1}^N \mathcal{M}_i\right)^2,  Eq. L: L = L_{render} + \lambda_m L_m.
Mask: M = gumbel_softmax(log_softmax(score), tau=1, hard=True)[:, 0]  (code gaussian_model.py:160).

## 3. Hyper-parameters
| name | value | source |
|---|---|---|
| mask score init | [10, 1] per Gaussian (p(exist)=σ(9)≈0.99988) | code gaussian_model.py:185 |
| mask lr | 0.01 (Adam, eps 1e-15) | code arguments/__init__.py `mask_lr` |
| Gumbel τ | 1 (torch default), hard=True (straight-through) | code gaussian_model.py:160 |
| λ_m, window (Ours-α, per-scene tables) | 0.1, 19000 ≤ it ≤ 20000 | paper §4.1, run_all.py, Tab. 1 caption |
| Ours-β | 0.0005, 0..30000 | paper §4.1; code default |
| Ours-γ | 0.001, 0..30000 | paper §4.1 |
| prune samples | 10 | code densify_and_prune / mask_prune |
| mask_prune_iter | 1000 (after densify_until) | code arguments |
| all 3DGS params | unchanged (densify 500..15000 every 100, grad 2e-4, reset 3000, min opacity 0.005, lambda_dssim 0.2) | code arguments |
| background | black | code (white_background False) |
| iterations | 30000 | paper §4.1 |

## 4. Schedules
- Mask sampled every iteration from step 1 (masking active the whole run in all settings;
  only λ_m is scheduled): simple_scheduler(λ, it, from, until) = λ if from ≤ it ≤ until else 0.
- Densification steps (it>500, it%100==0, it<15000): clone, split, then prune
  (opacity<0.005) ∪ (never sampled in 10 draws) ∪ (big world-space after 3000).
- it ≥ 15000 and it % 1000 == 0: mask_prune (never sampled in 10 draws), incl. 30000.
- Final save at 30000 (before that iteration's mask_prune): keep one Gumbel sample.

## 5. Builds on
Vanilla 3DGS (graphdeco) training loop unchanged except the mask field/loss/pruning and the
masked rasterizer for training renders. Test renders use render_orig_3dgs (stock rasterizer)
on the pruned single-sample model.

## 6. Protocol
Mip-NeRF 360 (9), T&T (train, truck), DB (drjohnson, playroom); `--eval` llffhold 8;
`images/` with -r -1 default → width rescaled to 1600 for Mip-NeRF 360 (= our `mipnerf360-1600`);
T&T/DB original resolution. 30k iterations, RTX 4090. LPIPS = lpipsPyTorch VGG.
Per-scene numbers (supp. Tables) are for MaskGaussian-α. 3DGS row = authors' retrain in the
same code base (garden 27.38 / 5.81M, counter 29.09 / 1.16M at 1600 px).

## 7. Ambiguities / code-vs-paper
- **Backward accumulation uses the wrong mask (official kernel).** backward.cu updates
  `accum_rec = mask*last_alpha*last_color + (1-mask*last_alpha)*accum_rec` with the *current*
  Gaussian's mask, while last_alpha/last_color belong to the Gaussian behind it. The exact
  supplementary recursion needs that Gaussian's own mask. This changes dL/dM and dL/dα whenever
  adjacent masks differ (rare while p(exist)≈1). The published numbers come from the official
  kernel; the port vendors it unmodified (default) and offers an `exact` compile-time variant.
- Paper says pruning at "every densification step, and every 1000 iterations after
  densification"; code: densify steps (100-step cadence, 600..14900) + it%1000==0 for it≥15000.
- Training-time mask covers all N Gaussians (L_m mean over all, not only visible).
- Final model = a single random Gumbel sample (not a deterministic threshold).
