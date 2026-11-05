# Improving Adaptive Density Control for 3D Gaussian Splatting — implementation spec

Grubert*, Barthel*, Hilsmann, Eisert (HU Berlin / Fraunhofer HHI), VISIGRAPP/GRAPP 2025,
arXiv 2503.14274. Official code: https://github.com/fraunhoferhhi/Improving-ADC-3DGS
(commit `65901d8`, cloned to `source/code/`, read only). The code is a fork of **PixelGS**
(https://github.com/zhengzhang01/Pixel-GS), which is a fork of Inria 3DGS. Inria/PixelGS licence
(non-commercial research).

## 1. Method summary
- Plain 3DGS training (losses, LRs, schedule, SH, opacity reset) with a modified ADC.
- The paper's "base" already includes two prior-work changes (Sec. 4 intro, Sec. 2):
  - **B1 PixelGS gradient**: pixel-count-weighted, depth-scaled 2D gradient average (Eq. 9/10).
  - **B2 Revised opacity** (Bulò et al., "Revising Densification"): cloned/split Gaussians get
    o_new = 1 − sqrt(1 − o_old) (Eq. 8).
- Contributions (paper Sec. 4):
  - **C1 Corrected scene extent** (Sec. 4.1, Eq. 13): mean distance from the mean camera centre
    to the SfM points, instead of 1.1 × max camera distance.
  - **C2 Exponentially ascending gradient threshold** (Sec. 4.2, Eq. 14): from 1e-4 to 4e-4 over
    the densification window.
  - **C3 Significance-aware pruning** (Sec. 4.3, Eq. 15/16, Alg. 2): prune only low-opacity
    Gaussians whose accumulated blend weight sigma_k is in the bottom N_prune; size pruning
    (world and screen) removed.

## 2. Equations (verbatim from LaTeX)
- Blend weight (Eq. 3–5): `color_p = \sum_k c_k \alpha_k t_k`, `w_k = \alpha_k t_k`.
- Baseline extent (Eq. 11/12): `e_\text{scene} = 1.1 \cdot \max_i \| \overline{C} - C_i \|_2`,
  `\overline{C} = \frac{1}{N_\text{cam}} \sum_i C_i` (training cameras).
- **Eq. 13 (C1)**: `e_\text{scene}' = \frac{1}{N_{\text{SfM}}} \sum_{i=1}^{N_{\text{SfM}}} \left\| \overline{C} - p_i \right\|_2`
- **Eq. 14 (C2)**: `T_i = \exp ( \ln(T_s) (1 - \frac{i}{i_{max}}) + \ln(T_f) \frac{i}{i_{max}} )`
  with T_s = 0.0001, T_f = 0.0004.
- **Eq. 15/16 (C3)**: `\sigma_k = \sum_{p=1}^{N_\text{pixel}} w_k^p`, over all pixels of all views
  between two densification steps; `\Sigma = \{\sigma_1..\sigma_n\}`.
  Alg. 2: if `o_k < o_min` and `\sigma_k \in \text{bottom}(N_\text{prune}, \Sigma)` → prune, where
  N_prune = #Gaussians with o < o_min.
- **Eq. 9 (B1)**: `\tau_k' = \frac{\sum_v f_k^v \cdot count(g_k^v) \cdot \| \partial L / \partial \mu_k^v \|_2}{\sum_v count(g_k^v)}`
- **Eq. 10 (B1)**: `f_k^v = clip( depth(g_k^v) / (\gamma e_\text{scene}), [0,1])`, γ = 0.37.
  **Code differs**: `scaling = min(1, (depth / (0.37·e_scene))^2)` (squared;
  `submodules/diff-gaussian-rasterization/diff_gaussian_rasterization/__init__.py:145`), applied
  to the means2D gradient vector, and e_scene here is the **old camera extent**
  (`train.py: depth_threshold = opt.depth_threshold * scene.old_extent`).
- **Eq. 8 (B2)**: `o_{new} = 1-\sqrt{1-o_{old}}` for parent and clone (clone) and both split
  children (code applies it to split too: `gaussian_model.py:densify_and_split`).

## 3. Hyper-parameters
| name | value | source |
|---|---|---|
| iterations | 30000 | arguments/__init__.py:73 |
| densify_from / until / interval | 500 / 15000 / 100 | arguments:87-89 |
| T_s, T_f | 1e-4, 4e-4 | paper §4.2; train.py:64-65 |
| threshold steps | dense_iters = (15000−500)//100 = 145; factor = 4^(1/145); `current *= factor` **before** each densify call → call k (k=1..144, steps 600..14900) uses 1e-4·4^(k/145) | train.py:63-67,151 |
| percent_dense | 0.01 (× e') | arguments:83 |
| min_opacity | 0.005 | train.py:152 |
| opacity reset | every 3000 while < 15000, o ← min(o, 0.01) | train.py:155, gaussian_model:217 |
| γ (PixelGS depth_threshold) | 0.37 × old (camera) extent | arguments:91, train.py:122 |
| split N, scale factor | 2, /1.6 | gaussian_model:385-400 |
| size pruning | disabled (both screen 20 px and world 0.1·extent) | gaussian_model:466-471 |
| means LR scale (spatial_lr_scale) | **e' (corrected extent)** | scene/__init__.py:94 |
| LRs, λ_dssim, SH schedule | 3DGS defaults | arguments |

## 4. Schedules
- Stats (gradient, pixel count, σ) accumulate from step 1 while step < 15000; reset after each
  densify call (`densification_postfix` zeroes grad accum; `reset_denom`; `per_gaussian_alpha=None`).
- Densify + prune at step ∈ {600, 700, …, 14900}. Threshold rises per call (C2).
- Opacity reset at 3000, 6000, 9000, 12000 (after that step's densify).
- Densification stops at 15000; training to 30000.

## 5. Base and unchanged parts
Inria 3DGS: L1 + 0.2 D-SSIM, Adam LRs, exp means LR decay, SH up every 1000, SfM init, opacity
init 0.1, 3-NN scales, black background. Changes: only ADC (B1, B2, C1–C3) and the means LR scale.

## 6. Protocol
- Datasets: Mip-NeRF 360 (9 scenes; README/`full_eval.py`: outdoor `images_4`, indoor
  `images_2`; the paper says "we used the 3DGS evaluation pipeline"), Tanks&Temples (truck, train),
  Deep Blending (drjohnson, playroom). (`script/mipnerf360.py` from PixelGS uses `-r 4/2`; the
  paper text points to full_eval.py.) llffhold 8, `--eval`.
- 30k iterations, black background, LPIPS VGG (3DGS metrics.py), test every 8th image.
- Reported: **dataset averages only** (Table 1), ablation over 13 scenes (Table 3), 15k vs 30k
  over 3 scenes (Table 2). No per-scene tables in the paper or supplementary.

## 7. Ambiguities / code-vs-paper
1. Eq. 14 is continuous in i/i_max; code is a per-call geometric step over 145 intervals (used).
2. Eq. 10 unsquared vs code squared; extent for depth scaling = camera extent (code used).
3. Pruning, code: `scores = σ[old non-split]`, `q = torch.quantile(scores, N_prune/N_old)`,
   prune old non-split Gaussians with `σ < q` **and** `o < 0.005`; newly created clones/children are
   never pruned in that call. N_prune counts low-opacity Gaussians among the post-densify set.
4. σ computed in code via a fake-colour pass: gradient of Σ render(fake_color=0) w.r.t. the colour
   = Σ_p w_k^p (mean over the 3 equal channels). Black bg makes it geometry-neutral.
5. Clone: parent opacity also overwritten in place (Adam moments kept); clone has zero moments.
6. Official order: density before optimizer step (Inria).
7. 3DGS row in Table 1 (27.39 / 23.74 / 29.50) is not the 3DGS paper (27.21 / 23.14 / 29.41):
   presumably retrained or taken from PixelGS; source not stated.
