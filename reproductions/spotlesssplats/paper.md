# SpotLessSplats: Ignoring Distractors in 3D Gaussian Splatting — implementation spec

arXiv 2406.20055 (v2, ACM TOG). Sabour, Goli, Kopanas, Matthews, Lagun, Guibas, Jacobson, Fleet,
Tagliasacchi. Project https://spotlesssplats.github.io/ . Official code
https://github.com/lilygoli/SpotLessSplats (gsplat 1.1.1 fork, `examples/spotless_trainer.py`,
commit 0caae3cc45bb1fddf86bd47e4a521888f5c49889, Apache-2.0) cloned in `source/code/`.
Paper LaTeX in `source/latex/` (sections `sec/3_background.tex`, `sec/4_method.tex`,
`sec/5_results.tex`, `sec/X_suppl.tex`).

## 1. Method summary
- 3DGS trained on *casual* captures that contain transient distractors; test views are clean.
- Per-pixel inlier mask multiplies an L1 photometric loss (Eq. 1). Masks come from residuals
  (RobustNeRF-style trimmed estimator, Eq. 2) refined by **semantic features**: Stable Diffusion
  2.1 up-block-1 features (DIFT, t=261, empty prompt) precomputed once per training image.
- Two ways to turn features into masks: SLS-agg (per-image agglomerative superpixels, 100
  clusters, vote by inlier fraction, Eq. 3–4) and **SLS-mlp** (headline: a tiny MLP on features +
  positional encoding, trained concurrently with hinge supervision from two residual
  quantiles, Eq. 5–8).
- 3DGS adaptations: Bernoulli warm-up of the mask with a staircase exponential schedule (Eq. 9),
  residual-quantile tracking with a discounted histogram (§4.3.2), opacity reset replaced by
  **utilization-based pruning (UBP)** (Eq. 10), one-off SH-rest reset at step 8000, GLO
  appearance (Eq. 11, paper only — not in the released code).

Contributions
- C1 (§4.1, Eq. 1–2) robust masked L1 with RobustNeRF-style 3×3 spatial mask, threshold = τ-quantile of residuals.
- C2 (§4.1.2, Eq. 5–8) SLS-mlp: SD-feature MLP mask classifier co-trained with hinge loss between the τ=.5 and τ=.9 residual masks (+ Lipschitz reg λ=0.5).
- C3 (§4.1.1, Eq. 3–4) SLS-agg: agglomerative clustering variant (alternative to C2).
- C4 (§4.2.1, Eq. 9, supp. B) Bernoulli scheduled-sampling warm-up.
- C5 (§4.2.2) histogram-based running residual quantile (B buckets of width 1e-3 in the paper; code 1e-4).
- C6 (§4.2.3, Eq. 10) utilization-based pruning instead of opacity reset.
- C7 (§5 impl. details) opacity reset off; SH-rest reset to 0.001 at step 8000.
- C8 (§4.2.4, Eq. 11) GLO appearance (per-image latent → affine on SH). **Paper only**; the
  official gsplat code has `app_opt=False` and its README states its reproduced numbers are
  without appearance modelling.

## 2. Equations (verbatim from LaTeX)
Eq. 1 (`eq:nvs`): `\argmin_{\model} \sum_{n=1}^{N} \mask_{n}^{(t)} \odot \| \image_{n} - \hat{\image}_{n}^{(t)} \|_1`

Eq. 2 (`eq:robustnerf`): `\mask_{n}^{(t)} = \indicator\{(\indicator\{\residual_{n}^{(t)} > \rho\} \circledast \filter) > 0.5\},\; P(\residual_{n}^{(t)} > \rho) = \tau`
with `\residual = \|I-\hat I\|_1` per pixel over colour, ρ generalised median, B a normalised 3×3 box filter.
**Code** (`spotless_trainer.py:robust_mask`): `e = mean_c|I-Î|`; `inl = (e < thr)`;
`nb = conv3x3_box(inl, zero pad) > 0.5`; `mask = (nb + inl) > 1e-3` (i.e. inlier OR majority-inlier neighbourhood).
`thr` = left bin edge (`linspace(0,1,10001)[k]`) of the first histogram bin whose cumulative
count reaches `τ·total`.

Eq. 3: `P(c \in \mask^{(t)}) = (\sum_p \cluster[c,p] \cdot \mask^{(t)}[p]) / \sum_p \cluster[c,p]`
Eq. 4: `\mask^{(t)}_\text{agg}(p) = \sum_c \indicator\{P(c \in \mask^{(t)}) > 0.5\} \cdot \cluster[c,p]`
(code: sklearn `AgglomerativeClustering(linkage="ward", connectivity=kneighbors_graph(50×50 grid, 8), n_clusters=100)` on the 1280×50×50 features; cluster maps nearest-upsampled to the image.)

Eq. 5: `\mask^{(t)}_\text{mlp} = \mlp(\features; \theta^{(t)})`, 1×1 convs (i.i.d. per pixel).
Eq. 6: `\loss(\theta) = \loss_{sup}(\theta) + \lambda \loss_{reg}(\theta)`, λ=0.5.
Eq. 7: `\loss_{sup} = \max(\upperm - \mlp(\features), 0) + \max(\mlp(\features) - \lowerm, 0)`
Eq. 8: `\upperm = \mask` from Eq. 2 with τ=.5; `\lowerm = \mask` with τ=.9.
(Paper notation is confusing; the code is unambiguous: `p` ≥ mask(thr at quantile 0.5) and
`p` ≤ mask(thr at quantile 0.9): `mean(relu(p - M_0.9) + relu(M_0.5 - p))`.)
L_reg: Lipschitz bound of Liu et al. 2022 Eq. 13; **code**: `max|W1| * max|W2|` computed on
`.data` → contributes **no gradient** (logged only).

Eq. 9 (`eq:schedule`): `\mask^{(t)} \sim \mathcal{B}(\alpha \cdot 1 + (1-\alpha)\mask_*^{(t)})`
Supp. B: `\alpha = \exp(-\beta_1 \lfloor (t+1)/\beta_2 \rfloor)`, β1 = 3e-4 (3e-3 for the three
highest-occlusion On-the-go scenes), β2 = 1.5. **Code default β1 = 3e-3 for all scenes**
(`schedule_beta=-3e-3`, `sls_benchmark.sh` never overrides it).

Eq. 10 (`eq:bayessplat`): `\util_g = \sum_{t \in N_M(t)} E_{w,h} \| \mask^{(t)}_{h,w} \cdot \partial \hat I^{(t)}_{h,w} / \partial x^{(t)}_g \|_2^2`,
computed every M=100 steps over the previous 100 images; prune if `\util < 10^{-8}`.
**Code**: CUDA backward accumulates, per Gaussian, `Σ_pixels (v_x² , v_y²)` where `v` is the
per-pixel contribution to dL/d(means2d) of the masked-L1 loss (so `mask·sign(err)/(3HW)` takes
the role of the mask; only where `opac·G ≤ 0.999`), summed over the 100 steps between refinements;
prune if `Σ(x²+y²) < ubp_thresh = 1e-14`; newly densified Gaussians get utilization 1.

Eq. 11 (`eq:glo`): `\hat{\sphcoef}_{i} = \mathbf{a} \odot \sphcoef_{i} + \mathbf{b},\; \mathbf{a}, \mathbf{b} = \aglonet(\latent_n)`, latent ∈ R^64.

Loss in code: `rgbloss = mean_{HWC}(M ⊙ |clamp(Î,0,1) - I|)`, `ssim_lambda = 0` (no D-SSIM).

## 3. Hyper-parameters
| name | value | source |
|---|---|---|
| iterations | 30000 | paper §5, code `max_steps` |
| data factor | 8 (images_8) | README commands, code `data_factor=8` |
| loss | masked L1 only, SSIM weight 0 | code `ssim_lambda=0.0`, `loss_type="robust"` |
| robust_percentile (mask τ) | 0.7 | code `robust_percentile` (paper: generic τ) |
| MLP supervision quantiles | 0.5 (lower) / 0.9 (upper) | paper Eq. 8, code `lower_bound`, `upper_bound`; `0.3/0.8` for spot, patio-high in `sls_benchmark.sh` |
| histogram | 10000 bins on [0,1], discount 0.95 | code `bin_size`, `update_running_stats` (paper says width 1e-3) |
| quantile tracking stops | step 15000 | code: inside `if step < refine_stop_iter` |
| schedule | α = exp(-3e-3·⌊(t+1)/1.5⌋) | code `schedule_beta=-3e-3` (paper supp. 3e-4 except 3 scenes) |
| MLP | Linear(1280+80,16)-ReLU-Linear(16,1)-Sigmoid | code `utils.SpotLessModule` (paper: ~30K params, 2 non-linearities) |
| MLP optimiser | Adam lr 1e-3 | paper §5, code |
| MLP reg λ | 0.5 (no gradient in code) | paper Eq. 6, code |
| positional encoding | 20 freqs, sin/cos of y and x ∈[0,1], 2^k π | code `get_positional_encodings` |
| SD features | SD 2.1 (`stabilityai/stable-diffusion-2-1`), up_ft_index 1, t=261, prompt "", ensemble 4, image resized to 800×800 → 1280×50×50 | paper §5, notebook `sd_feature_extraction.ipynb` |
| feature upsample | bilinear to H×W (MLP); nearest for clusters | code |
| densify | every 100 from step>500 to <15000, grad2d ≥ 2e-4, grow_scale3d 0.01 | code (3DGS defaults) |
| prune | opacity < 0.005; scale prune never (`step > reset_every=300000`); no screen-size prune | code |
| opacity reset | off (`reset_every=300000`; skipped for robust loss) | paper §5, code |
| SH reset | step 8002 (0-based): `shN = clamp(shN, max=0.001)`, Adam moments zeroed | code `reset_sh=8002` (paper: "reset to 0.001 at 8000") |
| UBP | on for headline SLS-mlp; threshold 1e-14 (code) / 1e-8 (paper, different normalisation) | code `ubp_thresh`, paper §4.2.3 |
| LRs | means 1.6e-4·scene_scale (exp decay to 0.01×), scales 5e-3, quats 1e-3, opac 5e-2, sh0 2.5e-3, shN 2.5e-3/20 | code |
| init | SfM points, opacity 0.1, 3-NN scale, random quats | code |
| background | black (no background term) | code |
| SH degree | +1 every 1000 steps, max 3 | code |
| near/far | 0.01 / 1e10 in normalized world | code |
| scene_scale | 1.1 × max distance of ALL COLMAP cameras from their mean (after normalization) | code Parser |
| world normalisation | `similarity_from_cameras` + `align_principle_axes` | code `normalize=True` |
| seed | 42 | code |

## 4. Schedules
- t<15000 (0-based): residual histogram + 3 quantile thresholds updated every step (thresholds used at
  step t come from the histogram up to t-1; initial thr 1.0 / lower 0.0 / upper 1.0).
- Bernoulli warm-up α(t) from 1 → ~0 (α≈0.135 at t=1000, 0.0025 at t=3000 with β1=3e-3).
- t=8002: SH-rest clamp. t ∈ {600,700,…,14900}: densify + opacity prune + UBP prune.
- MLP trained every step for 30k steps.

## 5. Builds on
3DGS via gsplat 1.1.1 `simple_trainer.py`, with the robust loss replacing L1+D-SSIM, opacity reset
disabled, scale-based pruning effectively disabled, and the UBP CUDA backward addition.

## 6. Protocol
- Datasets: RobustNeRF (android, statue, yoda, crab(1)/(2)) and NeRF On-the-go (mountain, fountain,
  corner, patio, spot, patio-high). Train = `*clutter*` images, test = `*extra*` images (clean
  held-out views); RobustNeRF `*clean*` images unused (only for the "3DGS on clean" rows).
- Resolution: images_8 for all scenes (official `sls_benchmark.sh` uses default `data_factor=8`).
- 30k iterations, one image per batch, metrics PSNR/SSIM/LPIPS(VGG per paper; code uses torchmetrics LPIPS default = AlexNet).
- Paper numbers: mean of 5 runs (Fig. 8 gives ±σ).
- Preprocessed data with precomputed SD features: Borealis doi:10.5683/SP3/WOFXFT.

## 7. Ambiguities / paper-vs-code
1. GLO: paper uses it for all runs in Fig. 8 (including the 3DGS row); code does not implement it
   (README: gsplat reproduction "without enabling the appearance modeling"). README gsplat numbers
   (SLS-mlp+UBP): Android 24.96, Statue 22.65, Crab2 35.30, Yoda 35.26, Mountain 21.31,
   Fountain 22.30, Corner 26.36, Patio 22.2, Spot 25.12, Patio-High 23.00.
2. β1: paper 3e-4 (3e-3 for high-occlusion scenes); code 3e-3 everywhere.
3. UBP threshold 1e-8 vs 1e-14 (different gradient normalisation).
4. Histogram bin width 1e-3 (paper) vs 1e-4 (code, 10000 bins).
5. Lipschitz regulariser has no gradient in the code (`.data`).
6. Robust mask τ: paper footnote example 0.5; code 0.7.
7. SSIM: paper says 3DGS hyper-parameters; code uses ssim_lambda=0 (pure masked L1, Eq. 1).
8. LPIPS: paper VGG; code torchmetrics AlexNet.
9. Headline tables (Fig. 3/4/5) "SLS-mlp" = Fig. 8 "SLS-mlp w/ UBP" (identical numbers).
