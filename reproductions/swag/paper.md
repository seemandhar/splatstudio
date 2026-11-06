# SWAG: Splatting in the Wild images with Appearance-conditioned Gaussians

Dahmani, Bennehar, Piasco, Roldão, Tsishkou (Huawei Noah's Ark Paris), ECCV 2024 submission,
arXiv **2403.10427** (v2). Project page https://dahmanihiba.github.io/SWAG/ ("Code (TBA)").
**No official code**: the only `dahmanihiba/SWAG` GitHub repo is the project web page
(index.html + videos, checked 2026-11-05, cloned to `source/code/`); GitHub repo search for
"SWAG splatting wild" / "SWAG gaussian appearance" returns nothing else. Everything below is from
the LaTeX source (`source/latex/main.tex`, `supp/hyperparameters.tex`, `tables/*.tex`).

## 1. Method summary
- 3DGS + a per-image appearance embedding `l_I` (dim 24) + a multi-resolution hash grid on the
  Gaussian centres `emb(x)` (12 levels × 2 features = 24) + a small MLP `F_θ`.
- `F_θ(c, emb(x), l_I)` (input 3 + 24 + 24 = 51, output 4) returns the image-conditioned colour
  `c^I` and an opacity-variation location `Δα^I` per Gaussian and per image.
- `Δα^I` parameterises a Binary-Concrete sample `Δα̃^I` (T = 0.1, U ~ U(0,1) at train, U = 0.5 at
  eval); the image opacity is `α^I = max(α − Δα̃^I, 0)` → transient Gaussians.
- Transient Gaussians = `Var_I[Δα̃^I] ≠ 0` over training images (used for occluder removal, not
  for the metrics).
- Loss: unchanged 3DGS L1 + D-SSIM. Evaluation: NeRF-W protocol (optimise a test embedding on
  the left image half, metrics on the right half).

Contributions:
- **C1 Appearance-conditioned colour** (§4.1, Eq. 7): MLP on SH colour + hash-encoded centre +
  image embedding predicts the colour directly (not an affine transform).
- **C2 Transient opacity variation** (§4.2, Eq. 8–11): Binary-Concrete image-dependent opacity
  reduction, unsupervised.
- **C3 Transient/static decomposition** (§4.2, Fig. 4): variance of `Δα̃^I` over training images.
- (protocol) NeRF-W test-time appearance optimisation (§5.3).

## 2. Equations (verbatim LaTeX)
- Eq. 7 `\mathbf{c^{I}}=\mathcal{F}_{\mathbf{\theta}}(\mathbf{c},\mathbf{emb(x)}, l_I).`
- Eq. 8 `(\mathbf{c^{I}},\Delta\mathbf{{\alpha}^{I}})=\mathcal{F}_{\mathbf{\theta}}(\mathbf{c},\mathbf{emb(x)},l_I),`
- Eq. 9 `\Delta \mathbf{\Tilde{\alpha}^{I}}=\text{sigmoid} \left[ \frac{1}{\mathbf{T}} \left ( log \left ( | \Delta \mathbf{{\alpha}^{I}}| \right ) +log(\mathbf{U})-log(1-\mathbf{U}) \right ) \right ],`
- Eq. 10 `\mathbf{U} \sim \text{Uniform}(0,1),` — "we set T = 0.1 … During the evaluation, we fix U at 0.5."
- Eq. 11 `\mathbf{{\alpha}^I} =\max \left ( \mathbf{{\alpha}} - \Delta \mathbf{\Tilde{\alpha}^{I}}, 0 \right ).`
- Transient set: `\mathbf{Var} \left [ \Delta\mathbf{\Tilde{\alpha}^{I}} \right ] \neq 0` (Discussion: optionally `< λ`).
- Loss (§5.1): "Similarly to 3DGS, we minimize the L1 loss combined with D-SSIM term" (3DGS:
  `(1-λ)L1 + λ D-SSIM`, λ = 0.2).

`c` = Gaussian colour from its SH coefficients (view-dependent; the paper adds it "to render
view-dependent specularities"), `emb(x)` = hash-grid encoding of the centre `x`, `l_I` = image
embedding, `α` = the Gaussian's learned (sigmoid) opacity.

## 3. Hyper-parameters
| name | value | source |
|---|---|---|
| iterations | 60 000 | supp. Tab. 7 |
| densification interval | 2 000 | supp. Tab. 7 |
| densification end | 30 000 | supp. Tab. 7 |
| image embedding dim / lr | 24 / 0.0025 | §5.1, supp. Tab. 7 |
| hash table size T | 2^19 | §5.1, Tab. 7 |
| N_max / N_min / levels L | 2048 / 16 / 12 | §5.1, Tab. 7 |
| hash grid lr | 0.0025 | Tab. 7 |
| MLP layers / hidden / in / out | 3 / 64 / 51 / 4 | §5.1, Tab. 7 |
| MLP lr | 0.0025 | Tab. 7 |
| concrete temperature T | 0.1 (no annealing) | §4.2 |
| eval U | 0.5 | §4.2 |
| everything else | 3DGS defaults (implied) | — |

Features per hash level are not stated; 51 = 3 + 24 + 24 implies 2 per level (tcnn default).

## 4. Schedules
Only the densification schedule above (every 2000 it until 30 000) and 60 000 iterations. SH
degree, opacity reset, position-LR decay, densify start are not given (3DGS defaults implied).

## 5. Builds on
Original 3DGS (Inria code, SfM init, isotropic 3-NN init, adaptive density control, L1+D-SSIM).
Unchanged: rasteriser, densification criteria, optimiser for Gaussian params.

## 6. Protocol
- Photo Tourism (IMC 2020): Brandenburg Gate, Sacré-Cœur, Trevi Fountain; "Testing is performed
  on a standard set that is free of transient occluders" → NeRF-W split tsv files.
- NeRF-W evaluation: embedding optimised on the left half of each test image, metrics on the
  right half. Steps / lr / init of that optimisation not given.
- Resolution not given (NeRF-W / Ha-NeRF / CR-NeRF use ×2 downsampling; 3DGS default = full
  resolution unless width > 1600 px; all PT images here are ≤ 1600 px wide).
- PSNR/SSIM/LPIPS implementation (LPIPS net) not given. Train time 0.83 h, 15.29 FPS on a
  "high-tier GPU".
- Also NeRF-OSR (not reproduced: dataset/splits not available here) and Lego perturbations.

## 7. Ambiguities (no code to resolve them → assumptions in plan.md)
1. MLP: "3 layers of 64 hidden units" — 3 hidden layers or 3 linear layers; activations; output
   activation of `c^I` (sigmoid assumed) and of `Δα^I` (raw, `|·|` in Eq. 9).
2. Hash-grid position normalisation (bounding box) and initialisation; features/level (2).
3. Whether `c` is the view-dependent SH colour (assumed yes) and whether it is detached.
4. Embedding initialisation; test-time optimisation steps/lr/init and loss.
5. Opacity reset, SH schedule, densify start, position LR decay length for 60k iterations.
6. Background colour (3DGS default black), image resolution, metric implementation.
7. Whether densification gradients come from the image-conditioned render (yes — only render).
8. Eq. 9 at `Δα^I = 0` → log 0; a small epsilon is needed.
