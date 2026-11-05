# Trim 3D Gaussian Splatting for Accurate Geometry Representation (TrimGS)

arXiv 2406.07499 (Fan, Yang, Li, Li, Zhang; NeurIPS 2024 submission). Project https://trimgs.github.io/.
Official code: https://github.com/YuxueYang1204/TrimGS @ 6da2f212 (`Trim3DGS/`), modified rasterizer
https://github.com/Abyssaledge/diff-gaussian-rasterization @ 58790196. Sources in `source/`
(`latex/`, `paper.pdf`, `code/`, `rast/`).

We reproduce **Trim3DGS** (the 3DGS-based variant). Trim2DGS (2DGS-based) is a separate variant
and is not built here.

## 1. Method summary
- Start from a converged (30k-iteration) 3DGS model and fine-tune it for 7k iterations (§5.1
  "fast 7K-iteration optimization based on their pretrained models").
- Every 1000 fine-tune iterations, remove the 10% of Gaussians with the lowest *multi-view
  contribution* (normalized alpha-blending weight, top-5 views).
- Replace gradient-driven densification by *scale-driven* splitting: every Gaussian whose max
  scale exceeds a scene threshold is split into K=3 smaller ones.
- Add a normal-consistency loss between rendered Gaussian normals (shortest axis) and normals
  derived from the rendered depth map, plus (code only) a k-NN normal smoothness term.
- **Code-only, undocumented in the paper:** both the 30k "3DGS*" pre-training and the fine-tune use
  `FlattenGaussianModel`: the first scale axis is fixed at 1e-8 (flat, surfel-like 3D Gaussians),
  and the Gaussian normal is the first rotation column (`code:Trim3DGS/scene/gaussian_model.py:503-519`;
  `train.py:34`, `tune.py:155`). We *assume* the paper's "3DGS*" row (Table 3, "our re-implementation") is
  this flat 3DGS (see notes).

Contributions:
- **C1 contribution-based trimming** (§4.1, Eq. 4–6; kernel `rast/cuda_rasterizer/forward.cu:356-360`,
  `tune.py:prune_low_contribution_gaussians`).
- **C2 scale-driven densification** (§4.2; `gaussian_model.py:densify_and_scale_split`, `tune.py:culling`).
- **C3 normal regularization** (§4.3, Eq. 7–8; `tune.py:normal_regularization`).
- **C0 (code) flat Gaussians + two-stage protocol** (30k flat 3DGS → 7k Trim fine-tune).

## 2. Equations
Alpha blending (Eq. 1): `I_p = \sum_{i\in\Lambda} c_i \alpha_i \prod_{j=1}^{i-1}(1-\alpha_j)`.

Single-view contribution, unnormalized (Eq. 4):
`\mathbf{C}_k = \sum_{p \in \mathbb{P}_{k}}\alpha_{i(p)}\prod_{j=1}^{i(p)-1}(1-\alpha_{j})`

Normalized (Eq. 5):
`\mathbf{C}_k = \frac{1}{|\mathbb{P}_{k}|}\sum_{p \in \mathbb{P}_{k}}\left(\alpha_{i(p)}\right)^{\gamma}\left(\prod_{j=1}^{i(p)-1}(1-\alpha_{j})\right)^{(1 - \gamma)}`

**Code (authoritative):** `trans = pow(alpha, 2*GAMMA) * pow(T, 2 - 2*GAMMA)` with `GAMMA 0.5`
(`rast/cuda_rasterizer/auxiliary.h:25`), i.e. the paper's summand squared; at γ=0.5 the summand is
exactly the blend weight `alpha*T`. It is accumulated (atomicAdd) together with a pixel count for
every pixel of the Gaussian's tiles where the conic exponent `power <= 0`, *before* the
`alpha < 1/255` skip and before the `T < 1e-4` early termination; `alpha = min(0.99, o*exp(power))`.
Python divides: `transmittance_sum / (num_covered_pixels + 1e-6)` (`gaussian_renderer/__init__.py`).
`|P_k|` = number of such pixels.

Multi-view (Eq. 6): `\mathbf{C} = \frac{1}{|\mathbb{V}|}\sum_{k \in \mathbb{V}}\mathbf{C}_k`, V = the
top-5 views of that Gaussian (code: running top-K over all training views, K=5; views where the
Gaussian is invisible contribute 0). Trim: `prune C < quantile(C, 0.1)`.

Scale-driven densification (§4.2, code `densify_and_scale_split(no_grad=True)`):
select = scene_mask ∧ (max(scale) > percent_dense·extent·scale_factor ∨ max_radii2D > 100);
split each selected Gaussian into N=3: means = μ + R·(ε ⊙ s), ε~N(0,I) (s includes the flat
1e-8 axis), scales = s/(0.8·N), other attributes copied; originals removed; then prune
`opacity < 0.005`. `scene_mask` (`tune.py:culling`, expansion 2): axis-aligned box around the
camera-centre mean with half-size = camera span along x, y and — **code quirk** — along z the x span.

Normal from depth (Eq. 7): `\bn = \frac{1}{|\mathbb{T}|}\sum_{\bt_i, \bt_j \in \mathbb{T}} \bt_i \times \bt_j`.
Code: back-project the rendered (alpha-blended camera-z, background 0) depth map to world
points P; with dilation d=2 and replicate padding,
n1 = norm((P[y+d,x+d]−P[y−d,x−d]) × (P[y−d,x+d]−P[y+d,x−d])),
n2 = norm((P[y,x+d]−P[y,x−d]) × (P[y−d,x]−P[y+d,x])), n = norm(n1+n2), flipped to face the
camera. Gaussian normal = R[:,0] flipped to face the camera, rendered by alpha blending
(background 0) and normalized per pixel.

Loss (Eq. 8): `\mathcal{L} = \alpha_1 \mathcal{L}_c + \alpha_2 |\mathbf{N}_D - \mathbf{N}_G|_1`.
Code: `L = L_c + w·(L_normal + L_smooth)` with
- `L_normal = mean(|(N_G+1)/2 − (N_D+1)/2| · M)`, M = [Sobel|∇D|/3 < 0.05] eroded by a 3×3 min-pool;
- `L_smooth = mean(|(n_i+1)/2 − (norm(mean_{j∈kNN16(i)} sgn(n_j·n_i)·n_j)+1)/2|)` over visible Gaussians whose
  camera z is within 1.0 of the depth map sampled at their projection; the sign is detached; the
  16-NN (including self) is recomputed every 500 iterations and after every densification;
- a NaN normal loss is skipped (set to 0).
`L_c` = 0.8 L1 + 0.2 D-SSIM.

## 3. Hyper-parameters
| name | value | source |
|---|---|---|
| pre-training | 30000 it vanilla 3DGS, flat (eps 1e-8) | `scripts/run_Mipnerf360.py`, `train.py:34` |
| fine-tune iterations | 7000 | paper §5.1, `FinetuneParams.iterations` |
| trimming interval / ratio | 1000 / 10% | paper §5.1; cmdline `--contribution_prune_interval 1000`, `contribution_prune_ratio 0.1` |
| trimming start | `contribution_prune_from_iter 0` → t=1000..6000 (6 times) | cmdline |
| top-K views | 5 | paper §4.1, `tune.py` K=5 |
| γ | 0.5 (code exponents 2γ, 2−2γ) | Table 6 default, `auxiliary.h` |
| densification interval | 1000 (cmdline overrides 100), from 0, until 7000 | `run_Mipnerf360.py`, `FinetuneParams` |
| split mode | "scale" (no gradient criterion), N=3, shrink 1/(0.8·3) | `tune.py:242`, `densify_and_scale_split` |
| τ_s | percent_dense·extent·scale_factor = 0.01·extent·1.0 | `FinetuneParams` |
| screen-size split | max_radii2D > 100 px (radii reset after each densification) | `tune.py:243` |
| opacity prune | 0.005 (inside scale split) | `tune.py:243` |
| opacity reset | none in fine-tune (`--opacity_reset_interval 999999`) | cmdline |
| normal weight w | 0.1 outdoor (bicycle flowers garden stump treehill), 0.01 indoor (bonsai counter kitchen room) | `run_Mipnerf360.py:normal_weights` (paper Eq. 8: α2 unspecified) |
| normal start | 0 (from first fine-tune iteration) | `normal_regularity_from_iter` |
| depth grad thresh / mask dilation | 0.05 / 1 | `FinetuneParams` |
| normal dilation | 2 | `FinetuneParams` |
| close thresh | 1.0 (world units) | `FinetuneParams` |
| kNN K / reset | 16 / every 500 it + after densification | `knn_to_track`, `neighbor_reset_interval` |
| xyz LR (fine-tune) | 1.6e-6·extent constant (init = final) | cmdline `--position_lr_init 0.0000016` |
| other LRs | 3DGS defaults (f_dc 2.5e-3, f_rest 1.25e-4, opacity 0.05, scaling 5e-3, rotation 1e-3) | `FinetuneParams` |
| optimizer | fresh Adam (eps 1e-15) at fine-tune start | `tune.py:training_setup` |
| SH | degree 3 throughout fine-tune | `load_ply` sets active degree = max |
| background | black | `white_background False` |

## 4. Schedules
Pre-training 1..30000: 3DGS (densify 500..15000 every 100, opacity reset every 3000, SH +1 per
1000, xyz LR exp-decay 1.6e-4→1.6e-6 ·extent). Fine-tune t=1..7000: normal loss on from t=1;
at t ∈ {1000,…,6000}: scale split → opacity prune → contribution trimming → kNN reset;
kNN also reset at every t%500==0; constant xyz LR.

## 5. Builds on
3DGS (Kerbl et al. 2023) with flat Gaussians. Photometric loss, SH, densification of the
pre-training stage, LRs unchanged from 3DGS.

## 6. Protocol (Mip-NeRF 360 rendering table, Table 3)
9 scenes, `-i images_4` (outdoor) / `images_2` (indoor) precomputed images, `--eval` llffhold 8,
37k total iterations (30k + 7k), black background, metrics from 3DGS `metrics.py` (LPIPS VGG).
Per-scene PSNR/SSIM/LPIPS reported for 3DGS*, SuGaR, 2DGS*, Trim2DGS, Trim3DGS.
Geometry: DTU Chamfer (15 scans, TSDF mesh of median depth; Table 1) and point Chamfer (Table 2),
TnT F-score only for Trim2DGS (README). Not reproducible here (see report).

## 7. Ambiguities
- Paper Eq. 5 vs. code exponent (code squares the summand). Code followed.
- Flat Gaussians are never mentioned in the paper; code uses them for both stages. Followed.
- The paper's α2 is not given; code uses per-scene 0.1 / 0.01. Followed (two method names).
- The smoothness kNN term is code-only. Followed.
- Culling box z-extent uses the x span (likely a bug). Followed verbatim.
- Rasterizer clamp 0.99 (official) vs gsplat 0.999; tile/radius computation of gsplat used.

## Documentation notes (post-review)
- **Assumption, not established:** we treat our 30k flat pre-train (`trimgs --steps 30000`) as a stand-in for the paper's "3DGS*". The paper calls 3DGS* "our re-implementation" and says it fine-tunes from 3DGS pretrained models. The official scripts pre-train with `FlattenGaussianModel` (flat 3DGS). Our pre-train uses gsplat `DefaultStrategy` (as in our baseline), not the 3DGS densifier, so it may differ from the authors' 3DGS*.
- **Contribution weight deviation:** paper Eq. 5 uses α^γ·T^(1−γ). The official kernel computes α^(2γ)·T^(2−2γ) (the paper's term squared; at γ=0.5 this is α·T). We follow the code.
- **Top-5 views quirk depends on view order:** the official list only inserts a view that beats the current maximum and is initialised with K copies of the first view. It therefore holds the last K record-breaking values, not a true top-K, and the result depends on the order the training views are visited. The official order comes from Scene's seeded `random.shuffle`. Ours is a `random.Random(0)` shuffle of SplatStudio's train list, so the per-Gaussian values (and which Gaussians get trimmed) can differ from the official run.
