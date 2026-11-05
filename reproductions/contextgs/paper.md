# ContextGS: Compact 3D Gaussian Splatting with Anchor Level Context Model — implementation spec

- arXiv **2405.20721** (NeurIPS 2024). Yufei Wang, Zhihao Li, Lanqing Guo, Wenhan Yang, Alex C. Kot, Bihan Wen.
- Sources (untrusted, never executed): `source/latex/neurips_2024.tex` (main + appendix), `source/paper.pdf`,
  `source/paper.txt`, official code `source/code/` (github.com/wyf0912/ContextGS, commit `eff8d19` "code release").
  The repo's `results/<dataset>/<scene>.csv` files hold the authors' per-scene numbers (bytes); they agree with the
  appendix tables.
- The repo does **not** contain `submodules/` (README says "unzip diff-gaussian-rasterization.zip"; the zips are
  absent). The rasterizer is the Scaffold-GS fork with `visible_filter` (same API as HAC's); not readable here.
- `code:<file>:<line>` cites `source/code/`. **Where code and paper disagree, the code is used**; both are listed.

---------------------------------------------------------------------------------------------------
## 1. Method summary and contributions

- Backbone = **Scaffold-GS with HAC-style entropy coding** (no hash grid). Anchors: position x, feature f ∈ R^50,
  scaling l ∈ R^6, K=10 offsets O ∈ R^{10×3}, offset masks (Compact3DGS-style masking loss), view-dependent MLP decode
  into neural Gaussians. Attributes f, l, O are quantized with adaptive steps Q = Q0(1+tanh r) and entropy coded under
  a predicted Gaussian (mean, scale, r all predicted).
- **C1 Anchor partitioning (§4.1, Eq. 4–6)**: anchors are split into K=3 disjoint levels by voxelizing at coarser
  voxel sizes ε_k = κ_k ε; the representative of each coarse voxel is the anchor with the minimum index; level sets
  are made disjoint (V² = V̂², V¹ = V̂¹ \ V̂², V⁰ = V̂⁰ \ V̂¹). κ_k found once by **binary search** to hit a target ratio
  τ = 0.2 between adjacent levels.
- **C2 Anchor-level context model (§4.2, Eq. 7–8)**: coding order level 2 → 1 → 0; per-level MLP F^k predicts
  (μ, σ, Δ) of f, l, O for level-k anchors from the already decoded (quantized) feature + scaling (+ position) of the
  parent anchor in level k+1; coarsest level uses position only.
- **C3 Hyperprior (§4.2, Eq. 9)**: a learnable per-anchor latent z ∈ R^{50/4=12}, coded with a non-parametric fully
  factorized density (Ballé 2018 EntropyBottleneck, compressai); its quantized value ẑ is appended to every level's
  prior: ψ̂ = [ẑ; ψ].
- **C4 Objective + bitstream (§4.3, Eq. 10–11, App.)**: L = L_scaffold + λ_e L_entropy + λ_m L_m; real arithmetic
  coded bitstream (anchors 16-bit, not entropy coded in the main table), decoded-model evaluation.

---------------------------------------------------------------------------------------------------
## 2. Equations (verbatim LaTeX)

Scaffold-GS decode (§3): `\{\c^i, \r^i, \s^i, \alpha^i\}_{i=0}^k = F(\f, \bm{\sigma}_{c}, \dvc)`,
`\bm{\sigma}_{c} = ||\x-\x_c||_2`, `\dvc=\frac{\x-\x_c}{||\x-\x_c||_2}`;
`\{\bm{\mu}^0, ..., \bm{\mu}^{k-1}\} = \x + \{\O^0, ..., \O^{k-1}\}\cdot \l`.

Eq. 4 (partition):
```latex
\hat{\V}^k = \left\{M(\{\v_i^{k-1}: \hat{\x}_i^{k} = \hat{\x}\}): \hat{\x} \in  \left\{\hat{\x}^{k}_i: i=1,2,...,|\hat{\V}^{k-1}|\right\}  \right\}
```
Eq. 5:
```latex
\hat{\x}_i^{k} = \left\lfloor \frac{\x^{k-1}_i}{\epsilon^{k}} \right\rceil \epsilon^{k}, \quad \x_i^{k-1} \in \hat{\V}^{k-1};\qquad
M(\{\v_i^{k-1}: \hat{\x}_i^k = \hat{\x}\}) = \v^{k-1}_{j} \text{ s.t. } j=\min\{i: \hat{\x}_i^k = \hat{\x}\}
```
Eq. 6: `\V^2 = \hat{\V}^2, \V^1 = \hat{\V}^1 \backslash \hat{\V}^2, \V^0 = \hat{\V}^0 \backslash \hat{\V}^1`;
ε_i = κ_i ε, κ by binary search so that |V^{i+1}|/|V^i| ≈ τ.

Eq. 7 (per-level conditional):
```latex
p_{\f^k}(\f_{i}^k|\bm{\psi}_{i}^{k}) = (\mathcal{N} (\bm{\mu}_{i}^k,  \bm{\sigma}_{i}^k) * \mathcal{U}(-\tfrac{1}{2} \Delta_{i}^k,\tfrac{1}{2} \Delta_{i}^k))(\f_{i}^k), \quad
\bm{\mu}_{i}^k, \bm{\sigma}_{i}^k, \Delta_{i}^k = F_{\f_i}^k (\bm{\psi}_i^k)
```
Eq. 8 (prior): `\bm{\psi}_i^k = [\x^k_{i}]` for k = K-1, `[\f_{j}^{k+1}; \l_{j}^{k+1};\x^k_{i}]` for k < K-1.

Eq. 9 (hyperprior):
```latex
p_{\tilde{\z}|\Theta}(\tilde{\z}_i|\Theta) = \prod_{j=0}^{50//h_c-1}{ \left(p_{z_i^j|\Theta^{(j)}}(\Theta^{(j)})*\mathcal{U}\left(-\tfrac{1}{2}, \tfrac{1}{2}\right)\right)(\tilde{z}_i^j)}
```
final prior `\hat{\bm{\psi}}_{i}^k = [\hat{\z}_i^k; \bm{\psi}_i^k]`.

Eq. 10–11:
```latex
\L = \L_{scaffold} +  \lambda_e \L_{entropy} + \lambda_m \L_m
\L_{entropy} = \mathbb{E} [-\log_2 p_{\tilde{\z}|\Theta}(\tilde{\z}_i|\Theta)] + \sum_{i=0}^{K-1}\left[-\log_2\left[\prod_{\f\in\{\f^k,\l^k, \O^k\}} p_{\f}(\f_{i}|\hat{\bm{\psi}}_i^k) \right]\right]
```
App. A: λ_e uses HAC's normalisation "additionally utilizing the dimension of anchor features as the divisor".

### Code form (authoritative)
- `code:gaussian_renderer/__init__.py:25-110` (`generate_neural_gaussians`):
  - step ∈ (3000, 10000]: visible-anchor f, l, O + U(-½,½)·Q0 (Q0: feat 1, scaling 0.001, offsets 0.2);
  - step == 10000: `update_anchor_bound()` (min·1.2 / ·0.8 by sign);
  - step > 10000: `multi_scale_generating(..., predict_bpp=True, training=True)` on **all** anchors, then select
    the visible ones.
  - Eval (not decoded): same with `training=False` → STE rounding; decoded model: attributes used as-is.
- `code:scene/gaussian_model.py:1541-1707` (`multi_scale_generating`):
  1. `hyper_feat, likelihood_hyper = latent_codec(hyper, training)` (compressai EntropyBottleneck(12)): noise
     quantization in training, round(z − median)+median otherwise.
  2. `level_scale` = `find_divide_scale(anchor[mask_anchor], τ=0.2, 3)` once (first call, step 10001; then fixed and
     stored with the MLPs); `divide_levels`: level 1 voxelizes `anchor*mask_anchor` (masked anchors moved to the
     origin) at `voxel_size*level_scale[0]`, level 2 voxelizes level-1 representatives at `voxel_size*level_scale[1]`;
     representative = min index (`torch_unique_with_indices`, `scatter_reduce amin`).
  3. For i = 2, 1, 0: to-code anchors of level i (not in level i+1); `feat_in = [anchor_i, ẑ]` (i = 2) or
     `[content_pre_gathered, ẑ]`; `mlp_grid[i]` (Linear(in, 100) ReLU Linear(100, 175)) → mean/scale for f(50), l(6),
     O(30) and 3 Q-adjust scalars; Q = Q0(1+tanh r).clamp(1e-9); training: x + U(-½,½)·Q; eval: STE round to Q.
  4. `extract_context_feat`: `content_coded = [anchor, feat_after_Q, scaling_after_Q]` (59-D) gathered at the
     **parent** index for the next level's to-code anchors (so the position in the context is the parent's, not the
     anchor's own — differs from Eq. 8).
  5. Bits on a random 15% (`chosse_random_thresh = 0.15`) of masked anchors:
     `bit_hyper = -log2 likelihood_hyper`, `bit_feat/scaling/offsets = Entropy_gaussian(x, μ, σ, Q, x_mean)` with
     x_mean = mean(_anchor_feat) / mean(get_scaling) / mean(_offset); offsets bits × binary mask.
     `bit_per_param = (Σb_feat + Σb_scaling + Σb_offsets + Σb_hyper) / (numel_feat + numel_scaling + numel_offsets)
      · mask_anchor_rate` (this per-parameter mean is the "dimension divisor" of App. A).
- `Entropy_gaussian` (`code:utils/entropy_models.py:31-50`): clamp x to x_mean ± 15000Q; σ ≥ 1e-9;
  `bits = -log2(Low_bound(|Φ((x+Q/2-μ)/σ) − Φ((x−Q/2−μ)/σ)|))`, Low_bound 1e-6 with pass-through gradient rule.
- `code:train.py:225-232`: `loss = (0.8 L1 + 0.2 D-SSIM) + 0.01·mean(prod(scaling of neural Gaussians))`;
  if bits: `+ λ·bit_per_param + 5e-4·mean(sigmoid(mask))`.
- compressai EntropyBottleneck (v1.2.x, BSD-3-Clause-Clear; read from the sdist): filters (3,3,3,3), init_scale 10,
  tail_mass 1e-9, likelihood bound 1e-9 (LowerBound op, same pass-through gradient), cumulative logits
  `logits ← softplus(M_i)·logits + b_i; logits += tanh(a_i)·tanh(logits)`, likelihood
  `sigmoid(c(x+½)) − sigmoid(c(x−½))`; medians = `quantiles[:, :, 1]`, quantiles init [−10, 0, 10]. **ContextGS never
  calls the aux loss**, so quantiles (and medians = 0) never change.

---------------------------------------------------------------------------------------------------
## 3. Hyper-parameters

| name | value | source |
|---|---|---|
| iterations | 30000 | paper §5.1; `train_scripts/train_mlp360.py` |
| feat_dim / n_offsets | 50 / 10 | `arguments/__init__.py:48-49`, §5.1 |
| voxel_size | 0.001 (mip360), 0.01 tandt, 0.005 db | `train_scripts/*.py` |
| update_init_factor / depth / hierarchy | 16 / 3 / 4 | args, scripts |
| level_num K | 3 | `train.py --level_num 3`, §5.1 |
| target_ratio τ | 0.2 | scripts `--target_ratio 0.2`, ModelParams default 0.2, §5.1 |
| hyper_divisor h_c | 4 (z ∈ R^12) | ModelParams, §5.1 |
| λ_e (lmbda) | 0.004 low-rate, 0.0005 high-rate (mip360/tandt/db); bungee 0.001 high | App. tables, scripts |
| λ_m | 5e-4 | train.py:231, §5.1 |
| scaling reg | 0.01·mean(prod s) | train.py:227 |
| λ_dssim | 0.2 | args |
| Q0 feat/scaling/offsets | 1 / 0.001 / 0.2 | renderer, gaussian_model |
| rate sample | 15% of masked anchors | gaussian_model.py:1658 |
| noise phase | 3000 < step ≤ 10000 | renderer |
| entropy phase | step > 10000; bound update at 10000 | renderer |
| anchor stats/growing | start_stat 500, update_from 1500, interval 100, until 15000, pause [3000, 4000) | args, train.py |
| min_opacity / success_thr / grad_thr | 0.005 / 0.8 / 2e-4 | args |
| lr offset | 0.01 → 1e-4 (×spatial_lr_scale) | args |
| lr mask | 0.01 → 1e-4 (×spatial_lr_scale) | args |
| lr feat / hyper_latent / scaling | 0.0075 / 0.0075 / 0.007 | args |
| lr mlp_opacity / cov / color | 0.002→2e-5 / 0.004→0.004 / 0.008→5e-5 | args |
| lr latent_codec | 0.005 → 1e-5 | args (delay_mult 0.33 unused: delay_steps 0) |
| lr mlp_grid | 0.005 → 1e-5 | args |
| anchor position lr | 0 | args |
| Adam eps | 1e-15 | gaussian_model.py:475 |
| encode batch | 1000 anchors (10000 for hyper) | conduct_encoding |
| anchor quantization | 16 bit min–max in the bounds | encodings.py |

## 4. Schedules
- 1..3000 Scaffold-GS only; 3001..10000 uniform noise with fixed Q0 on visible anchors; 10000 bound update;
  10001.. context + hyperprior + rate loss (all anchors through the context model each step);
  level scales searched once at 10001.
- Anchor growing/pruning every 100 steps in (1500, 15000) except [3000, 4000); stats from step 501.
- All exponential LR decays over 30000 steps (`get_expon_lr_func`, delay 0).
- At 30000: estimate → encode → decode → render the **decoded** model for test metrics.

## 5. Builds on
Scaffold-GS + HAC's quantization/entropy scheme (HAC's code acknowledged; `utils/encodings.py` is HAC's) with the
hash grid replaced by the level context + hyperprior. Unchanged: Scaffold MLPs and decode, anchor growing/pruning,
L1+D-SSIM, scaling regulariser, masks.

## 6. Protocol
- Mip-NeRF 360 (all 9 scenes), Tanks&Temples (truck, train), Deep Blending (drjohnson, playroom), BungeeNeRF.
- Resolution: `-r -1` default → images wider than 1600 px are **PIL-resized to width 1600** from `images/`
  (`code:utils/camera_utils.py:21-35`) — identical to HAC/Scaffold → our `mipnerf360-1600` conversion applies
  (garden, bicycle, counter available).
- Split: sorted names, every 8th test (`llffhold=8`, `lod 0`). Background black (white_background False).
- Metrics: PSNR, SSIM (3DGS same-padding), LPIPS-VGG (`lpips` package), on the decoded model; size = bitstream
  files (meta + hyper + anchor + feat + scaling + offsets + masks + fp32 MLPs whose names contain `mlp`).
  Size unit is MiB (bytes / 2^20; repo CSV bicycle 22876468 B = 21.82).

## 7. Ambiguities / code-vs-paper
1. **Context position**: Eq. 8 uses the anchor's own position x_i^k; code concatenates the *parent's* position
   (content_coded gathered at the parent index). Code used.
2. **Level-1 context order**: `extract_context_feat` gathers parents for the to-code anchors of the next level in
   increasing *original index* order, while level-1 to-code anchors are enumerated in `torch.unique` voxel order.
   For level 0 the orders coincide; for level 1 they generally do not, so level-1 anchors receive the context of
   other anchors. It is consistent in training, encoding and decoding (lossless); the port reproduces it verbatim
   (and a probe checks the behaviour is the official one).
3. Masked anchors in training are moved to the origin before the level-1 voxelization (`anchor*mask`), so they can
   collapse into one voxel — reproduced.
4. Hyperprior coding uses compressai's rANS (16-bit CDF, escape for |ẑ| > 10 because quantiles stay at init); bits
   in training are −log2 of the factorized likelihood.
5. EntropyBottleneck parameters are not counted in the size (`get_mlp_size` counts names containing `mlp`).
6. Main-table high-rate Mip-NeRF 360 (27.75 / 0.811 / 0.231 / 18.41 MB) disagrees with App. Table (27.72 / 0.811 /
   0.231 / 21.58 MB, λ=0.0005); low-rate rows agree (27.62 / 0.808 / 0.237 / 12.68). App. text says high-rate λ for
   BungeeNeRF is 0.001.
7. The anchor-position-coding variant (App. Table "w/ encoding anchors") is not the main-table setting; anchors are
   stored as 16-bit integers (main table).
8. The rasterizer fork's source is not in the repo; its `visible_filter` is taken to be the Scaffold-GS one (same as
   HAC's fork).
