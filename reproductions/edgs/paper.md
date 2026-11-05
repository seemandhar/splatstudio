# EDGS: Eliminating Densification for Efficient Convergence of 3DGS — implementation spec

arXiv 2504.13204 (CVPR 2026), Kotovenko, Grebenkova, Ommer. Project https://compvis.github.io/EDGS/.
Official code: https://github.com/CompVis/EDGS (cloned to `source/code/`, commit at clone time;
RoMa submodule pinned to `370117431ffc5dc000fb46f6e581b74bdb2c3ff8`, 3DGS submodule `54c035f`).
README: "Updated training code will be released soon" — the released code is the arXiv-v1 era code
and lacks the paper's SH initialization (see C4).

## 1. Method summary
- Replace SfM init + densification by a **dense one-shot initialization**: for K reference training
  views, compute dense RoMa correspondences to their nearest training views, triangulate each match,
  keep geometrically consistent ones, and place one isotropic Gaussian per sampled correspondence.
- SfM points are discarded (`add_SfM_init: False`).
- Then the standard 3DGS optimizer runs **without densification** (no clone/split, no opacity reset);
  only low-opacity pruning (< 0.005) before step 15000, a slow opacity decay, and a clamped means LR.

Contributions
- **C1 Dense matching from a 2D prior** (§3.2): RoMa warps + confidences from each reference view
  to its J nearest training views (camera proximity by Frobenius norm of W2C matrices).
- **C2 Splat triangulation** (§3.3): DLT least-squares triangulation of each match.
- **C3 Sampling distribution** (§3.4, Eq. sampling): confidence-thresholded (τ_corr) uniform sampling of
  pixels, aggregated over neighbours by max; reprojection-error filter τ_proj; best neighbour per pixel.
- **C4 SH initialization** (§3.5, supp. "Spherical harmonics initialization"): DC from the reference
  colour; higher-order SH from a pseudo-inverse fit to the multi-view colour observations.
  **Not in the released code** (released code sets `features_rest = 0`).
- **C5 Densification-free optimization** (§4 Baselines, supp. "Optimization"): 3DGS schedule, no
  densification; code: prune opacity<0.005 every step before 15k, opacity ×0.99 every 10 steps before 15k,
  `max_lr`: means LR uses `lr(max(step, 8000))`, opacity LR 0.025.

## 2. Equations (verbatim from `tex/approach.tex`)
Matcher: $\mathcal{M}(I^i, I^j) \to (\mathcal{W}^{i \rightarrow j}, \mathbf{c}^{ij})$,
$\mathcal{W}^{i\to j}\in\mathbb{R}^{2\times H\times W}$, $\mathbf{c}^{ij}\in\mathbb{R}^{H\times W}$.

Triangulation:
$A^T = [\bm{P}^i_{\text{col},0} - u_k^i \bm{P}^i_{\text{col},2};\ \bm{P}^i_{\text{col},1} - v_k^i \bm{P}^i_{\text{col},2};\ \bm{P}^j_{\text{col},0} - u_k^j \bm{P}^j_{\text{col},2};\ \bm{P}^j_{\text{col},1} - v_k^j \bm{P}^j_{\text{col},2}]$,
$\bm{g}_k^x := \arg\min_{\bm{x}} \|A \bm{x} + b\|^2$ with $\bm{x}=[x,y,z,1]$ (code: lstsq on the 3 xyz columns,
rhs = −4th column). $(u,v)$ are NDC coordinates of the 3DGS `full_proj_transform`.

Reprojection error: $\varepsilon_k^i = \|\pi(\bm{P}^i,\bm{g}^x_k) - (u^i_k,v^i_k)\|_2$,
$\varepsilon_k^{ij}:=\max(\varepsilon_k^i,\varepsilon_k^j)$. **Code uses L1** (`abs().sum(1)`) in NDC.

Sampling: $\mathbf{p}^{ij}_{\text{corr}} \sim \mathcal{U}(\{k \mid \mathbf{c}^{ij}(u^i_k,v^i_k) > \tau_{\text{corr}}\})$,
$\mathbf{p}^{ij}_{\text{proj}} \sim \mathcal{U}(\{k \mid \varepsilon^{ij}_k < \tau_{\text{proj}}\})$,
$\mathbf{p}^i(k) \propto \max_{j \in \mathbb{I}_i}(\mathbf{p}_{\text{corr}}^{ij}(k)\,\mathbf{p}_{\text{proj}}^{ij}(k))$,
$\mathbf{p}(k) \propto \bigcup_i \mathbf{p}^i_k$.
Code: $c^i(u,v)=\max_j c^{ij}(u,v)$ (supp. Eq. too); weights $w=c^i$ with $w[w>0.05]=1$;
`torch.multinomial(w, M, replacement=False)` (so below-threshold pixels keep weight $c^i<0.05$, not 0);
triangulate the sampled pixels with *every* neighbour, keep the neighbour with minimal
$\max(\varepsilon^i,\varepsilon^j)$; points with that error $>\tau_{proj}$ get opacity logit −10 (pruned at step 1).

SH: $\hat{\mathbf{H}}_k = \arg\min_{\mathbf{H}} \| \mathbf{Y}_k \mathbf{H} - \mathbf{O}_k \|_F^2$,
$\hat{\mathbf{H}}_k = \mathbf{Y}_k^{+}\mathbf{O}_k$ ($n<16$); index 0 replaced by the reference colour.

## 3. Hyper-parameters
| name | value | source |
|---|---|---|
| matcher | RoMa outdoor (`roma_model: outdoors`), coarse 560², `upsample_preds=False`, `symmetric=False` | code `configs/train.yaml`, `corr_init.py:533-539` |
| τ_corr | 0.05 (`roma_model.sample_thresh`) | paper supp.; code `corr_init.py:541` |
| τ_proj | 0.01 NDC (L1) | paper supp.; code `proj_err_tolerance` |
| num_refs | 180 (min with #train views), chosen by k-means on flattened W2C | paper; code `full_eval.py` |
| nns_per_ref | **3** (code full_eval) vs 2 (paper) | `full_eval.py:44-60` |
| matches_per_ref | **25000** Mip-NeRF360, 15000 T&T/DB (code) vs 20000 (paper) | `full_eval.py` |
| initial scale | 0.001·‖p − c_ref‖, then ×0.5 after init → 0.0005·dist, isotropic | `corr_init.py:670`, `trainer.py:242` |
| initial opacity | sigmoid(0)=0.5 (bad points logit −10) | `corr_init.py:665` |
| rotation | identity | copied from SfM init (identity) |
| SH rest | 0 (code) / pinv fit (paper) | `corr_init.py:659` / supp. |
| seed | 228 | `train.yaml` |
| iterations | 30000 | full_eval |
| position lr | 1.6e-4→1.6e-6 ·extent, evaluated at max(step, 8000) (`max_lr: True`) | `trainer.py:154-157` |
| feature/opacity/scaling/rotation lr | 0.0025 / **0.025** / 0.005 / 0.001 | `configs/gs/base.yaml` |
| λ_dssim | 0.2 | base.yaml |
| prune | opacity < 0.005 every step while step < 15000 | `trainer.py:252-258` |
| opacity decay | logit += log(0.99) every 10 steps while step < 15000 (`reduce_opacity: True`) | `trainer.py:78-82` |
| opacity reset | none (reset only lives in densify branch; interval 30000) | `trainer.py:177-191` |
| background | black (`white_background: false`), no random bg | base.yaml |

## 4. Schedules
SH degree +1 every 1000 steps (3DGS). Pruning and opacity decay stop at 15000. Means LR flat at
lr(8000) for steps <8000, then 3DGS exponential decay.

## 5. Builds on
3DGS (graphdeco 54c035f rasterizer, no antialiasing). Photometric loss 0.8·L1 + 0.2·D-SSIM unchanged.

## 6. Protocol
Mip-NeRF360 9 scenes (outdoor images_4, indoor images_2), T&T truck/train, DB drjohnson/playroom at
original resolution; llffhold8 (every 8th sorted image test; official split code `data_utils.py`);
init uses **training views only** (`scene.getTrainCameras()` after the split). 30k steps, A100.
LPIPS VGG (`lpips` package in trainer; 3DGS `metrics.py` for tables). Times include ~2 min init.

## 7. Ambiguities
- Paper (2NN, 20k matches) vs code (3NN, 25k matches for M360) — code values used.
- SH init (C4) absent from released code. Paper ablation: w/o SH init 27.80 vs full 28.02 PSNR (M360 avg).
  Our pinv details: observations = reference colour and best-neighbour colour at the matched pixel,
  directions = normalized (point − camera centre), target = rgb − 0.5 (3DGS SH-to-RGB offset), DC
  then overwritten by RGB2SH(reference colour).
- Error norm: paper L2, code L1 — code used.
- Below-threshold pixels: paper says uniform over thresholded set (zero weight outside); code keeps
  weight = confidence below 0.05 — code used.
- Reported "EDGS + 3DGS" table includes pruning; "3DGS*" row is the authors' retrained 3DGS.
