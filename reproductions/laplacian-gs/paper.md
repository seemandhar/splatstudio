# Laplacian-GS — Laplacian Frequency Hierarchies for Efficient 3D Gaussian Splatting Training

Yang, Zhang, Yan, Shi, Wang (HIT Shenzhen / XGRIDS). arXiv 2609.03334 (Pacific Graphics 2026).
Official code: https://github.com/SorenZhang574/Laplacian-GS (MIT), commit 33af46d, cloned to
`source/code/` (read only). Key files: `train_lap.py`, `render_pn.py`, `laplacian/laplacian_torch.py`,
`scene/gaussian_model.py`, `gaussian_renderer/__init__.py`, `submodules/*_pn` (signed rasterizer),
`run_mipnerf360_1k_{taming,fastgs}.sh`, `arguments/__init__.py`.

## 1. Method summary
- Each training image is decomposed into an L-level Gaussian (resolution) pyramid {I^l}; the
  scene is a hierarchy of independent Gaussian fields G^{L-1} (low-frequency base, rendered at
  the coarsest pyramid resolution) and G^l, l<L-1 (signed high-frequency residual fields
  rendered at pyramid level l's resolution).
- Training is frequency-staged, coarse-to-fine: G^{L-1} is trained first; then it is
  **archived** (frozen, removed from optimisation / GPU) and G^{L-2} is created by inheriting
  only the xyz of G^{L-1}, and trained; etc. Default L=2, iteration split 10K/20K (30K total).
- Stage loss is **reconstructed-image supervision**: Î^l = Up(Î^{l+1}) + R̂^l compared to the
  GT pyramid level I^l with the standard 3DGS photometric loss (L1 + 0.2 D-SSIM).
- Inference: render every field at its level's size and compose in the image domain with
  Laplacian reconstruction.
- Plug-and-play on Taming-3DGS and FastGS backbones (paper never reports it on vanilla 3DGS).

Contributions
- C1 Laplacian frequency hierarchy: per-level Gaussian fields, signed residual renderer, image-
  domain Laplacian composition (Sec. 3.2, Eq. 4–7).
- C2 Frequency-staged training with archiving + reconstructed-image supervision (Sec. 3.3,
  Alg. 1, Tab. 6 loss ablation).
- C3 Stage initialisation by xyz-only inheritance with ratio ρ (Sec. 3.3 "Initialization and
  inheritance", supp. B).
- (C4 plug-and-play on Taming/FastGS — a property, not a mechanism.)

## 2. Equations (verbatim LaTeX)
Eq. (4) pyramid / residual:
`I^{\ell+1}=\mathrm{Down}(I^{\ell}), \qquad R^{\ell}=I^{\ell}-\mathrm{Up}(I^{\ell+1})`,
ℓ∈{0..L-2}; level 0 = full resolution.
Eq. (5) base colour mapping: `c^{\mathrm{rgb}}_i(\mathbf{d})=\mathrm{clamp}\!\big(c_i(\mathbf{d})+0.5,\,0,\,1\big)`;
residual fields remove shift and clamp (signed colours).
Eq. (6) `\hat I^{L-1}(v)=\mathrm{Render}_{\mathrm{base}}(\mathcal{G}^{L-1};v)`, `\hat R^{\ell}(v)=\mathrm{Render}_{\mathrm{res}}(\mathcal{G}^{\ell};v)`.
Eq. (7) composition: `\hat I^{\ell}_v = \mathrm{Up}(\hat I^{\ell+1}_v) + \hat R^{\ell}_v, \ \ell=L-2,\ldots,0`.
Stage loss (Alg. 1 l.19 / Sec. 4.4): `\mathcal{L}_{\text{rec}}^\ell = \mathcal{L}_{\mathrm{photo}}(\hat I_v^\ell, I_v^\ell)`.
Ablation alternatives (not default): online residual `L_photo(R̂, I^l - Up(Î^{l+1}))`, GT
Laplacian `L_photo(R̂, R^l)`.

Operators (code `laplacian/laplacian_torch.py`):
- kernel k = outer([1,4,6,4,1],[1,4,6,4,1])/256.
- Down(x) = conv2d(reflect_pad(x,2), k, stride 2) (per channel) → H' = ⌊(H−1)/2⌋+1.
- Up(x, size) = bilinear interpolate to size (align_corners=False), then, with `--blur_up`
  (set in both run scripts), conv2d(reflect_pad(·,2), k, stride 1).
- Rendering at level l: same camera FoV, `render_size` = pyramid level size (focal scaled).
- Base render clamped to [0,1]; residual render clamped to [−1,1]
  (`gaussian_renderer/__init__.py:129`); background black (`white_background=False`).
- Residual rasterizer `_pn`: `forward.cu` drops `result += 0.5` and the `max(·,0)` clamp.

## 3. Hyper-parameters
| name | value | source |
|---|---|---|
| levels L | 2 | paper §4.1; script `LAP_LEVELS=2` |
| iteration split | 10000 / 20000 (base / residual) | paper §4.1; `--iter_plan 10000_20000` |
| inheritance | xyz only, ρ = 1.0, equidistant subset if ρ<1 | supp. A; `--inherit_state only_xyz --inherit_ratio 1.0`, `_equidistant_indices` |
| new-field init | knn scales (distCUDA2, 3-NN), quat identity, opacity 0.1, SH all 0 | `create_from_xyz` |
| densify_until per stage | 15000·stage_iters/30000 (5000 / 10000) | `train_lap.py:446` |
| position LR | 1.6e-4·extent → 1.6e-6·extent over each stage's iterations (restart per stage) | `train_lap.py:447`, `get_expon_lr_func` |
| SH degree | +1 every 1000 stage-local iterations (new field starts at 0) | `train_lap.py:584` |
| optimizer | fresh Adam per stage (Taming: SparseGaussianAdam) | `training_setup` |
| lr f_dc / f_rest / opacity / scaling / rotation | 0.0025 / 0.005/20 / 0.025 / 0.005 / 0.001 | `arguments/__init__.py` (`lowfeature_lr`, `highfeature_lr`/20) |
| percent_dense | 0.01 (Taming script) | `run_mipnerf360_1k_taming.sh` |
| densify grad thr / from / interval | 0.0002 / 500 / 100 | arguments |
| opacity reset | every 3000 stage-local its, while < densify_until; min(σ,0.01) | `train_lap.py:775` |
| prune | opacity < 0.005 only, prune-first then top-k clone/split (DashGaussian ADC, rate 1.0 = unbounded); no size prune | `prune_and_densify_dash` |
| λ_dssim | 0.2 (fused_ssim, same padding) | `train_lap.py:707` |
| Taming backbone | `resolution_mode=const`, `densify_mode=free` (DashGaussian schedulers disabled) | arguments defaults |
| FastGS backbone | per-scene loss_thresh / grad_abs_thresh / highfeature_lr, VCD/VCP | `run_mipnerf360_1k_fastgs.sh` |

## 4. Schedules
Per stage s with T_s iterations (local iteration i = 1..T_s): LR log-linear 1.6e-4→1.6e-6 over
T_s; SH degree min(⌊i/1000⌋,3); densify for 500 < i < 0.5·T_s every 100; opacity reset at
i % 3000 == 0 while i < 0.5·T_s. Stage boundary: archive previous field, cache Up(Î^{l+1}) per
training view (one render each), create the next field. Last iteration of each stage has no
optimizer step (code bug, negligible).

## 5. Builds on
Taming-3DGS (efficient backward + Sparse Adam, 3DGS densification via DashGaussian ADC with
schedulers off) or FastGS ("Big"). Everything else = 3DGS (llffhold 8, SfM init, L1+D-SSIM).

## 6. Protocol
- Mip-NeRF 360 "1K": `-i images`, default 3DGS loader → every image wider than 1600 px is
  resized to width 1600 (all 9 scenes; our `mipnerf360-1600` dataset). 2K/4K settings also.
- Deep Blending, Tanks&Temples (standard 3DGS data, already ≤1600 px). `--eval` llffhold 8.
- 30K iterations; L40 GPU (1K). LPIPS from `lpipsPyTorch` (VGG) in `metrics.py`.
- Per-scene tables: supp. Tab. 9 (m360 1K), 10 (DB 1K), 11 (T&T 1K), 12/13 (m360 2K/4K).

## 7. Ambiguities
- Paper says "Taming-3DGS"; code backbone `taming` uses the DashGaussian rasterizer/ADC with
  both schedulers disabled, i.e. 3DGS-equivalent densification (prune opacity only).
- f_rest LR 0.005/20 = 2.5e-4 (FastGS-style `highfeature_lr`), not 3DGS's 1.25e-4.
- Archived field is moved off GPU in the paper (speed/memory claim) — no effect on quality.
- Background black in code; the stage cache is rendered with the same background.
- No result of Laplacian-GS on vanilla 3DGS is reported; nearest backbone to our gsplat
  baseline is Ours(Taming).
