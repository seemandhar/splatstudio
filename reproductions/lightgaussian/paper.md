# LightGaussian — implementation spec

Paper: Fan, Wang, Wen, Zhu, Xu, Wang. *LightGaussian: Unbounded 3D Gaussian Compression with 15x
Reduction and 200+ FPS*, NeurIPS 2024, arXiv 2311.17245 (LaTeX source in `source/latex/neurips_2024.tex`).
Official code: github.com/VITA-Group/LightGaussian @ 6676b98 (`source/code/`), modified rasterizer
github.com/Kevin-2017/compress-diff-gaussian-rasterization @ 9893d8c (`source/rast/`).
License: Inria Gaussian-Splatting license (non-commercial research use) for both.

## 1. Method summary
- Post-processing pipeline on a trained 3D-GS model (30k iterations, unchanged 3D-GS).
- **C1 Global significance + prune + co-adaptation (§3.2)**: score each Gaussian by summing
  σ_j·T over all training pixels it hits, times a normalised-volume factor γ(Σ)=V_norm^β; prune the
  lowest 66 %, fine-tune 5000 iterations without densification.
- **C2 SH distillation with pseudo views (§3.3)**: truncate SH degree 3 → 2; distil 5000 iterations
  from the (pruned, SH-3) teacher to the SH-2 student on training views and on pseudo views whose
  camera centre is jittered by Gaussian noise.
- **C3 Significance-weighted VQ of SH (§3.4)**: the 60 % least significant Gaussians get their
  (DC + rest) SH replaced by a code of an 8192-entry codebook learned with significance-weighted
  EMA; all other attributes stored in float16 (xyz float32 in the code); 5000-iteration VQ fine-tune
  of codebook + attributes with fixed assignment (paper; not in the released code).

## 2. Equations (verbatim)
Eq. 3 (significance):
`\text{GS}_{j} = \sum_{i=1}^{MHW} \mathbbm{1}(\text{G}(\Mat{X}_j), r_i) \cdot \sigma_j \cdot \mathbf{T} \cdot \gamma(\boldsymbol{\Sigma}_j)`,
`\mathbf{T} = \prod_{i=1}^{j-1}(1-\sigma_i)`, `V(\Sigma_j)=\frac{4}{3}\pi abc`.
Eq. 4: `\gamma(\Sigma) = (V_{norm})^\beta`, `V_{norm} = \min(V(\Sigma)/V_{max90}, 1)`.
Eq. 5/6 (distillation):
`\mathcal{L}_{distill} = \frac{1}{HW}\sum_i \lVert C_{teacher}(r_i;[R|t]) - C_{student}(r_i;[R|t]) \rVert_2^2`,
`t_{pseudo} = t_{train} + \mathcal{N}(0,\sigma^2)`.
VQ update (§3.4): `c_k = \lambda_d c_k + (1-\lambda_d)\, \frac{1}{T_k} \sum_{g_j\in R(c_k)} GS_j g_j`,
`T_k = \sum_{g_j \in R(c_k)} GS_j`, λ_d = 0.8.

### What the code actually computes (code overrides paper)
- Score kernel (`rast/cuda_rasterizer/forward.cu:renderCUDA_count`): per pixel, front to back,
  skip `power>0` and `alpha<1/255` (alpha = min(0.99, σ·G)); if `T(1-alpha) < 1e-4` the pixel stops
  and that Gaussian is **not** counted; else `score[j] += σ_j · T` (σ_j = sigmoid opacity, T the
  transmittance *before* j, i.e. product of (1-alpha_i), not (1-σ_i)). Summed over every training
  view (`prune.py:prune_list`; the kernel `+=` is non-atomic — we use atomics).
- `prune.py:calculate_v_imp_score`: `volume = prod(exp(scales))`; `kth = sort(volume, desc)[int(0.9 N)]`
  (i.e. the volume that 90 % of Gaussians exceed); `v = (volume/kth)^0.1 * score` — **no clip at 1**,
  no 4/3π (constant, irrelevant for ranking).
- `gaussian_model.py:prune_gaussians`: `thr = sort(v)[int(p (N-1))]`, remove `v <= thr` (p = 0.66).
- Distillation loss (`distill_train.py`): **L1 + 0.2 D-SSIM** between student and teacher renders
  (not L2); pseudo view on iterations with `iteration % 3 != 0` (2/3 of steps), camera centre +
  N(0, 0.05²) per axis in world units (rotation std 0) (`utils/pose_utils.py:gaussian_poses`);
  other steps: same training view, teacher render as target (GT never used in this stage).
- VQ (`vectree/vectree.py`, `vq.py`): features = [f_dc (3), f_rest (8×3)] = 27-d (DC included).
  Non-VQ set = top `int(N·0.4)` by v-score (recomputed on the distilled student, β = 0.1).
  Codebook EuclideanCodebook, `kmeans_init=False` ⇒ **kaiming-uniform init** (bound
  sqrt(6/(8192·27))), not K-means as the paper says. 1000 iterations; each samples 80000 VQ
  features with replacement; weights w = imp·n/Σimp; `cs ← 0.8 cs + 0.2 Σ onehot·w`;
  `embed ← 0.8 embed + 0.2 (Σ w x onehot) / ((cs+1e-5)/(Σcs+K·1e-5)·Σcs)`; then the 10 codes with
  smallest `cs` are overwritten by the 10 highest-weight batch vectors. Assignment: nearest code
  (cdist). Storage: codebook fp16, VQ indices 13-bit packed, non-VQ SH fp16, opacity/scale/rot fp16
  (raw logit/log/unnormalised quat), xyz **fp32**, mask packbits; each `np.savez_compressed`,
  `zip -r`; size = zip bytes / 1024².

## 3. Hyper-parameters
| name | value | source |
|---|---|---|
| prune ratio | 0.66 | `scripts/run_prune_finetune.sh` (paper Tab. 2: 365→123 MB) |
| prune iteration | 30001 (once, decay 1) | `prune_finetune.py` default `--prune_iterations` |
| β (v_pow) | 0.1 | paper §4.1, script |
| V_max90 | `sort(V,desc)[int(0.9N)]`, no clip | `prune.py:68` |
| co-adaptation | 5000 its (30001–35000), no densify, no opacity reset | paper §3.2, `--iteration 35000` |
| prune-stage LR | Adam state restored from the 3DGS ckpt (weight_decay 0 restored from Adam groups); non-xyz LR ×0.95 every 400 its; xyz 3DGS exp schedule with max_steps 35000 | `prune_finetune.py` |
| SH degree | 3 → 2 | paper §4.1, `--new_max_sh 2` |
| distill its | 5000 (35001–40000) | `--iteration 40000` |
| distill optimiser | fresh `AdamW` (default weight_decay 0.01, eps 1e-15), default 3DGS LRs, ×0.90 every 500 its, xyz max_steps 40000 | `distill_train.py` (training_setup after restore), `gaussian_model.py:217` |
| distill trainable | xyz, SH, scaling, rotation (`--enable_covariance`); **opacity frozen** | script, `distill_train.py` |
| pseudo-view σ | 0.05 (code) vs 0.1 (paper §4.1) | `distill_train.py:gaussian_poses(...,0.05,0)` |
| pseudo-view frequency | `iteration % 3 != 0` | `distill_train.py` |
| codebook size | 8192 | paper, script |
| VQ ratio | 0.6 (least significant) | paper, script |
| VQ EMA decay | 0.8 | paper, `vectree.py` |
| VQ iterations / batch | 1000 / 80000 | `vectree.py` |
| dead-code replacement | k = 10 per iteration | `vectree.py:k_expire` |
| VQ fine-tune | 5000 its, fixed mapping, photometric loss, no densify | paper §3.4/supp. (no released code) |
| 3DGS stage | standard 3DGS 30k | paper (3D-GS* retrained with original code) |

## 4. Schedules (global iteration numbering)
1–30000 3D-GS; 30001 prune 66 %; 30001–35000 co-adaptation; 35001–40000 SH distillation;
after 40000 VQ (codebook learning off-line); 40001–45000 VQ fine-tune; save compressed.

## 5. Builds on
Vanilla 3D-GS (Inria) — densification, loss, LRs unchanged in the first 30k iterations. Black
background in the official code (dataset convention; our runtime uses white).

## 6. Protocol
Mip-NeRF 360 (9 scenes, images_4 outdoor / images_2 indoor as 3D-GS), Tanks&Temples truck/train,
llffhold 8 (`--eval`), PSNR/SSIM/LPIPS (LPIPS-VGG via `lpipsPyTorch`, 3D-GS scripts). Evaluated
model = final compressed (VQ fine-tuned) model. Per-scene tables in the appendix (Tabs. 6–8) report
"Ours" = final model: room 31.399 equals Tab. 2 #8/#9 (31.40, after VQ fine-tune).

## 7. Ambiguities / discrepancies
1. Per-scene Tab. 6 "Ours" mean over 9 scenes = 27.257 dB, but Tab. 1 says 27.13; the 3D-GS* row
   mean (27.397) matches Tab. 1 (27.40). Rows Plenoxels/INGP/mip-NeRF360 have room/treehill
   swapped; the 3D-GS and Ours rows are consistent with the header (room 31.40 = Tab. 2).
2. Teacher for distillation: paper Alg. 1 uses G' (pruned + recovered); the script's placeholder
   is `TEACHER_CHECKPOINT/chkpnt30000.pth`. We follow Alg. 1 (the 35k pruned model), which also
   keeps the 40k iteration count used by the scripts (VQ reads `iteration_40000`).
3. VQ fine-tune is not in the released code (vectree writes the dequantised ply, nothing trains
   it); optimizer, LRs and schedule are unspecified. See plan.md for the choice.
4. Paper: K-means codebook init, L2 distill loss, σ = 0.1, float16 positions; code: kaiming init,
   L1+D-SSIM, σ = 0.05, fp32 positions. Code wins.
5. Paper Tab. 1 "Size" = zip MB of the `extreme_saving` folder (MiB).
