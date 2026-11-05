# PUP 3D-GS: Principled Uncertainty Pruning for 3D Gaussian Splatting — implementation spec

- arXiv: 2406.10219 (v3, CVPR 2025; v1 June 2024 reported an 88.44% = 66%+66% variant)
- Official code: https://github.com/j-alex-hanson/gaussian-splatting-pup (cloned to `source/code/`),
  Fisher CUDA kernel: submodule `rasterization_and_pup_fisher` (commit eaef2fa, Inria 3DGS
  licence: non-commercial research/evaluation use; Fisher code derived from FisherRF).
- Sources: LaTeX `source/latex/sections/{3_method,4_experiments,X_suppl}.tex`, v1 PDF text
  `source/v1/paper_v1.txt`.

## 1. Method summary
- Post-hoc pruning of a **pretrained, converged 3D-GS model** (30k iterations, standard 3DGS).
- Per-Gaussian sensitivity score = log-determinant of the 6×6 block of the Fisher
  approximation of the L2-reconstruction Hessian w.r.t. that Gaussian's mean (3) and
  log-scale (3), summed over all training views, computed on 4×4 "patches" (= rendering at 1/4
  resolution with unchanged FoV).
- Prune the lowest-scoring fraction, fine-tune ("refine") 5k iterations without densification,
  repeat: 80% then 50% (90% total) in v3; 66%+66% (88.44%) in v1.

Contributions
- **C1 Fisher sensitivity score** (§3.1–3.2, Eqs. fisher / U_i): U_i = log|∇_{x_i,s_i} I ∇_{x_i,s_i} I^T|.
- **C2 Patch-wise computation** (§3.3): 4×4 patches = Fisher over pixels of the 1/4-res render.
- **C3 Multi-round prune-refine** (§3.4): two rounds (80%, 50%), 5k fine-tune each, no densification.

## 2. Equations (verbatim from `3_method.tex`)
- L2 objective: $L_2 = \frac{1}{2}\sum_{\phi \in \Set{P}_{gt}}||I_{\Set{G}}(\phi) - I_{gt}||_2^2.$
- Hessian: $\nabla_{\Set{G}}^2 L_2 = \sum_{\phi}\nabla_{\Set{G}} I_{\Set{G}}(\phi) \nabla_{\Set{G}} I_{\Set{G}}(\phi)^T + (I_{\Set{G}}(\phi) - I_{gt})\nabla_{\Set{G}}^2 I_{\Set{G}}(\phi)$ (eq:l2_d2)
- Fisher (residual→0): $\nabla_{\Set{G}}^2 L_2 = \sum_{\phi}\nabla_{\Set{G}} I_{\Set{G}}(\phi) \nabla_{\Set{G}} I_{\Set{G}}(\phi)^T$
- Block: $\Mat{H}_i = \nabla_{\Set{G}_i}I_{\Set{G}} \nabla_{\Set{G}_i}I_{\Set{G}}^T$
- Score (eq:uncertainty): $U_i = \log |\nabla_{x_i,s_i}I_{\Set{G}} \nabla_{x_i,s_i} I_{\Set{G}}^T|$

Definitions as implemented in code (`fisher_pool_xyz_scaling.py:pool_fisher_cuda`,
`backward.cu:renderFisherCUDA`):
- The "Jacobian" rows are per **(pixel, colour channel)** of the rendered image:
  g_{p,c,i} = ∂ I(p)_c / ∂(x_i, log s_i) ∈ R^6 (the CUDA kernel multiplies dI/ds by s —
  "Our scaling INCLUDES the scaling activation"; i.e. derivative w.r.t. the log-scale parameter
  `_scaling`). F_i = Σ_views Σ_p Σ_c g g^T (21 unique entries, float32 atomics).
- Gradients include the alpha-compositing path (mean2D, conic → mean3D, cov3D → scale), the
  background term (−T_final/(1−α)·bg) and the SH colour-direction path to the mean
  (`preprocessCUDA`). Kernel quirk: `cur_dL_dcolors` is not reset between channels, so for
  channel c the SH→mean path also includes the colour gradient of channels < c.
- Score: `torch.log(torch.linalg.svdvals(F)).sum(1)` (= log|det F|; F=0 ⇒ −inf, pruned first).
- Pruning rule (`gaussian_model.py:prune_gaussians`): sort scores, k = int(p·(N−1)),
  threshold = sorted[k], prune `score <= threshold` (prunes k+1 Gaussians plus ties).

## 3. Hyper-parameters
| name | value | source |
|---|---|---|
| base model | 3D-GS, 30k iters, official defaults | paper §5.2; code `full_eval.py` (images_4 outdoor / images_2 indoor) |
| prune ratios | 0.80 then 0.50 (90%) | paper §5.3.2, `scripts/full_pruning_pipeline.sh` round1/round2 |
| v1 ratios | 0.66 then 0.66 (88.44%) | v1 §5.3.1 |
| fine-tune per round | 5000 iterations (prune at iteration 30001; iterations 30001..35000) | `prune_finetune.py` defaults + script `--iterations 35000` |
| fisher resolution (patch) | 4 (render at ceil(H/4)×ceil(W/4), same FoV) | script `--fisher_resolution 4`; paper §3.3 |
| fisher params | means (3) + log-scales (3) → 6×6 | paper Eq. U_i; kernel |
| fine-tune xyz lr | 3DGS exp schedule 1.6e-4→1.6e-6 (×cameras_extent), max_steps 35000, evaluated at lr_iter 30001..35000 (both rounds; round 2 uses `--first_iter 30000`) | script `--position_lr_max_steps 35000` |
| fine-tune other lrs | f_dc 2.5e-3, f_rest 1.25e-4, opacity 0.05, scaling 5e-3, rotation 1e-3, × ExponentialLR(γ=0.95) stepped whenever lr_iter % 400 == 0 (also hits xyz for that iteration only, overridden next iteration) | `prune_finetune.py` L74, L117-120; `arguments/__init__.py` |
| optimizer per round | fresh Adam (model reloaded from .ply) | `prune_finetune.py` load_ply + training_setup |
| densification in fine-tune | none | paper §3.4 |
| SH degree in fine-tune | 3 (load_ply sets max) | `gaussian_model.py:load_ply` |
| loss in fine-tune | (1−0.2)·L1 + 0.2·(1−SSIM) | `prune_finetune.py` |
| background | black (`white_background=False`) | `arguments/__init__.py` |

## 4. Schedules
- 0–30k: standard 3D-GS (densify 500–15k every 100, opacity reset every 3k, SH up every 1k).
- Round 1: prune 80% using Fisher of the 30k model at iteration 30001, fine-tune to 35k.
- Round 2: prune 50% of the round-1 model, fine-tune 5k more (total 40k optimisation steps).
- Note: at the prune iteration the gradient of that iteration is discarded (params replaced),
  so each round has 4999 effective updates.

## 5. Builds on
Unmodified 3D-GS training for the base model; LightGaussian's prune-finetune framework
(`prune_finetune.py`, ExponentialLR) for refinement. Rasterizer unchanged for training.

## 6. Protocol
Mip-NeRF 360 (9 scenes, images_4 outdoor / images_2 indoor), T&T (train, truck), DB
(drjohnson, playroom); `--eval` (llffhold 8); 3DGS `metrics.py` (PSNR, SSIM, LPIPS-VGG from
`lpipsPyTorch`); black background. The 3D-GS baseline row is the authors' own pretrained models
(numbers differ from the 3DGS paper: e.g. bicycle 25.09 vs 25.25, room 31.51 vs 30.63).
Seeds not stated. FPS on RTX A4000 (not reproduced).

## 7. Ambiguities
- Which 3DGS code version/resolution produced the pretrained models: `full_eval.py` uses
  `-i images_4/images_2` (not PIL `-r`), so we use our native images_4/images_2 data.
- v3 SSIM/LPIPS per-scene "Baseline" and SSIM "Ours" rows are identical to v1's; PSNR rows
  differ. We treat v3 PSNR (90%) as the target and record v1 (88.44%) too.
- Paper says "prune at 30k then fine-tune 5k"; code prunes at iteration 30001 inside the
  fine-tune loop (equivalent up to one discarded gradient).
- Fisher bg: computed with the training background (black officially).
