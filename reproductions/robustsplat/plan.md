# RobustSplat — native SplatStudio plan

Method `robustsplat` (module `ss_robustsplat`); ablation variants `robustsplat-nodg`
(3DGS + Mask + MB: 3DGS densification/reset schedule) and `robustsplat-nomb`
(3DGS + Mask + DG: fine supervision from step 500) — Table 3 rows, not benchmarked by default.

## Components
| component | kind | factory | requires → provides | config (paper/code values) |
|---|---|---|---|---|
| `rs-renderer` | renderer | `ss_robustsplat.renderer:create_renderer` | state.gaussian3d, camera.pinhole → render.rgb, gradient.screen.signed | background black (code), near 0.01 |
| `rs-density` | density | `ss_robustsplat.density:create_density` | gradient.screen.signed | refine_start 10000, refine_stop 20000, refine_every 100, grow_grad2d 2e-4, percent_dense 0.01, prune_opa 0.005, reset_start 15000, reset_every 3000, reset_value 0.01, opacity_lr 0.025 |
| `rs-masked-photometric` | loss | `ss_robustsplat.losses:create_photometric` | render.rgb | lambda_dssim 0.2, mask_start 500, mask_threshold 0.25, erode_kernel 7 |
| `rs-mask-mlp` | loss | `ss_robustsplat.losses:create_mask_mlp` | render.rgb | lambda_residual 0.5, lambda_cos 0.5, lambda_reg 2.0, beta_reg 2000, bootstrap_iter 20000, fine_feat 36, coarse_feat 16, lower 0.6, upper 0.8, bins 10000, decay 0.95, pause 300, reset_start 15000, reset_every 3000, mlp_lr 1e-3, hidden 16 |

No new per-Gaussian fields. Run-wide state (MLP, its Adam, histogram, DINOv2 backbone, GT
feature tables, coarse GT table) lives in `ss_robustsplat.context.Shared` (one per process),
looked up by the training camera's viewmat (runtime exposes no frame name).

## Mapping (equations → code)
- **C1 Delayed Gaussian growth** → `rs-density`: Inria `densify_and_prune` reproduced with
  `gsplat.strategy.ops` (clone if max scale ≤ 0.01·extent, split otherwise, /1.6 scale, prune
  σ < 0.005, no screen/world-size pruning), for 10000 < step < 20000, step % 100 == 0; grad stats
  (‖∂L/∂μ2D‖ in pixels, visibility radii > 0) accumulated from step 1; opacity reset
  `min(σ, 0.01)` for step ≥ 15000, step % 3000 == 0, step < 20000. Sets opacity LR 0.025
  (Inria 2024 default; runtime default 0.05) in `initialize`.
- **C2 Scale-cascaded mask bootstrapping** → `rs-mask-mlp`: for step < 20000 renders the same
  camera at the coarse (images_{4f}) size (no grad), residual masks from coarse GT vs coarse
  render, GT/render DINOv2 features at 16×16 (224² input); for step ≥ 20000 residual from the
  training render, features 36×36 (504²). Masks/cosine maps bilinearly upsampled to training res.
- **C3 Mask MLP + supervision** (Eq. 3–6, supp. S1): `rs-masked-photometric` evaluates the MLP on
  upsampled fine features, stores the graph-attached mask in `outputs["robustsplat"]`, and
  returns `0.8·mean(m·|I−I*|) + 0.2·mean(1 − SSIM_map(m·I, m·I*))` with
  `m = erode7(M.detach() > 0.25)`; before step 500 plain L1 + D-SSIM (Inria same-padding SSIM).
  `rs-mask-mlp` returns `0.5·mean|M − M_cos| + 0.5·hinge(M, lower, upper) + L_reg`; the MLP
  optimiser steps at the start of the next call (identical order to official: MLP_{i} is used for
  the mask at step i+1). Paused (no loss, no step) for i < 500 and for
  reset_start < i < reset_start + 300 when i ≥ 15000.
- **C4 Early mask regularisation** (Eq. 7) → `rs-mask-mlp` term `2.0·mean((1 − M)·e^{−i/2000})`.
- DINOv2 ViT-S/14-reg (`dinov2_vits14_reg4_pretrain.pth`, sha256-checked) re-implemented in
  `dino.py` (architecture copied from WildGaussians port; `forward_features` = 12 blocks + norm,
  patch tokens), raw [0,1] input bilinearly resized to 14s×14s.

## Probes
| probe | measures | fails if |
|---|---|---|
| `delayed-growth` | refine/reset step sets of `rs-density` on a synthetic scene with forced large grads | growth before 10100 / after 19900, reset outside {15000,18000}, size pruning present |
| `bootstrap-scale` | feature-grid size, residual source resolution and cosine values at step 19999 vs 20000 | no switch, wrong grids, coarse residual not from the coarse render |
| `mask-supervision` | robust mask/histogram vs NumPy reference, cosine map Eq. 4, hinge, reg weight e^{-i/β} | any formula differs |
| `masked-photometric` | erosion+threshold mask vs NumPy, masked-L1/SSIM value vs independent computation, transient blob gets zero gradient | mask unused / not eroded |
| `mlp-schedule` | MLP is not stepped before 500 and in pause windows, is stepped otherwise | pause or start removed |
| `dino-parity` (CUDA) | our ViT vs timm `vit_small_patch14_reg4_dinov2` with the same weights | architecture/weights wrong |
| `pipeline-step` (CUDA) | renderer → both losses → density for a few steps on a synthetic scene with features | anything nonfinite / not wired |

## Files
`plugin/pyproject.toml`, `plugin/ss_robustsplat/{__init__,renderer,density,losses,masks,dino,context,probes}.py`,
`plugin/tests/test_robustsplat.py`, `plugin/README.md`; `tools/prepare_robustsplat.py` (data).

## Data / protocol
Authors' released data (hf.co/datasets/fcy99/RobustSplat-data), converted by
`tools/prepare_robustsplat.py` to `~/datasets/robustsplat/native/<scene>` (→ /bigdata). Datasets
registered as `robustsplat-onthego` (patio ×4, patio-high ×8) and `robustsplat-robustnerf`
(android, crab2 ×8). Benchmark: patio-high (On-the-go, paper 22.87, 3DGS 17.04) and crab2
(RobustNeRF, 34.88, 3DGS 31.76), 30k steps, `--recipe 3dgs`; plus `gsplat` on the same scenes
as protocol baseline.

## Fidelity gaps
1. Optimiser step before densification (runtime order) — official densifies before the step;
   negligible.
2. View sampling with replacement (runtime) vs Inria's shuffled stack without replacement.
3. SH degree increments at step 1001 (runtime `(step−1)//1000`) vs 1000; negligible.
4. Base photometric replaced, so runtime SSIM (valid conv) is not used; plugin uses the Inria
   same-padding SSIM, matching the official code. Evaluation SSIM/LPIPS: runtime (valid SSIM,
   `lpips` VGG) — compare PSNR primarily.
5. Means LR schedule: runtime pure exponential 1.6e-4→1.6e-6 (Inria identical with delay_mult
   0.01 only affecting delay_steps=0 → none). Matches.
6. Initial quats identity (Inria identity) — matches; init scales k-NN (recipe 3dgs) — matches.
7. Single seed; paper single run.
8. DINOv2 attention without xFormers (numerically equivalent).
9. Validation = copies of test (protocol requirement of runtime), never used for selection.
