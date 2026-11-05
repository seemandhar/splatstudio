# PUP 3D-GS — native SplatStudio plan

Method names: **`pup-3dgs`** (v3/CVPR: 80% then 50% = 90%, 40k steps) and **`pup-3dgs-v1`**
(arXiv v1: 66% then 66% = 88.44%, 40k steps). Module `ss_pup3dgs`.

## Single-run design (chosen)
The runtime trains one schedule, so one `ss-train --steps 40000` run expresses the paper's
whole pipeline inside the density component:

| steps | phase | what the density does |
|---|---|---|
| 1–30000 | pretrain = standard 3DGS | delegates to `splatstudio.backends.gsplat.Density` (DefaultStrategy + opacity-reset fix, identical config to `gsplat`); overrides the runtime's means LR (which decays over `--steps`=40000) with `lr0·0.01^(s/30000)`, i.e. exactly what `ss-train gsplat --steps 30000` uses → first 30k steps follow the shared `baseline3dgs-*-30000` recipe (same seed, same view sequence) |
| end of 30000 | prune round 1 | Fisher over all training cameras (official CUDA kernel, 1/4-res), score = log|F|, prune lowest 80% (official `<=` percentile rule), fresh Adam (state cleared) |
| 30001–35000 | refine 1 | official fine-tune LRs: means `lr0·0.01^(t/35000)` with t=30000+k; other fields base_lr·0.95^(⌊t/400⌋−75); means also ×0.95 when t%400==0 |
| end of 35000 | prune round 2 | same with 50% |
| 35001–40000 | refine 2 | same schedule restarted (official `--first_iter 30000`) |

Densification is inactive after 15k (gsplat `refine_stop_iter`), so refinement has none, as in
the paper. Starting from the shared baseline checkpoint is not needed: the single run is
faithful and the pre-prune reference is the same-recipe `baseline3dgs-*-30000` run.

## Components
| component | kind | factory | requires/provides | config (paper values) |
|---|---|---|---|---|
| gsplat | renderer | `splatstudio.backends.gsplat:create_renderer` (unchanged) | — | baseline |
| pup-density | density | `ss_pup3dgs.density:create_density` | requires `gradient.screen.signed` | `base` (baseline DefaultStrategy config), `base_steps` 30000, `prune_steps` [30000, 35000], `prune_ratios` [0.8, 0.5], `fisher_resolution` 4, `finetune_lr_start` 30000, `finetune_lr_max_steps` 35000, `lr_final_ratio` 0.01, `exp_gamma` 0.95, `exp_every` 400, `reset_optimizer` true, `background` [1,1,1] |
| photometric | loss | `splatstudio.losses:create_photometric` (unchanged) | render.rgb | baseline |

No new fields. Training cameras via `set_training_cameras`.

## CUDA
Vendor the official `rasterization_and_pup_fisher` sources (Inria 3DGS rasterizer + PUP
`renderFisherCUDA`, FisherRF-derived; licence copied) + glm headers into
`ss_pup3dgs/csrc/`, JIT-built with `torch.utils.cpp_extension.load` (via `tools/ssenv`).
Called directly: `_C.rasterize_gaussians` (forward at ceil(H/4)×ceil(W/4)) then
`_C.rasterize_gaussians_backward` with dL/dpixel = 1, which returns the [P,21] symmetric
per-view Fisher. Camera conversion: viewmatrix = viewmat^T; projmatrix built from the scaled
K with off-centre principal point (P00=2fx/W, P02=2cx/W−1, …) so 3DGS's `ndc2Pix` maps to
gsplat's pixel convention exactly; tanfov = W/(2fx); campos = inv(viewmat)[:3,3].
Inputs: means, exp(scales), normalized quats (wxyz, same as 3DGS), sigmoid(opacities)[:,None],
SH [N,16,3], degree 3.

## Probes
| probe | contribution | measures / fails when |
|---|---|---|
| `fisher-oracle` (CUDA) | C1 + C2 | kernel F vs independent autograd oracle: per (pixel, channel) gradients of a gsplat render of the 1/4-res camera w.r.t. (means, log-scales), outer products summed. Fails if scale activation factor, channels, patch resolution or camera conversion are wrong; also asserts that a "linear-scale" Fisher (no ×s) differs, so the probe discriminates |
| `prune-schedule` (CUDA) | C3 | runs the density on a small scene through two prune rounds: pruned set = lowest log|F| (independent `slogdet` in float64), counts per official percentile rule, Adam state cleared, LRs at selected steps equal an independent re-implementation of `get_expon_lr_func` + ExponentialLR, pre-prune means LR = 30k schedule |

Unit tests (CPU): LR schedule functions, percentile prune mask incl. ties, 21→6×6 expansion,
camera conversion (projected pixel == K·x/z − 0.5 under 3DGS ndc2Pix).

## Files
`plugin/pyproject.toml`, `ss_pup3dgs/{__init__,density,fisher,schedule,probes}.py`,
`ss_pup3dgs/csrc/{cuda_rasterizer/*,rasterize_points.cu,rasterize_points.h,ext.cpp,LICENSE-3DGS.md,third_party/glm}`,
`tests/test_pup3dgs.py`, `README.md`.

## Fidelity gaps
1. **Base 3DGS ≠ authors' models**: our pretrain is the gsplat runtime (white bg, random
   views with replacement, gsplat DefaultStrategy + reset fix, valid-SSIM). Authors' 3DGS
   row is higher on our chosen scenes (bicycle 25.09 vs our 24.92, garden 27.28 vs 26.89,
   counter 29.11 vs 28.85). Compare the pruning loss (Δ vs pre-prune) and absolute PSNR.
   Not closable without the authors' checkpoints.
2. **Fisher rasterizer**: official 3DGS-rasterizer kernel on a gsplat-trained model (alpha
   clamp 0.99 vs gsplat 0.999; otherwise same EWA projection, 0.3 dilation, 3σ tiles).
   Ranking effect expected negligible. Background white (our training bg) vs black officially.
3. **Fine-tune steps**: 5000 effective updates per round vs official 4999 (the prune
   iteration's gradient is discarded there); view sampling with replacement.
4. **Pre-prune model not evaluated in the same run** (runtime evaluates only at the end);
   the same-recipe `baseline3dgs-*-30000` runs are the pre-prune reference (statistically,
   not bitwise, identical first 30k steps).
5. FPS / size not benchmarked (paper reports A4000 FPS).
