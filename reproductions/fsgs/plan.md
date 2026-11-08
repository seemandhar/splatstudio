# FSGS native plan

Method `fsgs`, module `ss_fsgs`, package `splatstudio-method-fsgs` in `plugin/`.
Benchmark: LLFF 3-view at 1/8 res (paper Table 1: 20.31 / 0.652 / 0.288), all 8 scenes (averages
only), dataset `llff-fewview3-r8-fsgs` (= `llff-fewview3-r8` with the official `images_8` PNGs,
built by `tools/prepare_llff_fsgs.py`), 10 000 steps, `--recipe 3dgs`, all init points.

## Components
| name | kind | requires → provides | config (paper/code values) |
|---|---|---|---|
| `fsgs-renderer` | renderer | state.gaussian3d, camera.pinhole → render.rgb, render.depth.accumulated_z, render.camera, gradient.screen.signed | black bg, no clamp, SH +1 every 500 (eval degree 3), straight-through opacity cap 0.999 (Inria backward) |
| `fsgs-density` | density | gradient.screen.signed | grad 5e-4, prune 0.005, interval 100, from >500, until 10000, percent_dense 0.01, dist_thres 10, prox_thres 5, unpool_until 2000, K=3, reset 0.05 at 2001/5001/8001; owns the 10 000 pseudo poses (`set_training_cameras`) |
| `photometric` | loss | render.rgb | built-in 0.8 L1 + 0.2 D-SSIM |
| `fsgs-depth` | loss | render.depth.accumulated_z, render.camera | weight 0.05 → 0.001 after 9501; min of two Pearson variants vs DPT(gt) (cached per training view, no grad) |
| `fsgs-pseudo` | loss | render.camera, render.depth.accumulated_z | weight 0.5, ramp 500, 2000 < step < 9500, interval 1; renders a pseudo view, DPT with gradient |

DPT: MiDaS `DPTDepthModel(backbone="vitb_rn50_384", non_negative=True)` from `~/models/dpt/midas_repo`
+ `dpt_hybrid_384.pt`, timm 0.6.12 from `~/models/dpt/pylib` (sys.path, lazily on first loss
call; refuse to run if another timm is already imported). Weights frozen, eval mode, fp32.

No new FieldSpecs. Unpooling appends Gaussians through gsplat's `_update_param_with_optimizer`
(zero Adam moments, `requires_grad` copied from the old Parameter — the ResGS pitfall).
KNN: exact torch cdist (simple-knn is exact 3-NN excluding self), only for Gaussians with
max scale > extent (both FSGS rules require it), so no custom CUDA is needed.

## Probes
| probe | checks (fails if mechanism removed) |
|---|---|
| `proximity-unpooling` | fixture with isolated large Gaussians: new Gaussians = midpoints to the 3 NN, destination scale/opacity, identity quat, zero SH, Adam aligned, all params trainable; independent brute-force reference; none added when step ≥ 2000 or criteria fail |
| `fsgs-densify` | split rule `dist > 10·extent & scale > extent` fires without gradient; opacity-reset schedule 2001/5001/8001; prune 0.005 |
| `pearson-depth` | train-view loss equals independent numpy corrcoef formula (min of two), weight schedule 0.05/0.001, gradient reaches means |
| `pseudo-views` (CUDA) | poses: rotations, focus distance, centres in the training box; loss gating/ramp; gradient through DPT reaches sh0 (needs weights, else ProbeUnavailable) |
| `sh-schedule` | training renders use min(t//500,3); eval 3 |

## Fidelity gaps
1. View sampling with replacement (official pops a shuffled stack) — runtime; ±visits per view.
2. Optimizer step before densification (official densifies first) — runtime order; small.
3. gsplat vs Inria rasterizer (0.999 vs 0.99 alpha clamp, opacity-aware radii, ST cap emulation).
4. Runtime SSIM (valid conv) and `lpips` LPIPS-VGG vs lpipsPyTorch; evaluation PSNR identical.
5. Init cloud from COLMAP 3.9.1 MVS (authors' version unknown; authors' released clouds not used).
6. Position LR: runtime 0.01^(t/T) equals official exp schedule with delay_mult irrelevant (delay steps 0).
7. DPT_Hybrid from MiDaS master (Aug 2024) code; hub at the authors' time may differ trivially.
8. One seed per scene.
