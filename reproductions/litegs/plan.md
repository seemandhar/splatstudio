# LiteGS native plan

Method `litegs` (module `ss_litegs`) = **LiteGS-quality** (per-scene target = 3DGS count, 30K steps).
Benchmark: Mip-NeRF 360 garden (outdoor, paper 28.15) + room (indoor, paper 32.69), Appendix Table 5.

## Components
| component | kind | factory | requires / provides | config (paper/code value) |
|---|---|---|---|---|
| litegs-renderer | renderer | ss_litegs.renderer:create_renderer | provides render.rgb, render.depth.expected_z, gradient.screen.signed, litegs.raster_meta | sh_epoch_interval 5, clamp_training_rgb true |
| litegs-density | density | ss_litegs.density:create_density | requires litegs.raster_meta | total_steps 30000, densify_from 3, densification_interval 5, opacity_decay_interval 10, decay_rate 0.5, decay_floor 1/128, until_fraction 0.8, percent_dense 0.01, split_scale_div 1.6, target_primitives (lookup by SfM count, default 1e6), lr {opacities .025, shN .00025, sh0 .0025, scales .005, quats .001}, adam {no bias correction, visible-only, clear on decay}, score_opacity_power 2 |
| photometric | loss | splatstudio.losses:create_photometric | render.rgb | ssim_weight 0.2 |

No extra fields.

## Mapping
- C1 metric: CUDA kernel `csrc/litegs_stats.cu` re-traverses gsplat's tile lists (gsplat numerics:
  alpha=min(.999,oG), skip alpha<1/255, stop at T<=1e-4) with the detached final pixel colour
  C (incl. white background) and dL/dC (retain_grad on pre-clamp rgb). Per fragment:
  dL/dalpha = v·(T c_i − (C − P_i)/(1−alpha)), P_i = prefix Σ_{j≤i} T_j alpha_j c_j;
  g = G·dL/dalpha (= dL/do). Accumulate S+=g², M+=g, C+=1, W+=T·alpha. Score = clamp(S/(C+1) −
  (M/(C+1))²,0)·C·σ(o)²  (statistic_helper.get_var + get_score).
- C2 schedule: epochs from steps via n_train (set_training_cameras): epoch e ends at step
  (e+1)·n_train. total_epoch=int(total_steps/n_train), until=int(total_epoch·0.8/10)·10+1.
  Stats on epochs e%5==0, 3≤e<until; at their end budgeted split/clone (multinomial, no replacement)
  → weight prune (W==0) → decay if e%10==0 (o←logit(max(σ(o)/2,1/128)), clear all Adam states).
- Training recipe: `LiteAdam` (torch.optim.Adam subclass) replaces every runtime optimizer in
  density.initialize: m,v,param updated only on rows visible in the current view (radii>0), no bias
  correction, eps 1e-15; lr overrides. SH degree min(epoch//5,3) via a plugin-level schedule shared by
  density (writes step/n_train) and renderer (reads it while grad is enabled; eval uses the runtime
  degree 3). Training rgb clamped to [0,1] before the loss.
- Population ops via gsplat.strategy.ops._update_param_with_optimizer (append with zero moments) and
  ops.remove; requires_grad preserved; trainable-leaf check after every population change.

## Probes
| probe | checks (fails if mechanism removed) |
|---|---|
| litegs-stats-kernel (cuda) | S, M, C, W equal brute force per-pixel autograd of v·I(x,y) w.r.t. opacity on a tiny scene; M equals the summed opacity grad |
| litegs-variance-score | score equals an independent formula; a primitive with conflicting per-pixel gradients outranks one with the same |M| but consistent gradients |
| litegs-densify | budget arithmetic, multinomial only over score>0, split keeps parent, child scale/1.6 and offset ~ R·N(0,s), clone copies, zero moments, weight prune only of old W==0, fields stay trainable |
| litegs-decay-schedule | decay halves σ(o) with 1/128 floor and clears every optimizer state; densify/decay epochs and until for n_train=161 (garden) |
| litegs-adam | LiteAdam = independent bias-free Adam on visible rows; invisible rows and their moments unchanged |
| litegs-sh-schedule | degree = min(((step−1)//n_train)//5, 3) in training, runtime degree at eval |

## Fidelity gaps
1. Warp fp16 rasteriser / cluster-cull-compact not ported (speed only; Table 2 shows ≤0.05 dB).
2. Sparse Adam visibility per primitive (gsplat radii>0) instead of per 128-primitive Morton cluster.
3. Exact per-pixel squared gradients instead of the official kernel's per-thread running-sum quirk.
4. Runtime: views sampled with replacement (LiteGS: shuffled epochs); white background (LiteGS
   black); SSIM valid conv vs fused_ssim same conv; alpha thresholds 1/255 & 0.999 (LiteGS 1/256,
   255/256); gsplat zeroes backprop where oG>0.999.
5. Steps: 30000 vs LiteGS total_epoch·n_train (29946 on garden) — negligible.
6. xyz lr decays over runtime steps (same 30000).
7. Appends not truncated to multiples of 128 (clustering artefact).
