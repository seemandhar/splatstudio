# DashGaussian native plan

- **Method name**: `dashgaussian` (3DGS backbone, Adam). Module: `ss_dashgaussian`.
- **Feasible natively**: yes, with two component-side workarounds (below). No custom CUDA: the
  official code's rasterizer changes (`3dgs-accel` backward) are speed-only; gsplat is kept.

## Runtime facts checked (SplatStudio `runtime.py`)
- `target` passed to losses is always `frame["image"]` at full training resolution; the runtime never
  resizes it. `outputs["rgb"]` may have any size (losses are arbitrary functions), but the base
  `photometric` loss would fail on a size mismatch → the photometric loss is **replaced** by a
  resolution-aware L1 + 0.2 D-SSIM that downsamples the target to `outputs["rgb"]`'s size inside the loss
  (antialiased bilinear, as the official code).
- The renderer gets `(params, frame, sh_degree)`, no step. Order per step: render → `before_backward(step)`
  → losses → backward → optimizer step → runtime sets means LR → `after_backward(step)`.
- Evaluation (`measure`, both the end-of-training validation pass and `ss-eval`) runs under
  `torch.no_grad()` with the same renderer interface; `ss-eval` builds fresh components.
- `set_training_cameras` exposes cameras only (no images); the FFT schedule needs all training images
  before step 1.
- Means LR is overwritten by the runtime after every optimizer step, before `after_backward`.

## Workarounds (all inside the plugin)
1. **Renderer ↔ density link**: a module-level link (`ss_dashgaussian.link.LINK`) holds the density
   controller built in the same process. The renderer renders at `floor(H/r), floor(W/r)` with K scaled by
   (w/W, h/H) only when grad is enabled and the linked density has an active schedule; otherwise full
   resolution. Evaluation is therefore always full resolution (no_grad + fresh build in `ss-eval`).
   `before_backward` asserts the rendered size equals the size the schedule prescribes for this step
   (fails loudly on desync).
2. **Training images for the FFT schedule**: `set_training_cameras` locates the dataset from density config
   `data`/`downscale`, or else from the `ss-train` command line (`--data`, `--downscale`), reloads it with
   `splatstudio.data.load_dataset` (identical preprocessing), takes `splits["train"]`, and **verifies** that
   K / viewmat / width / height of every training camera equal the cameras the runtime passed (raises
   otherwise). Raises if no data path is found. Loading happens before the runtime's timer starts.
3. **Position LR**: `after_backward(step)` overwrites the means LR with DashGaussian's schedule for step+1.

Runtime hooks that would make these unnecessary: (a) `set_training_cameras` also passing the training
images (or a `set_training_images(images)` hook); (b) a step/phase argument to `renderer.render`
(or a runtime resolution-schedule hook that resizes `frame` K/width/height and `target` together).

## Components
| component | kind | factory | requires | provides | config (paper/code values) |
|---|---|---|---|---|---|
| dash-renderer | renderer | `ss_dashgaussian.renderer:create_renderer` | state.gaussian3d, camera.pinhole | render.rgb, render.depth.expected_z, gradient.screen.signed, render.resolution_schedule | — |
| dash-density | density | `ss_dashgaussian.density:create_density` | gradient.screen.signed | — | grow_grad2d 2e-4, grow_scale3d 0.01, prune_opa 0.005, refine_start 500, refine_every 100, densify_until 27000, reset_every 3000, reset_value 0.01, total_steps 30000, start_significance_factor 4, max_reso_scale 8, reso_sample_num 32, max_densify_rate 0.2, momentum_init_factor 5, integrate_factor 0.98, momentum_step_cap 1e6, max_n_gaussian -1, lr_final_factor 0.01, resolution_mode "freq", densify_mode "freq", data null, downscale null |
| dash-photometric | loss | `ss_dashgaussian.losses:create_photometric` | render.rgb, render.resolution_schedule | — | ssim_weight 0.2 |

No extra fields. Unchanged from 3DGS: Adam LRs of non-means fields, SH every 1000 steps, clone/split
geometry (gsplat ops), grad normalization (gsplat convention = 3DGS NDC gradient at the render size),
opacity reset value 0.01, 3dgs recipe init.

## Contributions → code → probes
| C | code | probe (fails if removed/wrong) |
|---|---|---|
| C1 resolution scheduler | `schedule.py:build_schedule/res_scale`, renderer, loss | `dash-reso-schedule` (CPU): white-noise analytic r_m≈2 and level scales r_i≈(fraction)^-1/2; independent grid-search re-implementation of levels/begins; floor/boundaries; `resolution_mode=const` gives r≡1 → probe asserts r>1 early. `dash-render-scale` (CUDA): training render size = floor(H/r) and equals a direct gsplat render with scaled K; no_grad render is full res. `dash-target-downsample` (CPU): loss target = PIL bilinear-reduce reference; loss equals photometric(rgb, ref). |
| C2 primitive scheduler + top-k densify | `schedule.py:densify_rate`, `density.py:prune_and_densify` | `dash-primitive-schedule` (CPU): rate vs independent Eq. 4 (code variant), cap 0.2, P_fin after 26900. `dash-topk-densify` (CPU): exactly the top-n_densify qualified Gaussians densified, clone/split by scale, prune-first, no size pruning, zero Adam moments, P_add returned |
| C3 momentum budget | `schedule.py:update_momentum` | in `dash-primitive-schedule`: Eq. 5 with max, cap, γ |
| C4 position LR | `density.py:after_backward` | `dash-lr-schedule` (CPU): hold until d, log-linear to 1% at 30000, independent formula |
| Integration | all | `dash-cycle` (CUDA): short runtime.train on demo data with compressed schedule: logs show reduced then full res, eval at full res |

## Fidelity gaps
1. Rasterizer: gsplat instead of 3DGS `3dgs-accel` diff-gaussian-rasterization → speed differs; numerics
   close. Times are not comparable to the paper's absolute minutes; compare the **speed-up vs our
   gsplat baseline** to the paper's 3DGS→3DGS+Ours speed-up (≈1.8× on Mip-NeRF 360).
2. Background white (paper black); SSIM valid conv (paper fused SSIM same padding); random view with
   replacement (paper without) — shared with the baseline.
3. Optimizer step order: official densifies before the optimizer step of that iteration (that step's update
   is lost for all params); runtime steps first. Negligible.
4. Baseline mismatch: our baseline3dgs (gsplat DefaultStrategy, densify until 15000, size pruning) vs the
   paper's 3DGS retrain.
5. Scheduler init time and dataset re-load are outside the timed region (as in the official code, whose
   timer excludes scheduler init).
6. Paper numbers predate the official 2025-08 fix; we follow the current code.

## Files
`plugin/pyproject.toml`, `plugin/ss_dashgaussian/{__init__,link,schedule,renderer,density,losses,probes}.py`,
`plugin/README.md`, `plugin/tests/test_dashgaussian.py`.

## Benchmark
Per-scene numbers reported (supp. Tables 7–9). Scenes: mipnerf360 garden, bicycle (outdoor), room (indoor);
tandt truck, train; db drjohnson, playroom. 30000 steps, `--recipe 3dgs`. Record `training_seconds` and
Gaussian count against `baseline3dgs-*`.

## Iteration 1
Loss `photometric` gains `ssim_padding` (`"same"` default = 3DGS/fused_ssim zero padding 5; `"valid"` = iteration 0).
Only the training loss changes; evaluation SSIM remains the runtime's valid SSIM. Motivation: at r=6–7 the valid
convolution drops ~10–14 % of pixels from the SSIM term during the long low-resolution phase (room, drjohnson gap analysis).
