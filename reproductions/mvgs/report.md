# MVGS (plan #54) — build report

**Status**: plugin built, all probes pass, smoke-tested; full benchmark not yet run (`bench_request.json`).

- Paper: arXiv 2410.02103; official code https://github.com/xiaobiaodu/MVGS @ 36d073a (read, not run).
- Methods: `mvgs` (indoor recipe: CIG M=48/24/12/8 at -r 8/4/2/-1, 3000/3000/3000/30000 it, `--steps 39000`),
  `mvgs-outdoor` (M=3 at w1600, 30000 it). Package `plugin/` (`ss_mvgs`).
- Multi-view batches are built inside the components: the density controller reloads the training images,
  samples M distinct views per step from the official shuffled stack and owns the stage schedule; the renderer
  renders all M views at the stage resolution; the loss sums them (Eq. 4). Densification statistics come from the
  last view only (official `add_densification_stats` behaviour); MVAD and CRD use all M views of the step.
- Probes (ss-verify `declared_checks_passed`): multiview-sum, cig-schedule, mvad, crd, cycle + built-ins.

## Smoke (2000 steps, w1600)
| run | PSNR | SSIM | LPIPS | #G | s |
|---|---|---|---|---|---|
| gsplat garden | 22.84 | 0.625 | 0.439 | 0.78M | 131 |
| mvgs-outdoor garden | 23.98 | 0.734 | 0.316 | 3.19M | 204 |
| mvgs counter (compressed 4 stages 600/600/400/400) | 27.12 | 0.879 | 0.308 | 0.26M | 1002 |

MVAD halves the threshold at every densify step on garden (unit camera directions from the COLMAP origin
almost always differ by >60°), so Gaussian growth is ~4× the baseline at 2k steps — faithful to the code but in
tension with the paper's "fewer Gaussians" claim (Table t_parameter); watch #G/memory in the benchmark.

## Fidelity gaps
1. gsplat rasterizer (saturated-alpha gradient drop, no 0.99 alpha clamp) — shared with the baseline.
2. Runtime applies Adam before density control on densify steps (3DGS skips that update) — shared.
3. Coarse-stage targets are antialiased-bicubic resizes of the w1600 images, not PIL resizes of full-res;
   sizes via `full_width_ratio = 3115/1600` (widths exact, heights ±1 px).
4. One continuous run instead of 4 checkpoint-chained processes; carried state identical (params, Adam moments,
   SH degree, zeroed stats); stage-local densify window, opacity resets (it%3000, it=500) and position LR restart.
5. Rays use the dataset's principal point (3DGS assumes the image centre).
6. CRD keeps the official `boxes[j]` indexing quirk (`box_index: official`); MVAD uses translation only (code).
7. Paper's 3DGS rows are 3DGS-paper numbers (images_4/2), MVGS was run at -r -1 (w1600): compare our gain over
   the w1600 baseline with care. No official T&T/DB recipe → not benchmarked.
8. Smoke tests used `tools/smoke.py` (runtime.train with a truncated/compressed stage list) because `ss-train`
   refuses `--steps` different from the schedule.

## Memory fix (indoor stage 1 OOM)
**Symptom**: `mvgs` on mipnerf360-1600 counter/room OOMed on 48 GB A6000s at step ~1100-1500 of stage 1
(48 views/step at -r 8, 2.5-3.4M Gaussians; process ~44 GB).
**Root cause** (measured, `torch.cuda.max_memory_allocated`, 48 views at -r 8, Gaussians replicated from the
gsplat counter checkpoint): the old renderer kept 48 stored per-view graphs until the single backward, at
**~16.3 GB per million Gaussians** (18.7 GB at 1M, 26.9 GB at 1.5M, so ~51 GB at 3M). Per Gaussian per view that is
~340 B. Most of it (192 B) was `torch.cat((sh0, shN))`, rebuilt for each view, plus a per-view `exp`/`sigmoid`.
The rest (~115 B) was gsplat's projection/SH tensors. After the backward, the runtime's `outputs` also kept all 48 views'
`info` alive through the next step's forward (~2.4 GB per million).
**Change** (maths unchanged):
- `ss_mvgs/renderer.py`: `activations()` computes `exp(scales)`, `sigmoid(opacities)` and the SH concatenation once
  per step, and every view shares them. `render_view_checkpointed()` wraps the non-statistics views in
  `torch.utils.checkpoint` (non-reentrant), so only their image outputs are kept and their rasterizer graph is
  recomputed during backward. The statistics view (the last one; every view with `stats_views: all`) is rendered
  as before and keeps `info`/`means2d`.
- `ss_mvgs/density.py`: `_make_batch` marks each view's `stats` flag. At the end of `after_backward`, the
  non-last views' `info` is released.
**Equivalence** (`evidence/memfix/equiv.json`; 8 counter views, the real 1.19M-Gaussian gsplat checkpoint,
old renderer vs new, same batch):
- The loss is bitwise identical. Every view's rgb is bitwise identical, and last-view radii and visibility
  counts are equal.
- The `grad_accum` statistics agree to ≤2.5e-7 relative.
- Parameter gradients agree to max-abs/max ≤1.4e-5 for means/scales and ≤5e-7 for opacities/sh0/shN at -r 8 and -r -1.
- quats agree to 1.5e-4 at -r 8 and 1.2e-3 at -r -1. This is gsplat's atomic-add nondeterminism: the old code
  compared with itself gives 2e-4 and 4.5e-2. The L2-relative old-vs-new difference is smaller than old-vs-old.
- New pytest `test_memory_path_matches_per_view_graphs` checks the same on a synthetic scene.
**Peak memory, 48 views at -r 8**:

| Gaussians | 1M | 1.5M | 3M | 6M | 10M |
|---|---|---|---|---|---|
| old (GB) | 18.7 | 26.9 | ~51 (extrapolated; doesn't fit) | – | – |
| new (GB) | 3.2 | 4.0 | 6.4 | 11.2 | 17.7 |

At 10M, most of the new peak is params + Adam + grads. Step time is +6-8% at 1-1.5M and +17% at 3M.
Stage 4 (8 views, w1600, 3M) uses 7.3 GB vs 15.3 GB at about the same speed (+4%). Outdoor (3 views, 2M) uses
4.8 GB vs 5.9 GB at the same speed.
pytest (17), ruff and `ss-verify mvgs` / `mvgs-outdoor --device cuda` pass. The finished garden `mvgs-outdoor`
result is mathematically unaffected. Its checkpoint can no longer be re-run with `ss-eval`, because the plugin
source manifest changed. Its eval.json already exists.
