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

## Gap analysis: indoor growth
**Question**: is the stage-1 Gaussian explosion of `mvgs` on room/counter (112k → 1.06M at step 1000 → 11.5M at 1800, then OOM / 12M budget) a port bug?
**Verdict**: it is **faithful to the official code** (xiaobiaodu/MVGS @ 36d073a). No plugin bug was found and nothing was changed.
The driver is **cross-ray densification (CRD)**, not MVAD. Evidence is in `evidence/indoor-growth/`.

**Code comparison (official → port)**
- *MVAD* (train.py:93, 139-151). `camera_t = C/‖C‖` is the unit direction from the world origin, recomputed from the current step's M views at every densify step.
  It is not cumulative: `densify_t` is rebuilt from `opt.densify_grad_threshold` each time, and it halves iff any pair has ‖t_i − t_j‖ > 1.
  The port (`geometry.mvad_threshold`, `density.densify`) does the same. Our native world frame is the COLMAP frame (camera centres equal the
  `sparse/0/images.bin` centres to print precision, `colmapcheck.py`), so the directions match. With 48 views of a 360° capture, β = 1e-4 at every densify step.
  The author confirms per-densify-step use and says the halving is intended (issue #4).
- *Statistics* (train.py:131-134, gaussian_model.py:522). The statistics come only from the last view's `viewspace_points.grad` and `visibility_filter`, with `denom += 1` per iteration and an NDC-scaled norm.
  The port does the same: last view only, pixel grad × W/2, H/2. The other 47 views affect the statistics only through the parameters.
- *Schedule* (train.py:136, 183; arguments/__init__.py; run_360.py). Each stage restarts at iteration 1. Densification runs when 500 < it < 15000 and it % 100 == 0, so stages 1-3 densify at 600…2900 (24 steps each).
  Opacity resets happen at 500 (white background) and at 3000. `size_threshold` is None for it ≤ 3000.
  Stage 1 is -r 8 of the full-resolution images: 389×259 for room, and the port computes the same size. Stats carry over the stage boundary in both: the official checkpoint holds `xyz_gradient_accum`/`denom`, and the port never zeroes them at a boundary.
  The official screen-size prune (`max_radii2D > 20`) never fires. `densification_postfix` (gaussian_model.py:347), which `densify_and_clone` always calls, zeroes `max_radii2D` before the prune.
  So the port's world-size-only prune is equivalent.
- *CRD* (train.py:154-181, gaussian_model.py:376-510). Every densify step builds one AABB per sorted view pair: 48·47/2 = **1128 cuboids**.
  Each AABB spans the closest-point midpoints of four corner rays of the max-L1 20×40 windows, with the `boxes[j]` index quirk. Every Gaussian strictly inside *any* cuboid is cloned, whatever its gradient or size.
  A literal port of the official loop matches `geometry.cross_ray_boxes` to 7e-7 on 12 room cameras at -r 8 (`crd_port.py`).
  The rays are unbounded lines (no t > 0, no distance check), so skew rays from distant views give midpoints far from the surfaces. A few cuboids have edges of 10-390 scene units, against a room p2-p98 extent of 7×9×12.

**Diagnostic** (room, stage 1 only: 48 views at -r 8, 1200 steps, GPU 6/2, instrumented `densify`). The table gives counts at each densify step, before densification.

| it | N | grad ≥ β (clone+split) | in a CRD cuboid | CRD-only clones | N, CRD off |
|---|---|---|---|---|---|
| 600 | 113k | 29k | 74k (66 %) | 61k | 113k |
| 700 | 180k | 37k | 148k (83 %) | 128k | 125k |
| 800 | 336k | 37k | 233k (69 %) | 216k | 163k |
| 900 | 569k | 41k | 115k (20 %) | 108k | 203k |
| 1000 | 680k | 42k | 131k (19 %) | 124k | 248k |
| 1100 | 808k | 42k | 222k (27 %) | 215k | 296k |
| 1200 | 1.02M | 42k | 498k (49 %) | 485k | 343k |

- With CRD on, the run reaches 1.46M after step 1200, at a 4.0 GB peak. CRD-only clones are about 1.35M of the 1.35M net growth: gradient densification adds about 42k per step and pruning removes about as much.
- With CRD off, growth is linear (+40-48k per step) and the run reaches 0.39M at 1200, at a 2.9 GB peak.
- The gradient-selected set is bounded, at about 37-48k per step whether CRD is on or off. So MVAD's halved threshold roughly doubles that set (grad ≥ 2β is about 15k), but it does not explode.
- CRD clones 20-83 % of all Gaussians at each densify step, i.e. it multiplies N by about 1.2-1.8 every 100 iterations. With 24 densify steps in stage 1 (and 24+24+143 more later), the count has no bound. Paper-style box indexing (`box_index: paper`) covers even more points (e.g. 767k vs 498k at it=1200).

**Official code and memory**
- The README says `run_360.py` "requires at least 40G GPU memory". The official trainer keeps all 48 per-view graphs, including the per-view `get_scaling`/`get_features`.
  That is the same structure as our pre-memfix renderer, measured at about 16 GB per million Gaussians. So the released code would OOM on a 40-48 GB GPU at about 2.5-3M Gaussians, which room reaches at about step 1300.
- Several issues report the same thing: #29 (users get 2-3× the 3DGS point count, e.g. bicycle at 9-10M), #35 (OOM at 24 GB even with `--mv 6`), #28 (last stage killed) and #6.
- Our finished outdoor garden run (M = 3, 3 cuboids per step) ends at 13.1M Gaussians, against about 5.8M for 3DGS.
- Paper Table t_parameter claims 0.8M Gaussians on average for 3DGS+MVGS (vs 1.1M for 3DGS). The released code is incompatible with that claim. The gap is between paper and code, not between code and port.

**Consequence**
- The faithful indoor recipe (`mvgs`, `--steps 39000`) cannot finish on room or counter: stage 1 alone passes 12-20M Gaussians by step ~1800-1900.
- With the memory-fixed renderer, stage-1 peak is about 1.6 GB per million Gaussians (17.7 GB at 10M, ~21 GB at 12M). So a 48 GB GPU would OOM at roughly 25M, and the runtime budget error at `--max-gaussians 20000000` would come first.
- Raising the budget does not help, because the growth is geometric. Record the indoor recipe as **not runnable as released** (a failed run, kept).
- A runnable variant needs a labelled deviation, e.g. CRD off. That variant has no matching paper row: Table t_ablation adds CIG only after CRD.
  It needs a new entry point (bench/ss-train cannot override component config). Estimated memory: about 1.2M Gaussians at the end of stage 1 (< 5 GB), then at 8 views/w1600 in stage 4 a 3DGS-like count with β/2 (≈3-6M, ≈8-12 GB).
- Outdoor (`mvgs-outdoor`) is unaffected: the plugin is unchanged and its eval.json stands.

## Final verdict: PARTIAL (outdoor matched; indoor not runnable as released)
| scene | ours | paper (3DGS+Ours) | Δ | our baseline | Gaussians |
|---|---|---|---|---|---|
| garden (mvgs-outdoor, 30k) | 27.32 / 0.865 / 0.119 | 27.23 | +0.09 | 26.93 | 13.1M |
| counter (mvgs, 39k) | — | 30.21 | | 29.04 | OOM / explodes |
| room (mvgs, 39k) | — | 32.84 | | — | 11.5M by step 1800 → OOM |

- **Indoor:** the faithful port of the official recipe grows geometrically. Cross-ray densification clones 20–83% of all Gaussians at every densify step, and stage 1 would pass 12–20M by about step 1900.
  - The official code has the same behaviour: README "≥40 GB", issues #6/#28/#29/#35.
  - The paper's Table t_parameter (about 0.8M Gaussians) is inconsistent with the released code.
- **Not run:** a CRD-off indoor variant was not run. It would be an ablation with no matching paper row, and the batch rules say no ablations.
- **Kept:** the failed indoor runs remain in runs/mvgs-indoor-1600{,-m1}.
