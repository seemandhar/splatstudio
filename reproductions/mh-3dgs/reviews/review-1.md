# Review 1 — mh-3dgs / mh-3dgs-mcmc-control

Reviewer: splat-reviewer (one round). Reference: official code SNU-VGILab/MH-3DGS @71ef8f9
(`source/code`, read-only). Plugin: `plugin/ss_mh_3dgs/` (core.py, density.py, control.py,
renderer.py, losses.py, probes.py). Tests: `pytest plugin/tests` → 8 passed; `evidence/verify.json`
all probes passed.

**Verdict: not approved — 0 critical, 1 major (control cannot be evaluated by ss-eval), 6 minor.**
The `mh-3dgs` method itself is a faithful port and can be launched now; fix M1 before launching the control.

## Smoke runs (garden, 2000 steps, --recipe 3dgs)

| run | test PSNR / SSIM / LPIPS | Gaussians @2k | train s | notes |
|---|---|---|---|---|
| mh-3dgs (evidence/smoke-1) | **22.40** / 0.606 / 0.446 | 270,801 | 280 | finite params; test.json written by ss-eval |
| mh-3dgs-mcmc-control (evidence/smoke-control) | **22.83** / 0.616 / 0.449 (validation.json = test alias) | 288,473 | 206 | training OK; **ss-eval crashed** (M1) |
| gsplat garden smoke reference | ~23.0 | — | — | white bg, opacity lr 0.05, ADC |

Both finite and sensible: slightly below the gsplat smoke as expected at 2k with black background,
opacity lr 0.025 and no ADC. MH density history (from checkpoint `density_state.history`, 15 events):
step 600: k=159 views, 810 relocated, 9,218 born (coarse 2,223/4,395 accepted, mean ρ 0.51;
fine 6,995/14,920, ρ 0.47); step 2000: k=144, 1,963 relocated, 8,228 born; event cost 4.5 s
(10.6 s first). Extrapolating ~0.45–0.5 acceptance over the 144 linearly decaying batches
(Σ(1−p) ≈ 72) gives ≈ 0.6–0.65 M births + 139 k SfM ≈ 0.75–0.8 M final, consistent with the
paper's 811,695 on garden — the count is controlled by the decaying batches × σ(I−0.5)·D
acceptance, with no cap. The control grows +5 %/100 steps toward cap 811,695 as intended.

## Fidelity check against the official code

| item | official (file:line) | plugin (file:line) | status |
|---|---|---|---|
| Multi-view error maps: SSIM *similarity* map (11×11, σ1.5, zero same padding, CHW) + per-channel \|I−Î\|, averaged over k views | train.py:83-109, utils/loss_utils.py:59-90 | density.py:150-166, core.py:85-104 | exact (official render is clamped to [0,1], see m2) |
| View subset k=max(int(n(1−p)),1), round robin start (it·k)%n over shuffled train list | train.py:60-67, 249-257; scene/__init__.py:65 | core.py:77-82, density.py:168-172, 128-130 | exact (own seeded shuffle) |
| Importance σ(0.8z(O)+0.5z(SSIM)+0.5z(L1)), unbiased std, +1e-6; world-x/y pixel lookup, channel mean | gaussian_model.py:424-442, 483-500 | core.py:107-129 | exact (probe vs numpy) |
| Order: maps → relocation → importance/voxel counts on post-relocation state | train.py:265-279; gaussian_model.py:581-611 | density.py:173-186 | exact |
| Relocation: dead σ(o)≤0.005, sample alive ∝ opacity, MCMC Eq.9 ratio=bincount+1, clamp [0.005,1−eps], copy to dead, zero moments of sampled | gaussian_model.py:502-564, 584-600 | density.py:176-180 → gsplat ops.relocate | exact |
| Proposals: idx=unique(multinomial(I,B,repl)); std=clamp(s·exp(scale)[idx]/(1+ln(1+denom)),1e-6) per axis | gaussian_model.py:612-622, 638-647 | core.py:149-155 | exact |
| Acceptance σ(I−0.5)/(1+5c), c = pre-step count in floor(x'/v) voxel (double), same counts for coarse/fine | gaussian_model.py:444-481, 623-629; train.py:274 | core.py:132-146, 156-158 | exact (probe vs dict loop) |
| Births copy parent raw sh0/shN/opacity/scale/quat, zero Adam moments, denom reset | gaussian_model.py:371-412, 630-680 | density.py:31-43, 195-198 | exact |
| denom += 1 on visible at densify iters only | train.py:244-245; gaussian_model.py:414-416 | density.py:139-145 | exact |
| Schedules coarse 10→5, fine 2→1, voxel 0.02→0.005, batches int((1−p)·4500/16000)≥1, window 500<it<15000, every 100 | train.py:244-263; arguments/__init__.py | core.py:61-74 | exact (144 events 600…14900) |
| Regularisers 0.01·mean σ(o) + 0.01·mean exp(s) (all axes) | train.py:214-222 | losses.py:13-14 | exact |
| opacity lr 0.025, other lrs = 3DGS | arguments/__init__.py (opacity_lr=0.025) | density.py:86-87; runtime state.py | exact |
| Black background, no antialiasing | arguments (white_background False, antialiasing False) | renderer.py, __init__.py:_renderer | exact |
| No cap; count emerges from acceptance | — | runtime only enforces --max-gaussians 12M | exact |
| Image reload: `splatstudio.data.load_dataset(--data, --downscale)`, `splits["train"]` only, per-view K/viewmat/size equality check vs runtime cameras | — | density.py:91-115 | correct (no test images used) |
| requires_grad pitfall | — | density.py:36, 46-49 (`p.detach()`, `requires_grad=p.requires_grad`, check after each event); gsplat relocate keeps requires_grad | safe |
| World coordinates (world-xy lookup and absolute voxel sizes depend on them) | Inria COLMAP loader (raw) | splatstudio/scripts/colmap_to_transforms.py keeps raw COLMAP world frame (only OpenCV→OpenGL axis flip on c2w, reversed by loader) | consistent |

## Issues

### Critical
None.

### Major
- **M1 Control density crashes under `ss-eval`** (control.py:17-25). `create_density` resolves
  the cap from `sys.argv --data` at *construction*; `ss-eval --run …` rebuilds the method
  (runtime.py `evaluate` → `build(method)`) without `--data`, so it raises
  `RuntimeError: mcmc-budget-density: unknown scene; set config 'cap_max'`
  (evidence/smoke-control.log, last line). bench.py runs exactly `ss-eval --run … --split test`,
  so every control job would end with an empty/failed eval.json. Fix: resolve the cap lazily
  (build the `MCMCStrategy` in `initialize()`, which evaluation never calls), or fall back to
  `json.load(<--run>/config.json)["data"]` when `--data` is absent. Re-run `ss-eval --run
  evidence/smoke-control --split test` to confirm.

### Minor
- **m1 Error-map / birth timing vs optimizer step** (density.py:131 via runtime.py:205-211 vs
  official train.py:248-291). Official computes maps, relocates and births *before*
  `optimizer.step()`; because relocation (`replace_tensors_to_optimizer`) and
  `densification_postfix` rebuild every `nn.Parameter`, `.grad` is None at that step and the
  official Adam step is effectively **skipped on every densification iteration** (≈144 of 30k
  steps). The plugin steps first and then densifies, so maps are from post-step parameters and
  no step is skipped. Already listed as gap 4; extend the note with the skipped-step effect.
  Negligible.
- **m2 Unclamped render in error maps and loss** (density.py:156, renderer.py:33). Official
  `gaussian_renderer/__init__.py:119` clamps the render to [0,1] before the loss and before the
  SSIM/L1 error maps. The plugin's error maps use the unclamped rgb. Cheap fix:
  `rgb.clamp(0, 1)` in `error_maps` (the loss-side clamp is a runtime-wide gap shared with the
  gsplat baseline; record it in report.md).
- **m3 SH-degree off by one at multiples of 1000** (density.py:173; runtime.py:197). Official
  `oneupSHdegree` at `iteration % 1000 == 0` → degree 1 at it=1000; ours `(step−1)//1000`.
  Error maps at steps 1000/2000/3000 use one degree lower. Gap 6; negligible.
- **m4 Relocation probe does not exercise the plugin path** (probes.py:relocation calls
  `gsplat.strategy.ops.relocate` directly). Deleting the relocate call in `MHDensity._step`
  (density.py:176-180) would not fail any probe; likewise no probe runs `_step` end to end
  (importance → proposals → append). Add a small CUDA probe that calls `MHDensity._step` on a
  fixture with dead Gaussians and asserts none remain, the population grows, and
  `history[-1]` counts match.
- **m5 Control cap = paper count, not our MH count** (control.py:20-25). Matches the paper's
  same-budget row (cap = paper's MH-3DGS count) — correct for comparing with the paper row; for
  the internal MH-vs-MCMC comparison also report our MH run's final count next to the cap. The
  control otherwise equals 3DGS-MCMC as in gsplat (`MCMCStrategy` defaults: noise_lr 5e5,
  refine 500→25000 every 100, +5 %/event, min_opacity 0.005, relocation) + same regularisers,
  opacity lr 0.05 (MCMC default), black background. Init opacity is 0.1 (runtime) whereas
  official 3DGS-MCMC's SfM init opacity may differ (check ubc-vision/3dgs-mcmc `create_from_pcd`) — record as a control gap if so.
- **m6 plan/paper.md event count** (paper.md §4 says "{600 … 14900}" — 144 events; the
  "p is never 1 … last batch 31/110" line is right). Cosmetic.

## Probe assessment
| probe | detects removal? | note |
|---|---|---|
| importance-formula | yes | independent numpy, also pins the world-xy lookup |
| voxel-acceptance | yes | dict-loop counts + empirical rates for c=0/4 |
| birth-proposals | yes | std ratio, copied attributes, zero moments, trainable leaves |
| relocation | partially | tests gsplat op, not its call site (m4) |
| schedules | yes | grid + anneal values |
| multiview-maps | yes | independent per-view average, subset dependence |
| regularizers | yes | analytic gradients |

## Launch
After M1 is fixed (mh-3dgs alone can go now):

```
mkdir -p /bigdata/sejain/runs/mh-3dgs-mipnerf360 && ln -s /bigdata/sejain/runs/mh-3dgs-mipnerf360 /home/sejain/splatify/runs/mh-3dgs-mipnerf360
python3 tools/bench.py launch --exp mh-3dgs-mipnerf360 --methods mh-3dgs mh-3dgs-mcmc-control \
    --dataset mipnerf360 --scenes garden room --steps 30000 --recipe 3dgs
```
Compare against `baseline3dgs-mipnerf360-30000`; report final Gaussian counts next to the
paper's 811,695 / 764,539 and the control cap.
