# AbsGS — review 1

**Approved: no.** 0 critical, 1 major, 4 minor. The densification core matches the official
code exactly. The one major issue is that the plugin leaves out a mechanism that the released
evaluation routine turns on by default. The fix is to add a variant; the `absgs` defaults
themselves do not need to change.

Checked against: `source/code/` (TY424/AbsGS @87d7a5a; `git log -p` shows the defaults below
are unchanged since the initial release 084f94e) and `paper.txt`. Tests: 5 passed. ss-inspect:
no issues.

## Which settings produced the paper's per-scene table
- Paper §4.1 (paper.txt:353-359) says AbsGS is trained with τ_S = 0.001, and "AbsGS-0004/-0008"
  means τ_p(split) = 0.0004/0.0008. Table 2 row τ_S=0.001, τ_p=0.0004 (27.49 / 23.73 / 29.67) is the
  main Table-1 row and the supplementary S=0.001 per-scene rows (garden 27.487, kitchen 31.621).
- Official defaults: `arguments/__init__.py:83` percent_dense 0.001, `:89` densify_grad_threshold
  0.0002 (clone, signed), `:90` densify_grad_abs_threshold 0.0004 (split, abs). `full_eval.py`
  passes only `-i images_4/_2 --eval`, so the table setting = code defaults.
- Plugin: `density.py:18-30` gives clone 0.0002 on the signed gradient, split 0.0004 on the abs
  gradient, grow_scale3d 0.001. `__init__.py:221-228`: `absgs` uses those defaults, and
  `absgs-0008` overrides only split to 0.0008. **This equals the paper's table setting.** The two
  disagree in neither the code nor the plan. The 0.0008-for-both, τ_S=0.01 numbers belong to the
  built-in `gsplat-absgrad` (`splatstudio/methods.py:30`), which this plugin does not use for
  density.

## Issues

### MAJOR — periodic opacity reduction is on in the official evaluation routine; the plugin turns it off and calls it "not used in the paper"
- Official `arguments/__init__.py:92-93`: `use_reduce = True`, `opacity_reduce_interval = 500`.
  `train.py:132-133` runs `reduce_opacity()` (`gaussian_model.py:260`: opacity → min(o, 0.8),
  Adam moments zeroed) every 500 iterations while iteration < 15000, after densify and before reset.
  `full_eval.py` ("the routine used in our evaluation", README) does not turn it off.
- The README's "None of these modifications were utilized in our paper" section names only
  **Initial pruning** (the SfM err/track filter in `read_points3D_binary` + `init_prune`) and
  **Weight-based pruning**. It does not mention opacity reduction, which appears only under
  "New Arguments" ("True by default"; the README says the interval is 3000, the code says 500).
  The paper text never mentions it. So `plan.md` fidelity gap 4, the plugin README and
  `density.py:29` ("not used in paper") claim more than the sources support.
- Short check (garden, 2000 steps, GPU 5, reduce_every=500 vs evidence/smoke-1): PSNR 23.90 vs
  23.74, Gaussians 2.33M vs 2.36M. One seed, so this only says the effect is not negligible
  early on.
- Fix: add an entry point `absgs-reduce` = `_make("absgs-reduce", ..., reduce_every=500)` (the
  implementation at `density.py:66-68` already matches the official order and value). Add it to
  the probe `PAPER` table and benchmark it next to `absgs`. Report both and state that it is
  unknown which one produced the paper numbers. Correct the README and plan wording. Keep
  `absgs` as is.

### MINOR
1. **SfM filtering claim.** It holds only by the README's statement. The released
   `dataset_readers.py:162-170` filters (err > 2, track < 3) whenever `points3D.ply` does not
   already exist, so it is on by default for fresh data. Garden init = 138,766 points (smoke
   config), the unfiltered count, which is consistent with "not used". Weight pruning:
   `use_prune_weight=False` by default (`arguments:95`), `init_prune=False` (`:57`). Both are
   correctly off.
2. **The probe does not exercise "clones never split in the same round".** `density.py:118` pads
   with zeros, as official `densify_and_split` does with `padded_grad` (`gaussian_model.py:424-
   426`). But the clone and split sets are disjoint by scale, so removing the padding only causes
   a shape error. It never changes which Gaussians are chosen. Correct as written; this needs no
   probe.
3. **Densify after the optimizer step** (`runtime.py:207-211`). Official densifies before
   `optimizer.step()` (`train.py:122-146`). The baseline does the same, so this does not affect
   the AbsGS-specific delta.
4. **White-background compositing** (`backends/gsplat.py:48`) adds the alpha term to dL/dμ2D,
   so it enters both accumulators. The official code uses a black background. This is the shared
   protocol gap that is already listed.

## Verified exact (no action)
- **C1 abs gradient.** Official `backward.cu:549-550` takes `fabs(dL_dG·dG_ddelx·ddelx_dx)` per
  pixel, in NDC units (ddelx_dx = W/2). gsplat `RasterizeToPixels3DGSBwd.cu:234,248,270` takes
  `abs(v_xy_local)` per pixel, then warp-sums and atomically adds, in pixel units. The plugin
  scales both by (W/2, H/2) (`density.py:78-82`). These are equal. One shared caveat: gsplat
  drops gradients where opac·G > 0.999 (`:222`).
- **Accumulation.** Official `add_densification_stats` (`gaussian_model.py:479-483`) adds
  ‖g[:2]‖ and ‖g[2:]‖ to separate accumulators with one shared denominator, counted when
  radii > 0. The plugin (`density.py:90-95`) does the same. Division: official grads/denom with
  NaN→0; plugin divides by clamp_min(1), which gives 0 for unseen Gaussians. Equivalent.
- **Clone / split.** `>=` on both, scale `<=` / `>` τ_S·extent, as in official `:427-450`. Split
  uses /1.6 scale and N=2 samples (gsplat `split`, matching official `0.8*N`). Scene extent:
  `runtime.camera_extent` = 1.1 × max camera distance = `getNerfppNorm`.
- **Window and reset.** Densify at step > 500 with step % 100 == 0 and step < 15000 (official
  `train.py:123,128`). Reset to min(o, 0.01) every 3000 before 15000, after densify, with Adam
  moments zeroed (official `:135`, `reset_opacity`). Prune: opacity < 0.005 always; world-size
  > 0.1·extent only after 3000 (official `size_threshold` is None at ≤ 3000). gsplat
  `_prune_gs` gives the same. Screen-size prune: inactive in both (official bug,
  `refine_scale2d_stop_iter=0`). The plugin's own reset replaces gsplat's broken one, and it
  does not run twice because the plugin owns the whole density component.
- **Probes.** `homodirectional-gradient` runs per-pixel autograd and sums |∂L_j/∂μ|. It would
  fail if absgrad were replaced by |Σ| (collision ratio check) or if the NDC scale were missing.
  `split-clone-criteria`: row 4 catches split-on-signed, row 2 catches clone-on-abs, the scale
  rows catch τ_S=0.01, and the count check catches a wrong denominator. The test also checks that
  gsplat-absgrad thresholds fail. The requires_grad / stale-optimizer check (`probes.py:150-154`)
  runs after a real duplicate + split + reset at step 3000. It would catch the ResGS
  `p.clone()`-under-no_grad pitfall. Good.

## Comparison caveat (already in plan)
Our baseline is τ_S = 0.01 (paper's "3D-GS"). Kitchen gains +1.30 dB in the paper vs 3DGS, but
+1.0 dB of that comes from τ_S alone (3D-GS* 31.312). Our gsplat kitchen is already 31.34.
Compare deltas: garden +0.08, kitchen +1.30.

## Launch (after adding `absgs-reduce`; `absgs`, `absgs-0008`, `absgs-noreset` can start now)
```
python3 tools/bench.py launch --exp absgs-mipnerf360-30000 --methods absgs absgs-0008 absgs-noreset absgs-reduce \
  --dataset mipnerf360 --scenes garden kitchen --steps 30000 --recipe 3dgs
python3 tools/bench.py summarize --exp absgs-mipnerf360-30000 --paper reproductions/absgs/paper_results.json --baseline gsplat
```
Compare against `runs/baseline3dgs-mipnerf360-30000` (garden 26.89, kitchen 31.34; garden had
4.76M Gaussians). The 12M budget leaves room, but AbsGS grows 2.8× faster by step 2000
(2.36M vs 0.84M), so watch the garden count.
