# HGS (ss_hard_gs) — review round 1

**Verdict: approved for benchmarking (0 critical, 0 major, 6 minor).** No fidelity bug found in PGHGS/REHGS or in the
Scaffold backbone. The gradient statistics are on the right scale, and an independent calibration against the paper's
own τ=7e-5 experiment supports this. Growth is large, but it is the growth the paper describes. The memory risk is
operational (shared GPUs), not a bug.

## Checks run (garden, mipnerf360-pilr, 2000 steps, `--recipe 3dgs`)
I used a wrapper that monkey-patches `HGSDensity.adjust_anchor` to log statistics. Plugin code is unchanged.
Script: `/tmp/claude-1013/-home-sejain/6ad4135e-056e-4bd4-9844-6b8764bea041/scratchpad/hgs/diag.py`.
- hard-gs reproduces smoke-2: 888,934 anchors at 2k; grown per interval 247k → 162k → 131k → 111k → 105k.
- **Scale calibration.** Median over neural Gaussians of g'_3 / avg is **2.65–2.86** at every interval. So PGHGS with
  λτ = 2e-4 behaves roughly like the average rule at τ/2.7 ≈ 7.4e-5. The paper's supplement (tab/grad.tex) says that
  lowering Scaffold's τ to **7e-5** gives "a Gaussian count comparable to HGS" (0.64 GB vs HGS 0.57 GB, Scaffold
  0.17 GB). I ran `scaffold-gs-hgsbase` with τ = 7e-5 under the same patch: **701k anchors at 2k**, grown/interval
  93k at step 2000, against HGS's 889k / 105k. The two are in the same ratio band as the paper's (HGS ≈ 0.9–1.3 × the
  7e-5 Scaffold run). With the NDC scaling (×W/2, ×H/2; `density.py:328-330`) PGHGS sits at the operating point the
  paper implies. If the scale were wrong (e.g. pixel units, a factor of ~650), PG would select either nothing or
  everything. **Not a bug.**
- PG level-0 hard ≈ 276–321k/interval. 75–99 % of them also pass Scaffold's denom > 40 mask (pg0_and_denom40 =
  240k/321k at step 2000). REHGS: ~1.4k hard / ~2.8k REHGS-only candidates per interval.
- Peak allocated memory is about 3.4–3.5 KB per anchor (2.78 GB at 785k; smoke-2 3.0 GB at 890k), roughly linear.

## Memory estimate for a 30k garden run
Anchors grow only until 15k (130 more intervals after step 2k). Projections:
- per-interval growth decaying ≈1/t (the current trend): ≈ 0.9M + 105k·20·ln 7.5 ≈ **5M anchors → ~17 GB**.
- paper ratio (HGS ≈ 3.4–3.8 × Scaffold; our Scaffold-like HAC garden ends at ~0.8M): **3–3.5M → ~11–12 GB**.
- worst case, growth stuck at ~100k/interval: ~13.9M anchors. The `--max-gaussians 12000000` budget stops the run
  at about 12M, which is about 42 GB.

**Prediction: a 46 GB GPU is not exceeded unless growth stops decaying. A *shared* GPU with < 25 GB free can OOM.**
Treat > 5M anchors at 15k as over-growth relative to the paper (> 6× Scaffold) and investigate (see m1).

## Issues
| sev | where | issue | paper ref | suggestion |
|---|---|---|---|---|
| minor | density.py:354-358 | Hard candidates bypass Scaffold's `offset_mask` (denom > 40). A more literal code-minimal port, `cand=((avg≥τ_i) \| hard) & offset_mask`, keeps it. Measured effect: −25 % PG candidates at 2k (fewer later). The REHGS "≥2 views" sentence suggests that the authors did *not* need 40 views for REHGS. PGHGS is ambiguous. | §5.1 Eq. 6, §5.2 last ¶ | Add a config flag `hard_respect_success_mask` for a gap-analysis ablation if counts overshoot; keep the default. |
| minor | density.py:268-270, 300-303, 452 | The top-k window is one interval (reset every 100 its). Eq. 3/5 share "the n viewpoints during the M iterations". Scaffold's own n (denom) spans until denom > 40. A literal reading on the Scaffold code would sort the same n gradients (reset where offset_mask). Measured kth/avg ≈ 2.7 with the 100-it window; a longer window raises kth, so the current choice is the conservative one. | Eq. 3, Eq. 5 | Document in report.md as a choice; no change. |
| minor | density.py:357 | PG uses λ·τ_i (level-scaled). Eq. 6 literally says λτ_grad. A level-scaled threshold is the natural Scaffold reading because Scaffold passes τ_grad and scales it inside `anchor_growing`. A literal unscaled threshold would grow more at levels 1–2. | Eq. 6 | Keep; list as ambiguity. |
| minor | density.py:347, 424 | REHGS needs ≥2 *hard detections*. The text "such Gaussians should be seen by at least two views" also admits "hard once, visible in ≥2 views". REHGS-only adds only ~2.8k candidates/interval, which may be weak compared with the paper's Model-B gain (T&T LPIPS 0.170 → 0.155). | §5.2, tab/ablation.tex | Run a 2k OG+REHGS-only smoke (pghgs off) and report Δanchors vs scaffold-gs-hgsbase. If < 2 %, note "REHGS likely under-powered". |
| minor | density.py:35 | `total_steps` is fixed at 30000 regardless of `--steps`, so the LR schedules ignore the CLI value. This is correct for the 30k benchmark; smoke runs at 2k use a 30k schedule. | — | Note only. |
| minor | bench_request.json | Checkpoints are about 1.1 KB per anchor (986 MB at 890k), so a 5M-anchor run is ~5.5 GB per scene, on a nearly full `/`. | CLAUDE.md Disk | Symlink `runs/hard-gs-*` to /bigdata before launch. Log per-interval `events` and peak memory, and ask bench placement for ≥ 30 GB free. |

## Backbone check vs Scaffold-GS @59c833b
These all match the official code:
- MLPs: [feat, view dir], tanh opacity > 0 mask, sigmoid colour, cov → sigmoid·scaling[3:], normalised quaternion,
  offsets·scaling[:3].
- `training_statis`: neural opacity clamp at 0, `combined_mask` = vis → sel → radii > 0.
- `anchor_growing`: τ·2^i, rand > 0.5^(i+1), voxel 0.001·16/4^i, dedup against current anchors, scatter-max feat,
  log(cur_size) scaling.
- Pruning: opacity_accum < 0.005·demon and demon > 80; the 0.05 raw-scale clamp on every prune.
- Learning rates: offset 0.01 → 1e-4 × camera extent (guarded), feat 0.0075, scaling 0.007, MLP schedules (cov
  constant 0.004), anchor lr 0, schedule lr(step) at step.
- Loss: 0.8 L1 + 0.2 D-SSIM + 0.01·mean(prod scale).

Eval path: MLPs travel in `hgs_blob` with a checksum, repacked after every step and after pruning. ss-eval runs in a
fresh process and reproduces the training-time metrics (smoke-2 eval, blob-eval probe).

requires_grad pitfall: `append_anchors`/`keep_anchors` rebuild Parameters from `detach()` with `requires_grad=True`,
and `assert_trainable` runs after every `adjust_anchor`. OK.

Known gaps (already listed in plan.md): white background, gsplat rasterizer, Adam-before-growth ordering, SSIM valid
conv in the loss.

## Equation coverage
- Eq. 3: exact (`adjust_anchor`).
- Eq. 5: exact (running top-3 = sort; probe checked).
- Eq. 6: exact, with the level-scaling choice.
- Eq. 7: exact (|S_i| from argmax_i w_i, τ_large·W·H).
- Eq. 8: exact (3DGS SSIM map at floor(means2d), the gsplat pixel convention).
- ≥2-views rule: one of two readings.
- Effi-HGS: not implemented (not the reproduced row).

## Probes
- pghgs-criterion and rehgs-criterion would fail if their mechanism were removed (the enabled/disabled growth
  assertions).
- No probe checks the gradient *scale*. The 7e-5 calibration above serves that purpose. Consider recording it in
  evidence/.
