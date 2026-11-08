# Perceptual-GS — review 1

Reviewer: splat-reviewer (one round). Scope: plugin `ss_perceptual_gs` vs official
`source/code` (eezkni/Perceptual-GS) and arXiv 2506.12400. Builder followed code over paper.

**Verdict: approved (0 critical, 0 major, 5 minor).** Ready to benchmark.

## Checked against the official code (all match)

| item | official | plugin |
|---|---|---|
| Sens. map: L + gamma 1.5 via `Image.point`, Sobel pad 1, `>0.05`, avg_pool 5 (stride 5), nearest upsample, `>0.3`, uint8 PNG | `preprocess.py:24-118` | `core.py:25-35`, `sidecar.py:67-82` |
| Full-res `images/` → PIL default (bicubic) resize to train size, /255 | `camera_utils.py:41-45`, `general_utils.py:21-27` | `sidecar.py:76-82` (raw = `360_v2/<scene>/images`, data = `pilr`, matches `run.py -r 4/2`) |
| Frame matching | by image name | by viewmat, tol 1e-4 (`sidecar.py:110-116`), errors on a miss |
| Loss `0.9·(0.8L1+0.2DSSIM) + 0.1·BCE(1−R[1−σ(ε)], gt)` while `it ≤ 15000`; black bg for the sens render; gradients reach geometry too | `train.py:105-117` | `loss.py:51-87` (gsplat `backgrounds=None` → black; separate means2d, so densify stats are only from RGB and scaled by 0.9, as in official) |
| Weights ω = max_v Σ_pix αT at `it%1000==0 or it%1500==0`, after the optimizer step, before densify | `train.py:132-144` | `density.py:103-106,138-146` (the runtime always steps first; on these steps official also steps first, so they match exactly) |
| HD: σ≥0.9 & ω>25 → clone if small / split if large; when split_only, split regardless of size, no HD clone. MD: 0.3<σ<0.9 & ω>10 → split, no size test. Both from step 1000 | `gaussian_model.py:386-407,435-444` | `density.py:167-188` |
| Standard 3DGS grad clone/split still runs (0.0002, percent_dense 0.01); prune α<0.005, world-size >0.1·extent after 3000; screen-size prune inert | `gaussian_model.py:463-475` | `density.py:174-211` |
| OD on all clone-selected originals (grad and HD), every densify step: `α^k/(1+√(1−α^k))`, k=1.2 | `gaussian_model.py:446-448` | `density.py:194-198`, `core.py:240-243` |
| Decision at 600, before the densify at 600: γ = share of top-25% largest Gaussians with σ∈(0.3,0.9); sparse if γ>0.55; β = mean over train views of #(sens>0)/#pix; split_only if β>0.85 | `train.py:166-198` | `density.py:154-164`, `core.py:224-237` |
| Sparse: skip densify and depth-reinit at 5000/10000 (prob 1−T, n = H·W·3.5M/(H·W·|V|) with replacement + unique, GT colours, opacity/sens 0.1, KNN scales, fresh Adam); no opacity reset; self-reinit from xyz+DC at 15000 | `train.py:207-264`, `gaussian_model.py:486-512` | `density.py:101-114,240-288` |
| Non-sparse: opacity reset `min(α,0.01)` every 3000 while `<15000`, after densify | `train.py:250-252` | `density.py:109-110,218-223` |
| Sensitivity field: lr 0.05, init logit(0.1), split/clone copy it, trainable after every op | `gaussian_model.py:154,180,419,456` | `__init__.py:51`, `state.py:43-46`, `_check_trainable` (`density.py:315-319`); probe pgs-guided-densify checks the grad after densify |
| Depth kernel: glm column-major R = Rᵀ (world→local), 3σ ellipsoid, t_mid=−b/2a, `if(depth<0) continue` before the T update | `_ms/forward.cu:578-625` | `ms_raster.cu:116-149` (same, including the skipped T update) |

**Data leakage: none.** The sidecar stores maps for all frames, test included. Only training
cameras ever look them up: the BCE runs on training frames and `_decide` uses
`set_training_cameras` (train split only). `prep.json:beta_all_frames` is only a log. Test
frames are never used for selection.

## Smoke −0.57 dB at 2k (counter): timing artifact, not a bug

- Same seed, so both runs see the same view sequence. Per-step training photometric loss
  (PGS value /0.9) tracks the gsplat baseline within ±0.002 at every logged step, 200–2000
  (`evidence/smoke-*/training.jsonl`). No sign of a broken RGB path.
- Step 2000 is an HD step. The densify there adds +64k clones, each with OD
  (`smoke-1.log`, 376,738 → 456,887). OD lowers the opacity of both the original and the
  clone (α=0.1 → 0.032 each, composite 0.063 < 0.1), so the scene is temporarily
  under-opaque until Adam recovers. The eval at 2000 runs right after this.
- Direct check at 1900 steps, before the HD event (scratchpad runs on GPUs 5/3, test split):
  **PGS 24.888 / SSIM 0.8417 / LPIPS 0.358 vs gsplat 24.659 / 0.8373 / 0.367 (+0.23 dB)**.
  At 2000 the result is −0.57 dB. That is a +0.80 dB swing over one step, so the dip at 2000
  is the transient from the HD+OD event. It is not a defect in the steady state.

## Issues

### Minor
1. **Alpha cap 0.999 vs 0.99** (`csrc/ms_raster.cu:107`; official `_ms/forward.cu:363,552`, and
   the RGB rasterizer `forward.cu:343`). The plugin is self-consistent with gsplat's renderer,
   which also caps at 0.999. Saturated Gaussians get slightly larger αT and absorb more of the
   pixel, so ω>25 / ω>10 selection drifts a little. Already listed as fidelity gap 3. Keep it
   there.
2. **Stop test `next_T <= 1e-4` + break vs `test_T < 1e-4` + `continue`** (`ms_raster.cu:110`).
   Equivalent in practice. Mention it under gap 3.
3. **Optimizer step on ordinary densify steps** (runtime `runtime.py:205-211`). Official calls
   densify before `optimizer.step()`, and newly concatenated tensors have no grad, so
   non-HD/MD densify steps skip Adam. HD/MD steps match. Gap 2. Fine.
4. **Image cache for depth reinit** (`density.py:226-238`) is filled only from rendered
   frames. With random-with-replacement sampling, P(a view is unseen by 5000) ≈ 0 for
   ~180–270 views. It raises instead of silently degrading, which is correct. Cost: about
   1 GB of GPU uint8 on counter.
5. **Decision timing**: official evaluates γ/β before the step-600 Adam update. The plugin
   evaluates after it (one step later). Negligible.

### Not issues (checked)
- `predicted_sensitivity` clamps 1−r into [0,1] (`core.py:251-253`). r ∈ [0,1] by
  construction, so the clamp never cuts a gradient in practice.
- Bicycle β = 0.93 → split_only. Counter β = 0.718, γ = 0.455 → dense/clone mode, no reinit.
  This is consistent with the paper's claim that dense outdoor scenes use split-only.
- Means-LR schedule, white RGB background, SH timing: runtime-wide, shared with the baseline.
  Compare deltas.

## Probes
All 9 pass (`evidence/verify.json`). Each probe compares against an independent reference
(numpy/loop Sobel, brute-force blend, finite-difference BCE, closed-form OD composite, explicit
γ sort), and each would fail if its mechanism were removed. One weakness: no probe asserts the
skip of densify on sparse 5000/10000 steps. The forced-sparse smoke covers it.

## Launch
```
mkdir -p /bigdata/sejain/runs/perceptual-gs-mipnerf360-pilr-30000 && \
  ln -s /bigdata/sejain/runs/perceptual-gs-mipnerf360-pilr-30000 runs/perceptual-gs-mipnerf360-pilr-30000
python3 tools/bench.py launch --exp perceptual-gs-mipnerf360-pilr-30000 --methods perceptual-gs \
  --dataset mipnerf360-pilr --scenes bicycle counter --steps 30000 --recipe 3dgs
```
(Baseline
`baseline3dgs-mipnerf360-pilr-30000`. Targets: bicycle 25.956 (+0.339 over 3DGS*),
counter 29.452 (+0.308). Put the run dir on /bigdata as CLAUDE.md says.)
