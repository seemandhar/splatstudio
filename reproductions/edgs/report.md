# EDGS (plan #33) — reproduction report

**Outcome: matched.** 7 scenes at 30k steps: mean PSNR gap −0.20 dB vs the paper, 6/7 within ±0.5 dB (truck −0.71 is the only miss). Without any densification, EDGS gains +0.22 to +0.73 dB over our gsplat 3DGS baseline.

Paper: *EDGS: Eliminating Densification for Efficient Convergence of 3DGS* (Kotovenko, Grebenkova, Ommer; arXiv 2504.13204, CVPR 2026). Official code: https://github.com/CompVis/EDGS (read, not run; RoMa submodule pinned at `370117431ffc`). Plugin: `plugin/` (`splatstudio-method-edgs`, methods `edgs`, `edgs-nosh`). Benchmarked method: **`edgs`** (correspondence init + SH init + densification-free training).

## Results (test split, llffhold8, 30000 steps, `--recipe splatstudio`)

Ours vs. the paper's supplementary per-scene tables (A3–A8). Train time is SplatStudio `training_seconds` on shared, contended RTX A6000s (other users' jobs at ~100 % utilization), so it is **not comparable** with the paper's A100 minutes. Init time is from `edgs_init.json` (second, parallel build; first sequential build in parentheses). The paper's times include ~2 min of init.

| scene | PSNR ours / paper (Δ) | SSIM ours / paper | LPIPS ours / paper | #G ours / paper (M) | train s (ours) / paper min | init s | init #G (M) |
|---|---|---|---|---|---|---|---|
| mipnerf360/bicycle | 25.14 / 25.39 (-0.25) | 0.783 / 0.792 | 0.186 / 0.161 | 2.90 / 2.3 | 9100 / 31 | 1699 (895) | 4.22 |
| mipnerf360/garden | 27.27 / 27.67 (-0.40) | 0.872 / 0.876 | 0.109 / 0.095 | 3.06 / 2.5 | 12363 / 34 | 1612 (491) | 4.02 |
| mipnerf360/counter | 29.39 / 29.62 (-0.23) | 0.921 / 0.931 | 0.209 / 0.088 | 2.33 / 1.7 | 11299 / 23 | 1609 (2200) | 4.50 |
| tandt/truck | 25.45 / 26.16 (-0.71) | 0.893 / 0.898 | 0.113 / 0.091 | 1.72 / 1.6 | 9020 / 25 | 1426 (544) | 2.70 |
| tandt/train | 22.25 / 22.39 (-0.14) | 0.848 / 0.837 | 0.186 / 0.172 | 1.35 / 1.1 | 8580 / 21 | 1726 (755) | 2.69 |
| db/drjohnson | 29.42 / 29.50 (-0.08) | 0.906 / 0.900 | 0.292 / 0.233 | 1.75 / 1.5 | 8800 / 30 | 2092 (1000) | 2.57 |
| db/playroom | 30.53 / 30.12 (+0.41) | 0.911 / 0.907 | 0.272 / 0.213 | 1.59 / 1.7 | 3378 / 30 | 1648 (1203) | 2.31 |

Mean PSNR gap over the 7 scenes: **-0.20 dB**; 6/7 within ±0.5. These are subset means (3 + 2 + 2 scenes), **not** dataset averages. LPIPS is systematically higher than the paper's (+0.01 to +0.12), most strongly on counter and DB. The 3DGS calibration in `reproductions/_calibration/` shows a similar LPIPS offset for our runtime (white background, `lpips` VGG on [-1,1], valid-conv SSIM).

## Baseline-relative comparison

Ours: `edgs` − our gsplat baseline (`baseline3dgs-<ds>-30000`, same protocol_id, original SfM data). Paper: EDGS per-scene − published 3DGS per-scene (Kerbl et al. supp., `tools/reference_results.json`). The paper's own retrained 3DGS* exists only as averages: M360 +0.53, T&T +0.52, DB +0.04 dB.

| scene | ours − our 3DGS (PSNR) | paper − published 3DGS (PSNR) | ours − our 3DGS (LPIPS) | #G ours / our 3DGS (M) | train s ours / our 3DGS |
|---|---|---|---|---|---|
| mipnerf360/bicycle | +0.22 | +0.14 | -0.057 | 2.90 / 6.59 | 9100 / 6796 |
| mipnerf360/garden | +0.38 | +0.26 | -0.022 | 3.06 / 4.76 | 12363 / 3199 |
| mipnerf360/counter | +0.54 | +0.92 | -0.051 | 2.33 / 1.22 | 11299 / 4129 |
| tandt/truck | +0.25 | +0.97 | -0.066 | 1.72 / 2.49 | 9020 / 3571 |
| tandt/train | +0.73 | +1.29 | -0.049 | 1.35 / 1.26 | 8580 / 5123 |
| db/drjohnson | +0.66 | +0.73 | -0.033 | 1.75 / 3.61 | 8800 / 4438 |
| db/playroom | +0.73 | +0.08 | -0.045 | 1.59 / 2.24 | 3378 / 6540 |

The gain direction matches the paper's. Our mean is +0.50 dB, against +0.63 for paper EDGS vs published 3DGS on these 7 scenes (the paper's own 3DGS* averages: +0.53 M360, +0.52 T&T, +0.04 DB). LPIPS improves on every scene (−0.02 to −0.07). Gaussian counts are lower than our densified baseline on 5/7 scenes; counter and train are the exceptions. The preprocessing cost (RoMa matching, 24–35 min here on contended GPUs) is not included in `training_seconds`.

## Training-view-only (no leakage) evidence

`ss-edgs-init` loads the scene with `splatstudio.data.load_dataset` and uses only `splits['train']` (llffhold8) for every step: k-means reference views, nearest neighbours, RoMa pairs, triangulation, colours and SH observations. The density's camera-extent hook receives training cameras only. Per scene (from `edgs_init.json` vs the sorted[::8] test names):

| scene | train views | reference views | test views | test ∩ train | test ∩ refs |
|---|---|---|---|---|---|
| mipnerf360/bicycle | 169 | 169 | 25 | 0 | 0 |
| mipnerf360/garden | 161 | 161 | 24 | 0 | 0 |
| mipnerf360/counter | 210 | 180 | 30 | 0 | 0 |
| tandt/truck | 219 | 180 | 32 | 0 | 0 |
| tandt/train | 263 | 180 | 38 | 0 | 0 |
| db/drjohnson | 230 | 180 | 33 | 0 | 0 |
| db/playroom | 196 | 180 | 29 | 0 | 0 |

The dataset copy hashes identically to the source (`dataset_sha256` = `source_dataset_sha256`; same transforms.json bytes, hard-linked images), so the benchmark's protocol_id equals the baseline's. Review-1 independently confirmed the training-only path in code and in garden's `edgs_init.json`.

## Method → components

| contribution | implementation | probe |
|---|---|---|
| C1 dense matching (RoMa outdoor, 3 NN by W2C Frobenius distance, ≤180 k-means refs) | `ss_edgs/preprocess.py` (`ss-edgs-init`) | `edgs-roma-matching`: real RoMa, 40 px shift → dx −40.0, dy 0.0 |
| C2 DLT triangulation (Sec. 3.3) | `core.triangulate` | `edgs-triangulation`: synthetic cameras, max error 5e-15; τ_proj filter separates perturbed matches |
| C3 sampling (c^i = max_j c^ij, weights 1 above τ_corr 0.05, multinomial; best NN = argmin max reprojection error; τ_proj 0.01 L1-NDC) | `core.sampling_weights/select_best` | `edgs-sampling` |
| C4 SH init (Sec. 3.5) | `core.fit_sh_residual` | `edgs-sh-fit`, `edgs-sh-composed` |
| C5 densification-free training (official `no_densify` trainer) | `density.EDGSDensity` | `edgs-no-densify` |

`edgs-density`: no clone/split/reset. It prunes opacity < 0.005 every step before 15000 and adds log 0.99 to the opacity logit every 10 steps before 15000. Opacity LR is 0.025, and the means LR follows 3DGS's schedule evaluated at max(step, 8000). Init: isotropic scale 0.0005·distance to the reference camera, opacity 0.5, identity rotation; SfM points are discarded. `ss-verify`: declared_checks_passed for `edgs` and `edgs-nosh` (8 probes); 11 unit tests; ruff clean.

## SH initialization (fixed after review-1)

The paper initializes DC from the reference colour and "all other coefficients with Y⁺O", but does not say how to make the two consistent, and gives no regularizer. The released code has no SH init at all. The literal reading (pinv over all 16 coefficients, then overwrite DC) double counts colour: on garden, the reference-view error is 0.19 once SH bands switch on, and near-parallel view pairs give coefficients up to 217 (0.69 % > 2). **Documented deviation used:** DC = RGB2SH(reference colour). Bands 1–3 fit the best-neighbour residual R = O_nn − O_ref, with rows projected off the reference direction, using ridge λ = 1/16: H = P Y_oᵀ(Y_o P Y_oᵀ + λI)⁻¹R. The reference colour is exact at every degree, and ‖H‖ ≤ ‖R‖/(2√λ) = 2‖R‖. λ was chosen from this conditioning bound, not from any metric. Observations are cached in `edgs_init.npz` (`ss-edgs-init --refit-sh` refits without RoMa). The rebuild left points bit-identical (same points_sha256).

| scene | max |coef| | frac > 2 | p99 per-point max |
|---|---|---|---|
| mipnerf360/bicycle | 1.487 | 0.0000 | 0.626 |
| mipnerf360/garden | 1.313 | 0.0000 | 0.380 |
| mipnerf360/counter | 1.465 | 0.0000 | 0.302 |
| tandt/truck | 1.517 | 0.0000 | 0.457 |
| tandt/train | 1.529 | 0.0000 | 0.484 |
| db/drjohnson | 1.283 | 0.0000 | 0.227 |
| db/playroom | 1.389 | 0.0000 | 0.269 |

Probes:
- `edgs-sh-fit` (synthetic, gsplat CUDA SH at degree 3):
  - reference colour error 1.2e-7;
  - neighbour residual relative error 0.05;
  - |H|/|R| ≤ 1.985;
  - the literal composition fails it (median reference error 0.19).
- `edgs-sh-composed` (prepared garden, 6000 points): renders each point's initial SH colour toward the cameras of its observing training views and compares it with the projected image pixels. MAE is 0.0003 at the reference view and 0.040 at the neighbour; the literal composition gives 0.194 / 0.194 and is rejected.
- Smoke runs, garden, 2k steps: revised `edgs` 24.34 dB (literal-SH v1 24.42, `edgs-nosh` 24.35, gsplat 23.00).

## Truck (−0.71 dB): hypotheses only (not tested)

- **Init density/budget:** T&T uses the official 15k matches per ref. We get 2.70M init points → 1.72M final (paper 1.6M), so the count is not low. The gain over our baseline (+0.25) is below the paper's per-scene gain over published 3DGS (+0.97). Our gsplat baseline (25.21) already matches published 3DGS (25.19), so the shortfall is in EDGS itself, not in the calibration.
- **Unbounded background / sky:** truck has large sky and far regions that RoMa matches poorly. Without densification, regions with no confident matches can only be covered by stretching existing splats. The paper's (unreleased) updated code may handle this differently.
- **White vs black background:** this runtime gap matters more when the init lacks sky coverage.
- **Reference-view selection:** truck has 219 train views but only 180 refs (k-means), so some views contribute no points. That matches the official code but could interact with our seeded k-means differently.
- **SH-init deviation:** our ridge composition could differ from the authors' unreleased implementation. The 2k smoke shows little SH effect, but the 30k effect is unmeasured. An `edgs-nosh` truck run would separate this.

## Fidelity gaps

1. **`--recipe splatstudio` workaround / runtime bug avoided.** The runtime's `--recipe 3dgs` calls `knn_log_scales`, which chunks rows by 4096 but computes `cdist` against *all* N points. That needs a [4096, N] float matrix (61.4 GiB at N = 4.0M), so training OOMs before step 1 for any init this dense. This is a scalability bug in SplatStudio core (it chunks the queries but not the keys); I did not modify the core. EDGS overwrites those scales anyway. With `--recipe splatstudio` the only other runtime difference is scene_scale (0.9-quantile of points instead of camera extent). `edgs-density` undoes it in `set_training_cameras`: it rescales the means LR by 1.1·max train-camera distance / runtime scene_scale (getNerfppNorm, training cameras only). Review-1 verified this as equivalent. Suggested core fix: chunk the keys too, or use a KNN library.
2. SH-init composition and λ = 1/16 are a documented deviation (see above); the paper's composition is unspecified and the released code has none.
3. Official-code values used where the paper differs: 3 NN (paper 2), 25k matches/ref M360 and 15k T&T/DB (paper 20k), L1 reprojection error (paper L2), raw-confidence weights below τ_corr (paper: uniform over the thresholded set).
4. Runtime: white background (official black), random-with-replacement view sampling (official shuffled stack), valid-conv SSIM.
5. RoMa `match()` zeroes certainty for out-of-image flow and clamps it (the official custom forward does not). Reference coordinates are exact pixel centres and colours exact pixels (official: bilinearly resized grid, round(r+0.5)), and the true principal point is used in NDC. Sub-pixel effects.
6. Matches with error > τ_proj are dropped in preprocessing (official: opacity logit −10, pruned at step 1). Equivalent.
7. Camera proximity uses our native W2C (same frame as the COLMAP conversion).
8. Training time excludes the init (reported separately). Per-step pruning triggers a runtime `check_state` and log row; GPUs were shared, so times are not comparable with the paper's A100 numbers.
9. Not reproduced: the paper's 5K/10K efficiency rows, EDGS+ADC variants, sparse-view setting, and the M360 dataset average (only 3/9 scenes run).

## Review items

Review-1 (`reviews/review-1.json`): not approved, 2 major + 3 minor. All addressed.
- **Major: SH colour double counting** → residual, reference-constrained fit; new composed-init probe that fails the literal version.
- **Major: SH outliers (max 217)** → ridge λ = 1/16 from the conditioning bound; max |coef| now ≤ 1.53, 0 % > 2.
- **Minor: stale `--recipe 3dgs` in README/plan** → fixed.
- **Minor: pixel-centre vs (W−1)/2 coordinate convention** → kept and documented (gap 5).
- **Minor: missing T&T/DB #G and time in `paper_results.json`** → added.

Testing fixes: the 3dgs-recipe OOM (gap 1); the built-in renderer-gradient probe was dropped, because it initializes the density on a synthetic scene without an init file and EDGS density consumes no screen gradients.

## Reproduce

```bash
# matcher (into the splatstudio env; no torch/numpy change) + weights in ~/models/roma/
tools/ssenv pip install -q einops kornia timm loguru h5py
tools/ssenv pip install -q --no-deps ~/models/roma/RoMa-3701174          # RoMa commit 370117431ffc
curl -L -o ~/models/roma/roma_outdoor.pth https://github.com/Parskatt/storage/releases/download/roma/roma_outdoor.pth
curl -L -o ~/models/roma/dinov2_vitl14_pretrain.pth https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_pretrain.pth
cd reproductions/edgs/plugin && ../../../tools/ssenv pip install -e . && cd -

# 1) correspondence init (training views only) -> ~/datasets/edgs-init/<dataset>/<scene>
tools/ssenv ss-edgs-init --source ~/datasets/mipnerf360/native/garden --output ~/datasets/edgs-init/mipnerf360/garden --matches-per-ref 25000
tools/ssenv ss-edgs-init --source ~/datasets/tandt_db/native/truck     --output ~/datasets/edgs-init/tandt/truck       --matches-per-ref 15000
#    (bicycle, counter: 25000; train: tandt; drjohnson, playroom: db with 15000)
#    SH refit only: tools/ssenv ss-edgs-init --source <same> --output <dir> --refit-sh

# 2) verify + train + eval
tools/ssenv ss-verify edgs --device cuda
tools/ssenv ss-train edgs --data ~/datasets/edgs-init/mipnerf360/garden --output runs/x --steps 30000 \
    --recipe splatstudio --num-gaussians 10000000 --max-gaussians 12000000
tools/ssenv ss-eval --run runs/x --split test

# benchmark: datasets.json keys mipnerf360-edgs / tandt-edgs / db-edgs (native_root ~/datasets/edgs-init/<ds>)
python3 tools/bench.py launch --exp edgs-mipnerf360-edgs --methods edgs --dataset mipnerf360-edgs \
    --scenes bicycle garden counter --steps 30000 --recipe splatstudio
python3 tools/bench.py summarize --exp edgs-mipnerf360-edgs --paper reproductions/edgs/paper_results.json
```
Runs: `runs/edgs-{mipnerf360,tandt,db}-edgs/edgs/<ds>-edgs-<scene>/`; baselines `runs/baseline3dgs-<ds>-30000/gsplat/`.

## Workspace
`paper.md`, `paper_results.json`, `plan.md`/`plan.json`, `plugin/`, `evidence/` (verify.json, test.json, smoke runs, init-logs), `reviews/review-1.json`, `bench_request.json`, `state.json`.
