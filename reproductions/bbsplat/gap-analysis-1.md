# Gap analysis 1 — BBSplat (bicycle matched, truck −1.99 dB)

| scene | ours | paper | gap | budget |
|---|---|---|---|---|
| bicycle (`bbsplat`) | 23.12 | 23.28 | −0.16 (noise/matched) | 60K |
| truck (`bbsplat-tnt`) | 22.20 | 24.19 | −1.99 (method/protocol) | 30K |
The paper's 3DGS row is budget-constrained (truck 20.43 @80K), so baseline-offset decomposition is not
meaningful; the paper's budget-matched 3DGS-MCMC truck is 23.91.

## Evidence (final checkpoints, diagnostics only; no test-set selection)
- Truck: **6284 / 30000 billboards (21 %) sit on the sky sphere** (r ≈ 212 = max|xyz| of the SfM
  points, inflated by far outliers; started at 10K = 33 %). Billboards off the sphere alone cover
  **99.0 %** of test pixels (alpha ≥ 0.5), i.e. the sky sphere buys almost no coverage.
  Bicycle: 8793 / 60000 (15 %), off-sphere coverage 97.0 %.
- Truck train PSNR (every 8th train view) 23.11 vs test 22.20 → under-fitting / capacity-limited,
  so freeing budget from the sky sphere should help. Bicycle train 22.51 / test 23.12.
- Loss terms: texture reg ~6e-4, opacity reg ~7.5e-4 vs photometric 0.07 — not dominant. Population pinned at
  cap from step 1 (only relocation acts), as designed.
- Per-view test PSNR on truck ranges 19.1–23.9 (uniform shortfall, not a few outlier views).

## Hypotheses (ranked)
1. **Sky-sphere size (review-1 a).** Supp. l.736: "for scenarios when we use 20K points and less, we reduce the number of
   sphere points to 2K". Iteration 0 used 20K SfM + 10K sky; if "points" refers to the SfM points (our 20K), the
   paper would use a 2K sphere. Ambiguous text, but consistent with the evidence above (21 % of the budget on a
   nearly useless sphere). Expected effect: +0.3…1 dB (capacity-limited regime). → iteration 1.
2. **Data/protocol (b).** Ours: 3DGS-distributed T&T truck (979×546, 3DGS COLMAP). Authors: their own COLMAP
   (`Truck_COLMAP_big`, convert.py), images "reduced ×2 (~900 px)" — likely 960×540 with different poses/SfM.
   Cannot be closed (authors' data unavailable); different SfM also changes FPS init and the sky radius.
   Unknown sign; plausibly part of the gap.
3. **xyz lr after 30000 (c).** Official keeps xyz at lr_final for 30002–31999 (only 30001 is zero). Tiny effect.

## Action taken (iteration 1, post-hoc)
New entry point `bbsplat-tnt-sky2k`: cap 30K, `max_read_points` 28K (cap − sky split), `sky_points` 2000,
`xyz_lr_after_freeze: official`. Labelled post-hoc (chosen after seeing truck). Main methods unchanged in
behaviour (new config key defaults to "zero"). verify passes (7/7), 7 unit tests, ruff clean, 600-step
truck smoke OK (2000 sky, 30000 billboards).
Note: editing density.py changes the source manifest/spec ids, so iteration-0 runs can no longer be
re-evaluated with ss-eval (their test.json results stand).
