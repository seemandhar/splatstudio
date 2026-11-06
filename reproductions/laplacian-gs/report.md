# Laplacian-GS (#83) — build report (benchmark pending)

- Slug `laplacian-gs`, method `laplacian-gs` (plus `laplacian-gs-smoke` for smoke tests only), plugin in `plugin/`.
- Ported from the official code (SorenZhang574/Laplacian-GS, MIT, commit 33af46d), `--backbone taming` path:
  2 levels, 10K/20K split, xyz-only inheritance ρ=1, reconstructed-image supervision.
- Mapping: `lap_level` field per Gaussian; renderer = base field at pyramid level 1 (clamp [0,1]) + signed
  residual field (no +0.5, clamp [−1,1]) composed by Up = bilinear+binomial blur (Eq. 7); during training only
  the active level gets gradients, the archived level is the cache. Density = stage-local 3DGS ADC
  (prune-first, active field only), per-stage LR restart, archive + fresh Adam at the stage boundary.
  Loss = L1 + 0.2 D-SSIM(same) vs Down^ℓ(GT). No custom CUDA (the signed rasterizer is gsplat with precomputed SH colours).
- Verify: declared_checks_passed (9 probes). Unit tests 6/6, ruff clean.
- Smoke (garden w1600, 2000 steps, 1000/1000 plan): method 22.86 dB / 0.589 / 0.471, 457K Gaussians;
  gsplat 22.95 / 0.626 / 0.439, 778K.
- Benchmark request: mipnerf360-1600 garden + counter, 30K steps. Paper Ours(Taming): 27.35 / 29.22
  (paper 3DGS 27.36 / 29.17). Our baseline: 26.93 / 29.04.

## Fidelity gaps
1. gsplat + dense Adam instead of Taming's fast backward + SparseGaussianAdam.
2. Archived field stays on the GPU and its cache is re-rendered every step, so the paper's speed and memory claims are not reproduced (quality is the same because the archived field is frozen).
3. White base background (runtime convention); the code uses black.
4. Views sampled at random with replacement.
5. The optimizer still steps on the last iteration of each stage (the official code skips it).
6. New-field scales use exact cKDTree 3-NN instead of distCUDA2.
7. FastGS backbone (Ours(FastGS)) not ported.

## Review resolution (review-1, approved)
- Base background switched to black to match the official code (minor 2).
- Table refs: the per-scene tables are supp. Tables 10/11/12, not 9 (minor 1).
- The delta vs our gsplat baseline also includes LR differences (opacity 0.025 vs 0.05, shN 2.5e-4 vs 1.25e-4). The paper itself shows ~0 dB vs 3DGS and -0.13/-0.21 dB vs its Taming backbone (minor 3).
- --steps must equal 30000 (minor 4, unchecked in code).

## Benchmark result (laplacian-gs-1600, 30k, mipnerf360-1600) — MATCHED
| scene | ours | paper Ours(Taming) | Δ | gsplat baseline | paper 3DGS / Taming | gain ours / paper vs 3DGS |
|---|---|---|---|---|---|---|
| garden | 27.36 / 0.853 / 0.156 | 27.35 | +0.01 | 26.93 | 27.36 / 27.48 | +0.43 / −0.01 |
| counter | 29.29 / 0.911 / 0.258 | 29.22 | +0.07 | 29.04 | 29.17 / 29.43 | +0.25 / +0.05 |
- Both scenes are within ±0.5 dB.
- Our gain over our baseline (+0.43 / +0.25) is larger than the paper's (≈0 vs its 3DGS; −0.13 / −0.21 vs its Taming backbone). Part of that comes from the LR differences against our baseline (opacity 0.025 vs 0.05, shN 2.5e-4 vs 1.25e-4), not only from the pyramid.
- Final Gaussians, archived base level plus residual level: garden 4.34M, counter 1.11M.
