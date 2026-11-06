# RobustSplat (plan #87) — build report (pre-benchmark)

**Method**: `robustsplat` (full model), with ablation variants `robustsplat-nodg` (3DGS+Mask+MB) and `robustsplat-nomb` (3DGS+Mask+DG). The plugin is in `plugin/` (`ss_robustsplat`). It is a port of the official code, github.com/fcyycf/RobustSplat @ a130281 (Inria-3DGS based, read only, never run). Where the code and paper differ, the code values win; paper.md §7 lists the differences.

## Contributions → components
| contribution | component | probe |
|---|---|---|
| C1 Delayed Gaussian growth: densify 10000<t<20000, opacity reset only 15000/18000, no size pruning, opacity lr 0.025 | `rs-density` (gsplat ops, Inria numerics) | delayed-growth (growth/reset step sets vs paper schedule, huge Gaussian not size-pruned) |
| C2 Scale-cascaded mask bootstrapping: t<20000 residual from images_{4f} render vs GT, DINOv2 16x16 (224²); then train-res residual, 36x36 (504²) | `rs-mask-mlp` | bootstrap-scale (grid/residual-size switch, coarse mask == independent coarse render) |
| Mask MLP (DINOv2 384-16-1) gating L1+D-SSIM, mask>0.25 + 7x7 erosion; residual hinge τ .6/.8 + feature-cosine supervision | `rs-masked-photometric`, `rs-mask-mlp` | masked-photometric, mask-supervision, dino-parity (vs timm, max err 7e-5) |
| Early reg λ_reg·e^{-t/2000}·mean(1−M); MLP paused 300 steps after resets | `rs-mask-mlp` | mlp-schedule, mask-supervision |

No custom CUDA (stock gsplat).

## Data
The authors released their own COLMAP-rerun, undistorted data (hf.co/datasets/fcy99/RobustSplat-data). Its protocol differs from the existing `onthego`/`onthego-undist`/`robustnerf` sets: SfM, undistortion and normalisation are all different, and patio is ×4. So I downloaded it instead of reusing those sets.
- **Scenes downloaded:** patio, patio-high, android, crab2. Total ≈ 5.4 GB under `/bigdata/sejain/datasets/robustsplat`, symlinked from `~/datasets/robustsplat`.
- **Conversion:** `tools/prepare_robustsplat.py` builds the native scenes:
  - images are kept as-is, with the Inria FoV camera (centred principal point) and the COLMAP world frame;
  - splits: train = clutter, test = extra, val = copies of test;
  - `coarse/` holds the images_{4f} views.
- **Registry:** the scenes are registered in `tools/datasets.json` as `robustsplat-onthego` and `robustsplat-robustnerf`.

## Tests
- `ss-inspect` reports no issues for all 3 methods.
- pytest: 15 passed; ruff is clean.
- `ss-verify` returns `declared_checks_passed` for all 3 methods (9 probes each).
- Smoke run on patio-high, 2000 steps:

| run | PSNR | SSIM | LPIPS | Gaussians | time |
|---|---|---|---|---|---|
| robustsplat | 20.996 | 0.696 | 0.329 | 137k (no densification before 10k, as designed) | 116 s |
| gsplat | 20.628 | 0.699 | 0.313 | 534k | 131 s |

Both loss terms are live from step 500: `rs-mask-mlp` is 0.43 at step 1000 and 0.39 at step 2000.

The 2000-step smoke run does not reach the delayed densification (10100) or the fine-supervision switch (20000). Only the synthetic pipeline probe covers those steps; real-data behaviour there will first show up in the benchmark.

## Benchmark request
patio-high (paper 22.87; its 3DGS row 17.04) and crab2 (paper 34.88; 3DGS 31.76), 30k steps, with `gsplat` on the same scenes as a protocol baseline.

## Fidelity gaps
- The runtime steps the optimiser before densifying; the official code densifies first.
- Views are sampled with replacement.
- The SH degree is raised one step late.
- Evaluation uses the runtime's SSIM (valid conv) and the `lpips` VGG, not the official `lpipsPyTorch` VGG.
- `get_regularizer` is omitted. In the official code it is a constant with no gradient.
- The residual histogram uses `torch.histc` on GPU instead of `torch.histogram`.
- Bootstrapping switches at 20000 (code). The paper text implies 10000.
- Single seed.

## Review-1 resolution (orchestrator)
Approved, with 0 critical, 0 major and 6 minor issues.
- **Post-10k/20k phases on real data:** the reviewer ran patio-high for 2600 steps with compressed thresholds. Densification, the opacity resets, the coarse→fine bootstrapping switch and the pause windows all executed correctly (val PSNR 21.17).
- **Released checkpoints:** they confirm the data factors (patio ×4, patio-high ×8, crab2 ×8) and the clutter/extra splits. crab2 scores 34.876, the paper's number exactly, which supports the code's bootstrap_iter = 20000.
- **Target caveat:** the patio-high checkpoint comes from a different experiment path, so its target may come from a slightly different code state.
- **Released Gaussian counts** for comparison: about 1.03M on patio-high, 0.55M on patio.
- **Minor:**
  - torch.histc on GPU instead of torch.histogram on CPU; negligible.
  - The robustsplat-nodg ablation keeps bootstrap at 20000, which is an interpretation.
  - Eval PSNR is computed on float renders, not the official 8-bit PNGs.
- **Benchmark:** robustsplat-onthego (patio-high) and robustsplat-robustnerf (crab2), robustsplat and gsplat at 30k. Compare each gain over gsplat with the paper's gain over its 3DGS row (17.04 / 31.76).
