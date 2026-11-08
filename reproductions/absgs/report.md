# AbsGS — reproduction report

**Outcome: matched** (both benchmarked scenes within ±0.5 dB of the paper's per-scene PSNR).

Paper: AbsGS: Recovering Fine Details for 3D Gaussian Splatting (ACM MM 2024, arXiv 2404.10484).
Official code: github.com/TY424/AbsGS (read-only in `source/code/`; densification ported from
`scene/gaussian_model.py`). Plugin: `plugin/` (`splatstudio-absgs`), method `absgs`.
Benchmark: `runs/absgs-mipnerf360-30000`, 30k steps, `--recipe 3dgs`, Mip-NeRF 360 `images_4` /
`images_2` (the official `full_eval.py -i` protocol, not `-pilr`), llffhold8, seed 0.

## Results (PSNR, test split)

| scene | ours `absgs` | paper AbsGS-0004 | Δ vs paper | our gsplat baseline | paper 3D-GS | paper 3D-GS* (τ_S=0.001) | Gaussians | train time |
|---|---|---|---|---|---|---|---|---|
| garden (outdoor) | 27.12 | 27.487 | −0.36 | 26.89 | 27.41 | 27.11 | 3.36M | 47 min |
| kitchen (indoor) | 31.56 | 31.621 | −0.07 | 31.34 | 30.317 | 31.312 | 1.03M | 39 min |

Baseline: `runs/baseline3dgs-mipnerf360-30000` (gsplat, 3DGS recipe, opacity-reset fix).

**Gain over baseline:** ours +0.23 (garden) / +0.22 (kitchen). The paper reports +0.08 / +1.30
over its 3D-GS row. That kitchen row (30.317) comes from the old Inria release. The authors' own
retrain with τ_S=0.001 (3D-GS*) already gets 31.31, so about 1.0 dB of the paper's kitchen gain
comes from τ_S alone. Our gsplat baseline already reaches 31.34 on kitchen, so the paper's
headline +1.30 dB is not expected over it. Against 3D-GS*, the paper's gains are +0.38 (garden)
and +0.31 (kitchen), close to ours. Our garden baseline is 0.52 dB below the 3DGS paper, and
this offset carries into the absolute AbsGS number (−0.36).

Only 2 of the 13 reported scenes were run (batch rule). No mean is reported, and no
hyper-parameters were tuned.

## What is implemented
- C1 homodirectional gradient: gsplat `absgrad=True` sums |∂L_j/∂μ| per pixel, which equals the
  official `backward.cu:549-550`. A CUDA probe checks it against independent per-pixel autograd.
- C2 decoupled densification: clone if the signed gradient is ≥ 0.0002 and the Gaussian is small;
  split if the absolute gradient is ≥ 0.0004 and the Gaussian is large; τ_S = 0.001. Clones are
  never split in the same round. A custom density component is used, because the built-in
  `gsplat-absgrad` uses the absolute gradient for both clone and split, with threshold 0.0008 and
  τ_S 0.01.
- 3DGS defaults for everything else: L1 + 0.2 D-SSIM, densify 500–15000 every 100, opacity reset
  every 3000, prune at opacity < 0.005 and world size > 0.1·extent after 3000.
- No custom CUDA.

## Deviations / fidelity notes
1. **Opacity reduction on (review-1 fix).** Before the benchmark, the official `use_reduce=True`
   default was enabled: every 500 steps before 15k, opacity → min(o, 0.8) and Adam moments are
   zeroed (`arguments:92-93`, `train.py:132`, `gaussian_model.py:260`). It is on in the release
   and in `full_eval.py`, and it is not in the README's "not used in our paper" list. The paper
   never mentions it, so whether the table used it is unknown. `absgs-noreduce` exists but was
   not benchmarked. In 2k-step smoke runs it changed garden PSNR by +0.15 (23.89 vs 23.74).
2. **Extras not used** (the official README says the paper did not use them): initial SfM point
   filtering (err > 2, track < 3) plus init_prune, and weight-based pruning. The released loader
   filters SfM points whenever `points3D.ply` is absent. We keep all points (garden init 138,766).
3. Runtime protocol differences shared with the baseline: white background (adds an alpha term to
   dL/dμ2D in both accumulators; the official code uses black), random view sampling with
   replacement, SSIM with valid convolution, LPIPS-VGG from the `lpips` package, and densification
   after `optimizer.step()` (official: before).
4. Screen-size pruning is inactive, as in the official code (its `max_radii2D` bug).
5. "Clones never split in the same round" is implemented but not probed: clone and split sets are
   disjoint by scale, so removing it only raises a shape error.
6. One seed and 2 scenes. The garden gap (−0.36) is within tolerance but in line with our
   baseline's offset on garden.

## Evidence
`evidence/verify.json` (all probes passed), `evidence/test.json`, smoke runs
`evidence/smoke-1` (no reduction) and `evidence/smoke-2` (with reduction), `evidence/smoke-baseline`,
`reviews/review-1.md`, `bench_request.json`.
