# 2DGS — reproduction report

**Outcome: matched** (within ±0.5 dB on both reported scenes). Paper: *2D Gaussian Splatting for Geometrically
Accurate Radiance Fields* (SIGGRAPH 2024, arXiv 2403.17888). Method `2dgs` (plugin `plugin/ss_twodgs`),
official code hbb1/2d-gaussian-splatting @f3e3b9f with the surfel rasteriser hbb1/diff-surfel-rasterization
@e0ed020 vendored unmodified (Inria licence, `csrc/SHA256SUMS`). Benchmark: `runs/2dgs-mipnerf360`.

## Results (Mip-NeRF 360, 30k steps, `mipnerf360` protocol = images_4 / images_2, llffhold 8)
| scene | ours PSNR | paper PSNR | Δ | ours SSIM / LPIPS | paper SSIM / LPIPS | Gaussians | min |
|---|---|---|---|---|---|---|---|
| garden (outdoor) | 26.73 | 26.95 | −0.22 | 0.844 / 0.165 | 0.852 / 0.115 | 2.66M | 68 |
| counter (indoor) | 28.11 | 28.55 | −0.44 | 0.894 / 0.295 | 0.900 / 0.208 | 657k | 98 |

Relative to 3DGS: our gsplat baseline (`baseline3dgs-mipnerf360-30000`) gives 26.89 / 28.85, so 2DGS is
−0.16 (garden) / −0.74 (counter). The paper's own table gives 2DGS −0.46 / −0.15 against its 3DGS row
(27.41 / 28.70). The per-scene gap to 3DGS therefore does not reproduce scene by scene. In both cases 2DGS
comes out slightly below 3DGS, but the indoor drop is larger here. Two scenes are not a dataset average: no
outdoor/indoor/9-scene mean is claimed. LPIPS is consistently worse than the paper (also true of our
baseline vs 3DGS; known calibration offset of the runtime's `lpips` VGG and float renders).

## Verification
`ss-inspect` no issues; 5 unit tests; `ss-verify` declared_checks_passed with 9 probes. The kernel matches an
independent ray–plane evaluation (alpha/depth/normal errors ≤ 5e-6). The low-pass filter is detected. The
distortion is exactly w1 w2 (m1−m2)² in NDC. The densification statistic equals the projected 3D-centre
gradient (rel. err 9e-8). The density recipe was checked: in-plane split, cull 0.05, reset 0.01, trainable
leaves. Smoke (garden 2k): 23.04 vs baseline 22.99. An early-regulariser smoke showed both losses active and
finite. No separate review stage was run, since the official rasteriser is vendored unmodified.

## Deviations from the paper / official code
1. **Distortion weight 0** (released code default; `scripts/m360_eval.py` passes no override). Paper §5
   states α = 100 for unbounded scenes. Under the batch rule "code overrides paper" the benchmark uses 0, so
   the distortion loss is inactive in these runs. `2dgs-paperdist` (λ_dist 100) exists, but it was not benchmarked.
2. **Released code ≠ paper code**: the official README states that the re-implementation removed the
   low-pass-filter gradient from densification (fewer Gaussians, "PSNR slightly dropped"). The paper-version
   code is not available, so a small negative offset is expected.
3. Runtime order: optimiser step before densification (official: densify, then step).
4. Random view sampling with replacement (official: shuffled stack); SSIM with valid convolution; SH degree
   increases one step later than official.
5. Metrics computed on float renders (official metrics.py reads 8-bit PNGs).
6. The paper's normal is from median-depth points; the code (and this port) uses expected depth
   (depth_ratio 0) for unbounded scenes. The code is followed.
7. **No mesh extraction / geometry evaluation** (TSDF, DTU Chamfer, TnT F1) — the paper's main geometry claims
   are not tested; only NVS PSNR is reproduced.
8. Implementation note: the official low-pass filter is centred on the bounding-box centre of the projected
   3σ ellipse (compute_aabb), not on the projected mean as written in the paper. This port keeps the official
   behaviour.
