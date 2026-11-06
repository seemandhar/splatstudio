# Gradient-Aware Upscaling (#80): build report

**Status:** the plugin is built and verified. Nothing has been benchmarked yet; see `bench_request.json`.

The official repo has no code (README only), and the authors' brush fork contains only WGSL viewer shaders. The CUDA kernel was therefore written from scratch:
- the forward pass computes derivative jets (I, I_x, I_y, I_xy) on top of gsplat's projection and tile lists;
- the backward pass is reverse mode with the inversion trick.

The upscaler is a separable Hermite spline, which is identical to the paper's A = C⁻¹F C⁻ᵀ.

## Methods
| method | what it is |
|---|---|
| `gradient-upscaling-{2x,4x}` | the paper's method (train with upscaling) |
| `bicubic-upscaling-{2x,4x}` | PyTorch bicubic upscaling in the training loop |
| `gradient-upscaling-3dgs` | 3DGS control: train with `--downscale s`, evaluate at full res with `plugin/apply_upscaling.py` |

All five use Taming densification to 1 M Gaussians, as the paper does.

## Probes (all pass)
- **Forward value:** matches gsplat to 1.2e-7.
- **Forward and backward vs an independent float64 autograd reference:** ≤2.5e-7.
- **Analytic gradients vs finite differences:** median error 2e-5.
- **Spline:** equals the paper's matrix formula (4e-15), and equals Catmull-Rom when given finite-difference gradients.
- **Ablation:** analytic gradients beat finite differences and bicubic by 1.5–2.5 dB on synthetic scenes.
- **End to end:** the opacity gradient matches a finite difference to 2e-4.

## Smoke test (truck, 2000 steps, test set at full 979×546)
| run | PSNR | SSIM | LPIPS |
|---|---|---|---|
| spline 4× | 20.65 | 0.693 | 0.431 |
| bicubic 4× | 20.48 | 0.662 | 0.459 |
| gsplat at full res | 22.05 | 0.796 | 0.310 |

The full-res gsplat run is not a row in the paper. On the control trained at 4× downscale (8 views, via `apply_upscaling`):

| control rendering | PSNR |
|---|---|
| native full res | 19.55 |
| spline | 21.42 |
| spline with finite differences | 21.21 |
| bicubic | 21.21 |

## Speed
At 979×546 with about 300k Gaussians, rendering at low res plus upscaling is **slower** than native rendering on the A6000: 8.1 ms vs 5.4 ms. At this size, fixed per-frame costs dominate. The paper's 3–5× speedup is measured at 5187×3361 on an L4 GPU, so I expect a gap here.

The jet rasterizer adds 4–8% over gsplat's forward pass.

## Fidelity gaps
1. There is no official code, so the kernels are derived from the paper.
2. Rasterization uses gsplat's numerics (0.999 alpha clamp, with zero derivative where clamped).
3. The background is white, and its term is included in the derivatives.
4. The upscaler extrapolates the edge cell at the border, whereas brush mirrors the fraction.
5. Only Taming's densification is used, with interval 100 (an assumption).
6. Mip-NeRF360 at full res (5187×3361, 5 M Gaussians) is not benchmarked.
7. The control uses LANCZOS downscaling of the ground truth.
8. The upscaler is PyTorch ops, and timings are same-GPU ratios.
9. Drjohnson is natively 1332×876; the paper's table header says 1264×832.

Per-scene paper numbers for 2×/4×/8× are in `paper_results.json`.

## Review-1 resolution (orchestrator)
Approved, with no critical or major issues. The reviewer stress-tested the kernel independently:
- opacity up to 1.0 and 30–60 overlapping Gaussians;
- terminating transmittance;
- result: ≤4.3e-7 relative error against a float64 reference.

**Benchmark (reduced per the batch scene rule):** 4× only, on tandt/truck and db/playroom, 6 runs.
- Methods: `gradient-upscaling-4x`, `bicubic-upscaling-4x`, and `gradient-upscaling-3dgs` trained with `--downscale 4`.
- The control is scored by `apply_upscaling.py --factor 4` (`quality.native` and `4x.spline`).
- Experiments: `gup4x-{tandt,db}` and `gup4x-ctl-{tandt,db}`.

**Paper targets at 4×:**

| scene | spline | bicubic | 3DGS | ΔPSNR spline − 3DGS | ΔPSNR spline − bicubic |
|---|---|---|---|---|---|
| truck | 23.00 | 22.10 | 21.68 | +1.32 | +0.91 |
| playroom | 29.42 | 28.96 | 29.18 | +0.24 | +0.47 |

The comparison is mainly on these per-scene deltas.

**Not claimed:** the speed result (3–5× at 5187×3361 on an L4) cannot be tested at T&T/DB resolution.

**Gaps added:**
- **Taming score resolution:** the methods score at full resolution, the control at low resolution.
- **Border cells:** extrapolated here, where brush mirrors them.
- **Low-res size:** floor here, where Inria rounds.

## Benchmark: truck at 4× (gup4x-tandt + gup4x-ctl-tandt), interim
| row | ours | paper | Δ |
|---|---|---|---|
| spline 4× (`gradient-upscaling-4x`) | 22.82 / 0.767 / 0.339 | 23.00 | −0.18 |
| bicubic 4× (`bicubic-upscaling-4x`) | 22.23 / 0.708 / 0.402 | 22.10 | +0.13 |
| spline − bicubic | +0.59 | +0.91 | |
| control (3DGS at 4× downscale) + spline (`apply_upscaling`) | 22.81 | — | |
| control + bicubic | 22.43 | — | |
| control, native render at full res | 18.75 | 21.68 ("3DGS") | −2.93 |
| control, scored at its training res (ss-eval) | 27.19 | — | |

- Both method rows match within ±0.5 dB.
- The paper's "3DGS" row (21.68) is not reproduced by either control reading.
  - Rendering the low-res model natively at full res gives 18.75.
  - Scoring at low res gives 27.19.
  - The paper's exact rendering for that row is unclear: there is no official code, so it is unresolved. It is a single-scene control discrepancy and does not affect the method rows.
- Playroom (gup4x-db, gup4x-ctl-db) is still pending.

### Playroom control (gup4x-ctl-db), 4×
- **Control (3DGS at 4× downscale):** spline 28.68 / bicubic 28.51 / native at full res 26.74 / scored at its own low resolution 29.84.
- **Paper:** spline 29.42, bicubic 28.96, "3DGS" 29.18.
- **Still pending:** the method runs (gup4x-db: gradient-upscaling-4x and bicubic-upscaling-4x).
