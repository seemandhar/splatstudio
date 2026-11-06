# AdamW-GS (plan #93) — build report (benchmark launched)
- **Paper:** A Step to Decouple Optimization in 3DGS (arXiv 2601.16736).
- **Official code:** EliottDJay/3DGS_AdamWGS (MIT).
- **Methods:**
  - `adamw-gs-gs8`: the Table-2 headline outdoor row GS8 (noise + opacity reset every 3000 + StSS GS2 + world big prune). No released config exists for it.
  - `adamw-gs`: the released `mipnerf_adamwgsp` config, equal to row GS9.
  - `adamw-gs-indoor`: the released `mipnerf_adamwgs` config, equal to row GS7/GS8.
- **Targets:** outdoor GS8 plus indoor GS7/GS8. These per-scene rows average to Table 2 (27.678). GS9 is kept as a variant.

## Optimizer port
- `AdamWGSDensity.initialize` swaps each runtime optimizer in place for `SparseAdamWGS` (an Adam subclass with the same state keys). The install probe checks this.
- The update reproduces the adam.cu numerics: visible rows only, a per-primitive `asystep` for bias correction, and DAR outside the moments. It is float32 torch ops, not the fused kernel.
- `asystep` lives in the density state, so gsplat's duplicate, split and remove copy or drop it.

## Deviations and decisions
1. **α2 (RSR v-fading):** 0.01, the value the code effectively uses (it reads a misspelled key). The paper says 0.04. α1 = 0.2.
2. **img_scale:** the code's two-digit rule (bicycle/garden 100000, counter 160000). The paper text implies one digit.
3. **Step order:** the runtime steps before densifying. That gives 144 extra updates (600..14900). Under GS8, the reset iterations are also stepped. asystep counts are identical.
4. **Visibility:** gsplat culls opacity < 1/255, uses opacity-aware radii and near plane 0.01. diff-gauss has no cull, 3σ radii and near plane 0.2.
5. **GS8 big prune:** only the world test (0.1·extent) is active. The official 20 px screen test is dead code, because max_radii2D is re-zeroed before prune(). A live 20 px test collapsed training (7.3 dB at 3.2k steps); matching the official behaviour gives 20.6 dB.
6. **Runtime-wide:** white background, sampling with replacement, valid-convolution SSIM, means LR one step late.
7. **Not packaged:** T&T/DB configs and the GS5 StSS rows.

## Smoke tests (not diagnostic)
| run | PSNR | SSIM | LPIPS | Gaussians |
|---|---|---|---|---|
| garden adamw-gs, 2k | 22.52 | 0.608 | 0.445 | 511k |
| garden gsplat, 2k | 23.05 | 0.643 | 0.419 | 841k |
| counter adamw-gs-indoor, 2k | 24.39 | 0.824 | 0.395 | 282k |
| garden adamw-gs-gs8, 3.2k (val) | 20.56 | 0.623 | 0.426 | 756k |

These runs only show that training runs and stays finite. The LR schedule is compressed into the short budget while the DAR/RSR/noise windows stay absolute, and Sparse Adam trails dense Adam early (GS2 < GS1 in the paper).

## Review-1 resolution (orchestrator)
2 major (target rows) and 6 minor issues, all fixed; 27 tests pass and all 3 variants verify.
Benchmark (one launch per method), compared on the gain over baseline3dgs-mipnerf360-30000 (bicycle 24.921, counter 28.846):
- adamw-gs-gs8 on bicycle: primary, 25.454 (+0.216 vs GS1).
- adamw-gs on bicycle: secondary, GS9 25.481 (+0.243).
- adamw-gs-indoor on counter: 29.112 (+0.066).

## Result: counter (adamw-gs-indoor, 30k) — within tolerance
- **Result:** 29.08 / 0.907 / 0.268, with 577k Gaussians.
- **Paper GS7/GS8:** 29.112 / 0.907 / 0.206, with 535k Gaussians (Δ −0.03).
- **Gain:** +0.24 over our baseline (28.85), vs the paper's +0.07 over GS1.
- **Gaussian count:** 0.49× the baseline's 1.19M, vs the paper's 0.49× (535k / 1.09M).
- **Still pending:** bicycle (gs8, gs9).

## Result: bicycle with the released config (adamw-gs = GS9, 30k) — within tolerance
- **Result:** 25.40 / 0.778 / 0.247, with 3.47M Gaussians.
- **Paper GS9:** 25.481 (Δ −0.08).
- **Gain:** +0.48 over our baseline (24.92), vs the paper's +0.24 over GS1.
- **Still pending:** the primary headline variant adamw-gs-gs8 (target 25.454).

## Result: bicycle with the headline config (adamw-gs-gs8, 30k) — within tolerance
25.27 / 0.769 / 0.258, with 3.11M Gaussians, vs paper GS8 25.454 / 0.778 / 0.219, with 2.72M (Δ −0.18). The gain over our baseline is +0.35, vs the paper's +0.22.

## Verdict: MATCHED (all 3 runs within ±0.5 dB)
| run | ours | paper | Δ | gain ours / paper | Gaussians ours / paper |
|---|---|---|---|---|---|
| bicycle GS8 (primary) | 25.27 | 25.454 | −0.18 | +0.35 / +0.22 | 3.11M / 2.72M |
| bicycle GS9 (released config) | 25.40 | 25.481 | −0.08 | +0.48 / +0.24 | 3.47M / — |
| counter GS7/GS8 | 29.08 | 29.112 | −0.03 | +0.24 / +0.07 | 0.58M / 0.54M |

Our gains over our gsplat baseline are larger than the paper's gains over its GS1. That fits our baseline sitting below the paper's 3DGS (bicycle 24.92 vs 25.24).
