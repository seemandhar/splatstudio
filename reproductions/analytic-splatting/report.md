# Analytic-Splatting (#70, arXiv 2403.11056) — build report (pre-benchmark)

**Status:** ready for benchmark. Method `analytic-splatting` (plugin `plugin/`, module `ss_analytic_splatting`).

## Implementation
- Official CUDA rasterizer (lzhnb/Analytic-Splatting @ a905939, `submodules/diff-gaussian-rasterization`)
  vendored **unmodified** in `plugin/ss_analytic_splatting/csrc/` (Inria licence + repo MIT), JIT-built via tools/ssenv.
- C1 pixel-area integral (Eq. 10–13) + C2 conditioned logistic CDF (Eq. 7) live in the kernel's forward/backward.
- Density = official train.py ADC (`dense=False`, no screen-size prune — official no-op), opacity reset with the
  Mip-Splatting filter coefficient `min(o, 0.01/coef)` (official quirk kept, filter not rendered). Loss = L1 + 0.2 D-SSIM.

## Verification (evidence/verify.json, test.json)
| probe | result |
|---|---|
| forward vs independent Eq. 13 (float64) | max rel 3e-7 |
| forward vs pixel quadrature, σ≈0.4 px | kernel 2.5 % vs undilated centre sample 42 % (no 0.3 px² dilation, so not 3DGS's actual error) |
| backward vs float64 autograd (means, scales, quats, opacity, sh0, means2D) | ≤ 3e-6 |
| backward vs central finite differences | ≤ 1.5e-4 |
| logistic CDF | max abs 3.9e-4 (paper Fig. 4 ~4e-4); cubic term essential |
| density schedule / reset | pass |

Smoke (garden, 2000 steps, recipe 3dgs): **23.35 dB** / 862k Gaussians vs gsplat baseline 23.02 / 845k
(Inria-rasterizer ADC control from the SteepGS port: 23.23). No NaNs, ~4 min.

## Benchmark plan
Paper reports per-scene STST (supp. Tab. single_360, `images_4`/`images_2` = our native `mipnerf360`):
garden 27.39, counter 29.11 (3DGS row 27.41 / 28.70). Compare to `baseline3dgs-mipnerf360-30000`
(gsplat 26.89 / 28.85). Optional second protocol `mipnerf360-pilr` (Ours* 27.73 / 29.24 vs authors' 3DGS* 27.70 / 29.08).

## Fidelity gaps
1. Headline multi-scale (MTMT Blender / 360) protocol not benchmarked: runtime lacks multi-resolution sampling;
   anti-aliasing under footprint change is therefore validated only by the probes, not by PSNR.
2. Densify/reset after optimizer step; random view sampling with replacement; runtime SSIM/LPIPS differences.
3. Log-space reset coefficient (avoids official fp32 0/0). 4. "Ours + 3D filter" not ported.
