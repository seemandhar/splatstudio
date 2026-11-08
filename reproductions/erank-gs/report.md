# erank-gs — reproduction report (plan id 130)

**Paper:** Effective Rank Analysis and Regularization for Enhanced 3D Gaussian Splatting (NeurIPS 2024, arXiv 2406.11672).
**Official code:** github.com/junhahyung/erank_gs @99fd128 (WIP release), used for every numeric choice.
**Outcome: NOT MATCHED on the DTU test split (held-out every-8th view), which is the split that matches the paper's protocol.** The method is 0.65 and 0.85 dB below the paper. The control 3DGS run under the same protocol is 0.18 and 0.51 dB below the paper's 3DGS row.

## Method
`erank-gs` = paper "+a+b" / "3DGS+e": vanilla 3DGS (gsplat classic, 0.3 px dilation, black background) + GOF-style fixed ADC (quantile-matched abs gradient, resampled clones, min opacity 0.05) + erank loss in the code's form: 0.01·mean(clamp(−log(erank−1+1e-7), 0)) + 1·mean(s_min), active from step 7000.
Controls: `erank-gs-3dgs` (paper 3DGS row: Inria ADC, prune 0.005) and `erank-gs-adc` ("+a" ablation, not yet benchmarked). No custom CUDA. Probes: erank formula/schedule, abs-gradient units vs a per-pixel reference (CUDA), ADC abs criterion, density recipe. `ss-verify` passes for all three.

## Data / protocol
2DGS DTU release (`dtu-2dgs`, built by `tools/prepare_dtu_2dgs.py`): `-r 2` gives 777×581. This is the official loader's actual output for the paper's stated "800×600", so the resolution is not a gap. Images are RGB·alpha-mask on black, black background, init from COLMAP points, llffhold8 split. 30k steps, `--recipe 3dgs`.

## Results (runs/erank-gs-dtu, PSNR)
| scene | split | erank-gs | paper +a+b | Δ | erank-gs-3dgs | paper 3DGS | Δ | ours: method − control | paper: method − control |
|---|---|---|---|---|---|---|---|---|---|
| scan24 | **test** | 30.25 | 30.90 | **−0.65** | 30.27 | 30.45 | −0.18 | −0.01 | +0.45 |
| scan122 | **test** | 37.38 | 38.23 | **−0.85** | 37.56 | 38.07 | −0.51 | −0.18 | +0.16 |
| scan24 | train | 36.28 | — | +5.4 | 37.69 | — | +7.2 | −1.41 | |
| scan122 | train | 42.99 | — | +4.8 | 43.65 | — | +5.6 | −0.67 | |

**Which split the paper used (review-1 minor 1):** train-view PSNR is 5–7 dB above every paper number, for both methods. Test-view PSNR is within 0.2–0.5 dB for the control. So the paper's DTU PSNR is held-out-view PSNR (it would need `--eval`, which `train.sh` lacks; `train.sh` is the run used for Chamfer distance). The test split is the comparable split.

## Gap analysis
- **The regularizer works as the paper describes.** Needles (erank < 1.04) at 30k: control 373,117 → erank-gs 321 on scan24; 48,368 → 0 on scan122. Mean erank goes 1.19 → 1.99 (paper Table 'iteration', scan37: 16,320 → 23). The final barrier term is 2.7e-4 vs 4.4e-2 for the control, so the loss is active and converged; its magnitude is not the issue.
- **Gaussian count is not what the paper reports.** We get 0.44× / 0.53× the control's count (309k vs 696k; 99k vs 187k). The paper reports about 10% fewer (DTU storage 113 → 98 MB, 15-scan mean). The deficit appears before the erank loss starts (step 2900: 234k vs 348k), so it comes from the "+a" densification. The 2k-step diagnostics put most of it on the official min-opacity-0.05 prune (count 158k → 246k with 0.005), on masked DTU where background Gaussians are driven to zero opacity, combined with the 3000-step reset to 0.01. This is the official recipe, so it was kept.
- **The control itself is far more needle-heavy than the paper's 3DGS.** 54% of Gaussians are needles vs about 4% implied by the paper's 16k on scan37. The backbone (gsplat vs the Inria rasterizer, runtime sampling with replacement, optimizer step before densification) may form needles differently; these are known runtime gaps.
- **No official-code-grounded fix is expected to close ≥0.5 dB.** The only unported official item is the opacity-mask BCE loss in `train.py`. It crashes as released (`if image_mask:` on a tensor) and is not in the paper, so adding it would be speculative. No iteration-1 launch.
- **Recommended next step (cheap diagnostic):** benchmark `erank-gs-adc` on scan24/122 (paper +a: 30.69 / 38.21). This tells whether the missing gain sits in the ADC (+a) or in the erank term (+b).

## Fidelity gaps
1. The paper does not state the DTU split; test = llffhold8 is assumed. Train-view numbers rule out a train-view protocol.
2. The opacity-mask BCE loss of the WIP release is not ported.
3. Runtime: view sampling with replacement; optimizer step before densification; SH degree increase one step late; SSIM valid convolution; gsplat rasterizer.
4. Only NVS PSNR is measured; no mesh / Chamfer-distance evaluation.
5. 2 of 15 scans, compared per scene (no dataset average claimed).

Smoke runs (2k steps, scan24): erank-gs 25.19 vs control 26.68. Diagnostics in `evidence/` (smoke-diag-*). All evidence kept.
