# Opti3DGS reproduction report

Paper: *Optimized 3D Gaussian Splatting using Coarse-to-Fine Image Frequency Modulation*
(arXiv 2503.14475). Plan id 5. Plugin: `plugin/` (`ss_opti3dgs`), methods `opti3dgs`
(Table 1 "3DGS + Ours*") and `opti3dgs-filter` (Table 2 "Ours-Filtering", not benchmarked).

**Outcome: matched.** The T&T average PSNR is 23.46 against the paper's 23.15 (+0.31 dB, within ±0.5).
The Gaussian count drops by 41%, against the paper's 64%. This run has not had a separate review.

## Results — Tanks & Temples, 30k steps, llffhold8 (`runs/opti3dgs-tandt`)

| | truck PSNR | train PSNR | mean PSNR | mean SSIM | mean LPIPS | mean #G (M) |
|---|---|---|---|---|---|---|
| **opti3dgs (ours)** | 25.18 | 21.74 | **23.46** | 0.844 | 0.219 | **1.113** (truck 1.498, train 0.729) |
| gsplat baseline (`baseline3dgs-tandt-30000`) | 25.21 | 21.52 | 23.36 | 0.849 | 0.207 | 1.876 (truck 2.495, train 1.257) |
| paper 3DGS + Ours* | – | – | 23.15 | 0.823 | 0.217 | 0.658 |
| paper 3DGS | – | – | 23.69 | 0.844 | 0.178 | 1.825 |

Change from adding the method:

| | ΔPSNR | ΔSSIM | ΔLPIPS | Δ#G |
|---|---|---|---|---|
| ours vs our gsplat baseline | **+0.10** | −0.005 | +0.012 | **−41 %** (truck −40 %, train −42 %) |
| paper Ours* vs paper 3DGS | −0.54 | −0.021 | +0.039 | −64 % |

- **Peak training memory:** truck 2.35 GB vs 3.96 GB baseline (−41%); train 1.18 GB vs 2.03 GB (−42%). The paper reports up to −40%.
- **Training time:** truck 1717 s vs 3571 s, train 1803 s vs 5123 s. The GPUs are shared and the two runs used different GPUs, so these times are only a rough indication. The paper reports −20%.
- **Smoke test** (garden, 2k steps): 20.45 dB / 216k Gaussians vs gsplat 22.91 dB / 848k. At step 2k the method is still training on blurred targets.

### Interpretation
- **PSNR:** the absolute number matches the paper. The change differs in sign: we gain +0.10 dB where the paper loses 0.54 dB.
- **Gaussian count:** the reduction is real but smaller than the paper's (−41% vs −64%). LPIPS gets worse, as in the paper.
- **Likely cause:** our port keeps more Gaussians than the paper's Ours* run. That would explain both the smaller reduction and the better PSNR. Candidate reasons are:
  - the densification mapping (gsplat's gradient normalisation and the `refine_every` mapping);
  - the white background.
- **Only T&T was benchmarked.** DB and the `opti3dgs-filter` variant were not run.

## Deviations from the paper
1. **No official code** has been released. Everything was implemented from the paper text and figures.
2. **Step boundaries:** 1000 / 4000 / 8000 / 12000 with kernels 15 / 9 / 5 / 3, read off the Fig. 3 "Step" curve. The text only states the initial kernel (15) and the stop step (12000).
3. **Padding of the mean filter** is unspecified in the paper. We use reflect padding (= `cv2.blur` default BORDER_REFLECT_101). This is configurable via `padding_mode`.
4. **`densification_interval=500` maps only to gsplat `refine_every`.** `refine_start` 500, `refine_stop` 15000 and the opacity reset every 3000 are unchanged. `densify_grad_threshold` 1e-4 maps to `grow_grad2d`, and `scaling_lr` 0.01 is applied to the scales optimizer.
5. **Runtime differences from Inria 3DGS:**
   - white background;
   - training views sampled at random with replacement;
   - valid-convolution SSIM in both the loss and the evaluation;
   - gsplat rasteriser;
   - LPIPS-VGG.
6. **The blur covers the whole loss** (L1 and D-SSIM), because the paper's filtered images replace the originals. Evaluation uses the original images.
7. **Baseline mismatch:** the paper's baseline is the authors' own 3DGS re-run on an H200. We compare against our shared gsplat baseline.

## Evidence
- `evidence/verify-opti3dgs.json` and `verify-opti3dgs-filter.json`: all probes pass (filter vs scipy reference, schedule boundaries, blurred loss target, density settings applied and fields still trainable).
- `evidence/test.json`
- `bench_request.json`
