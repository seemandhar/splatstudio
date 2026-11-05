# Reduced-3DGS: reproduction report

**Paper:** Papantonakis et al., *Reducing the Memory Footprint of 3D Gaussian Splatting*, Proc. ACM CGIT (I3D) 2024.

**Official code:** `dc905c13`, read but not executed.

**Method:** `reduced-3dgs`, the `full_final` configuration. The plugin is in `plugin/` (`ss_reduced_3dgs`) and is frozen at `77657ca3739e801a`. It uses custom CUDA kernels for the redundancy score and the SH-band culling.

**Outcome: not matched (borderline).**
- counter is within tolerance: −0.44 dB.
- bicycle (−0.54 dB) and garden (−0.60 dB) miss ±0.5 dB by 0.04 and 0.10 dB.
- Sizes and Gaussian counts are at or below the paper's.
- The method loses about 0.3 dB more against our baseline than the paper does against its own.

## Results
Setup: 30k steps, `mipnerf360` (images_4 / images_2, which matches the official `full_eval.py`), llffhold 8, decoded quantized model.

Size is the paper's memory figure: the half-float plus 256-codebook PLY, from `plugin/report_size.py`.

| scene | PSNR ours / paper (Δ) | SSIM ours / paper | LPIPS ours / paper | #G ours / paper | MB ours / paper |
|---|---|---|---|---|---|
| bicycle | 24.52 / 25.06 (**−0.54**) | — / 0.761 | — / 0.221 | 2.29M / 2.41M | 41.9 / 48 |
| garden | 26.43 / 27.03 (**−0.60**) | — / 0.854 | — / 0.119 | 1.66M / 2.36M | 31.8 / 47 |
| counter | 28.10 / 28.54 (**−0.44**) | — / 0.898 | — / 0.213 | 0.44M / 0.50M | 8.6 / 11 |

The mean over the three scenes is SSIM 0.824 and LPIPS 0.239. That is a subset mean, not the paper's 9-scene average.

**Against our baseline.** Our baseline is `baseline3dgs-mipnerf360-30000` (gsplat 3DGS on the same images). The paper's numbers are against its own 3DGS row.

| scene | our baseline | ours − baseline | paper − its 3DGS |
|---|---|---|---|
| bicycle | 24.92 | −0.40 | −0.04 |
| garden | 26.89 | −0.46 | −0.25 |
| counter | 28.85 | −0.75 | −0.44 |

## Where the extra loss is likely coming from
Paper Table 1 (Mip-NeRF 360 average):

| step | PSNR change |
|---|---|
| point culling | +0.01 |
| SH culling | −0.25 |
| quantization | −0.08 |

Two observations point to SH culling and the base densification:

1. **SH culling is more aggressive than the paper's average.** The table gives the fraction of Gaussians assigned each SH degree at step 15k.

   | | degree 0 | degree 1 | degree 2 | degree 3 |
   |---|---|---|---|---|
   | bicycle | 0.971 | 0.000 | 0.007 | 0.022 |
   | garden | 0.963 | 0.000 | 0.008 | 0.029 |
   | counter | 0.954 | 0.001 | 0.012 | 0.033 |
   | paper (dataset average) | 0.89 | 0.001 | 0.027 | 0.082 |

   - The algorithm is the official one: two passes, std 0.04 and cdist 6·√3/255. The reviewer checked the official quirks.
   - Candidate causes: per-scene variation (the paper gives only an average), and the colour statistics using gsplat's transmittance numerics (gap 4).
   - Fewer higher-order SH bands would explain part of the uniform extra loss of about 0.3 dB.
2. **Our base 3DGS has fewer Gaussians than the official 3DGS.** On garden, our baseline has 4.76M Gaussians against the paper's 5.77M. Mercy pruning removes a similar fraction in both: ours keeps 0.35 of the baseline count, the paper 0.41. So the reduced model starts from a smaller set. This is a shared gsplat-runtime difference.

**No iteration.** Neither cause has a fix grounded in the paper that doesn't involve tuning. Raising the SH thresholds would be tuning on test numbers. Changing densification would change the shared runtime. Per the batch rules, the result stays as is.

## Fidelity gaps (plan.md)
1. Runtime order: the optimizer steps before density operations. The official code prunes and runs mercy before the step.
2. Baseline differences:
   - white background;
   - sampling with replacement;
   - no 20-px screen-size prune in gsplat's default.

   These also cause the lower base Gaussian count.
3. SH culling uses gsplat's transmittance, which is the same definition with different numerics.
4. After culling, the Adam moments of dropped coefficients are zeroed. The official code keeps them.
5. The official SH-cull quirks are replicated: the band loop never reaches q = 0, and pass 1 resets DC to the mean colour.

## Reproduce
```bash
python3 tools/bench.py launch --exp reduced-3dgs-mipnerf360 --methods reduced-3dgs --dataset mipnerf360 --scenes bicycle garden counter --steps 30000
tools/ssenv python reproductions/reduced-3dgs/plugin/report_size.py runs/reduced-3dgs-mipnerf360/reduced-3dgs/mipnerf360-*
```
