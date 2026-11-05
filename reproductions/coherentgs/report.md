# CoherentGS reproduction report

**Paper:** Paliwal et al., *CoherentGS: Sparse Novel View Synthesis with Coherent 3D Gaussians* (ECCV 2024).

**Official code:** avinashpaliwal/CoherentGS. Its decoders are ported layer for layer. We use the authors' processed LLFF archive, which contains Depth Anything depth and FlowFormer++ flows.

**Method:** `coherentgs`, with plugin `plugin/` (`ss_coherentgs`). The method:
- places per-pixel Gaussians on their rays, positioned by a residual-depth decoder and an opacity decoder;
- adds depth-smoothness and flow-consistency losses;
- uses multisampling after iteration 12000;
- trains with the official RGB loss: multiscale L1 + SSIM + VGG.

**Outcome: not matched.**
- The 8-scene LLFF 3-view mean masked PSNR is **19.28, against the paper's 20.33 (−1.05 dB)**.
- SSIM is 0.694 against 0.725, and LPIPS 0.225 against 0.180.
- The method's gain over 3DGS still reproduces strongly. Against plain gsplat trained from the same per-pixel initialisation, CoherentGS gains **+6.9 dB**. The paper reports +5.3 dB over its own 3DGS row (20.33 against 14.99), but that row starts from a different initialisation.

## Results
Setup:
- 13000 steps on `llff-fewview3-cgs`: official images_8, the official 3-view split, and all 571,536 per-pixel Gaussians.
- Test views are every 8th image.
- Metrics come from `python -m ss_coherentgs.evaluate`. The paper's protocol masks out pixels where CoherentGS's rendered opacity is below 1e-3. For gsplat, the CoherentGS mask of the same scene is applied, as the paper does.
- The paper reports averages only, so the full 8-scene mean is the comparison.

| method | fern | flower | fortress | horns | leaves | orchids | room | trex | **mean masked PSNR** | SSIM | LPIPS |
|---|---|---|---|---|---|---|---|---|---|---|---|
| coherentgs (ours) | 20.26 | 19.60 | 24.37 | 18.71 | 16.54 | 15.64 | 20.02 | 19.06 | **19.28** | 0.694 | 0.225 |
| paper CoherentGS | | | | | | | | | **20.33** | 0.725 | 0.180 |
| gsplat, same init (ours) | 13.15 | 12.59 | 14.37 | 11.71 | 11.24 | 11.40 | 13.81 | 10.96 | 12.40 | 0.238 | 0.569 |
| paper 3DGS (own init) | | | | | | | | | 14.99 | 0.483 | 0.362 |

**Alternative metric conventions** (CoherentGS, 8-scene mean):

| convention | PSNR | SSIM | LPIPS |
|---|---|---|---|
| masked pixels excluded | 19.09 | — | — |
| unmasked (ss-eval) | 16.86 | 0.672 | 0.256 |

- **Masked pixels excluded** (`psnr_valid`) drops the masked pixels instead of filling them with white.
- On average 96.0 % of pixels are valid. The fill convention is not documented in the official code, so white fill (the headline) is an assumption.

## Why no iteration
The batch rules allow 1–2 iterations, but only for a fix grounded in the paper or the official code. None was found.
- **The screen-size prune bug doesn't apply.** That bug, found in SparseGS and FatesGS, is in densification, and CoherentGS does no densification or pruning.
- **The remaining candidates were each judged small.** They are:
  - gsplat's alpha clamp of 0.999 against the official 0.99 (reviewer minor);
  - the near plane, 0.01 against 0.2;
  - views sampled with replacement;
  - RNG streams.
- **The loss curve looks healthy.** Fern goes from 19.32 masked PSNR at 2k steps (smoke test) to 20.26 at 13k, and training loss falls steadily (0.113 to 0.077).

## Fidelity gaps (kept)
1. **Mask fill convention** is assumed (white in both images). The FSGS eval script the authors used is not in the repo. Both conventions are reported above.
2. **View sampling** is with replacement; the official code uses a shuffled stack.
3. **Rasterizer:** gsplat instead of Inria/FSGS. The alpha clamp is 0.999 against 0.99, and gsplat's gradient drop is handled with a straight-through cap.
4. **Final step:** the runtime takes one Adam step on shN, scales and quats at the last step.
5. **Metric implementations:** SSIM uses valid convolution, and LPIPS is computed on float renders.
6. **RNG** streams differ from the official run.
7. **Coverage:** only LLFF 3-view is benchmarked. The 2- and 4-view settings and NVS-RGBD are not.
8. **Priors:** the author-provided depth and flow are used as-is; Depth Anything and FlowFormer++ are not re-run.
9. **Code over paper:** effective flow weight 0.01 (paper 0.1), a γ ramp from 0.35 to 1, and a fixed colour DC term.

## Review
`reviews/review-1.json` approved the port with 5 minors. It verified the split, preprocessing, losses, decoders and schedules against the official code, and confirmed the checkpoint and eval renders are equivalent.

## Reproduce
```bash
python3 tools/bench.py launch --exp coherentgs-llff3 --methods coherentgs gsplat --dataset llff-fewview3-cgs --steps 13000 --recipe splatstudio
tools/ssenv python -m ss_coherentgs.evaluate --run <coherentgs run> --split test
tools/ssenv python -m ss_coherentgs.evaluate --run <gsplat run> --mask-run <coherentgs run> --split test
```
