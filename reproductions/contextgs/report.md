# ContextGS — reproduction report

**Paper:** *ContextGS: Compact 3D Gaussian Splatting with Anchor Level Context Model* (NeurIPS 2024, arXiv 2405.20721).

**Official code:** `eff8d19`, read only and never executed.

**Method:** `contextgs`, plugin `plugin/` (`ss_contextgs`), frozen at `c98cfe201ece825f`. It uses custom CUDA: the
HAC hash grid and arithmetic coder.

**Outcome: matched.** On all 3 scenes, the PSNR of the decoded model is within ±0.12 dB of the paper's per-scene
results (low-rate, λ = 0.004).
- The absolute sizes are 2–16 % larger than the paper's.
- ContextGS's size reduction over HAC (its main claim) reproduces. On the same images and the same runtime, our
  ContextGS is 16–24 % smaller than our HAC at equal PSNR. The paper reports 13–21 %.

## Results (30k steps, mipnerf360-1600 = PIL resize to width 1600, llffhold 8, decoded model)
Runs are in `runs/contextgs-mipnerf360-1600/contextgs/mipnerf360-1600-<scene>/`. Sizes come from
`plugin/report_size.py` (written to `contextgs_size.json`) and are real arithmetic-coded streams, with MB = 2^20 bytes.

| scene | PSNR ours / paper (Δ) | SSIM ours / paper | LPIPS ours / paper | size MB ours / paper (Δ) | anchors coded / total |
|---|---|---|---|---|---|
| bicycle | 25.18 / 25.08 (**+0.10**) | 0.743 / 0.738 | 0.296 / 0.271 | 25.38 / 21.82 (+16 %) | 1.05M / 1.16M |
| garden | 27.20 / 27.32 (**−0.12**) | 0.844 / 0.845 | 0.171 / 0.148 | 20.65 / 18.78 (+10 %) | 0.73M / 0.80M |
| counter | 29.33 / 29.38 (**−0.05**) | 0.912 / 0.912 | 0.265 / 0.197 | 6.40 / 6.30 (+2 %) | 0.24M / 0.30M |

LPIPS differs because we use the `lpips` package with valid-padding conventions and a white background. The paper
uses the official 3DGS metric code on a black background. SSIM matches.

**Size breakdown (MB):**

| scene | anchor | feat | offsets | scaling | hyper | masks | mlps | meta |
|---|---|---|---|---|---|---|---|---|
| bicycle | 5.98 | 8.99 | 4.59 | 2.80 | 1.51 | 1.16 | 0.32 | 0.04 |
| garden | 4.15 | 7.63 | 4.24 | 2.22 | 1.26 | 0.81 | 0.32 | 0.03 |
| counter | 1.36 | 2.34 | 0.92 | 0.70 | 0.52 | 0.23 | 0.32 | 0.01 |

The encode takes 12–23 s and the decode 25–70 s.

## Relative check against HAC (same images, same runtime: `reproductions/hac/report.md`)

| scene | HAC ours (PSNR / MB) | ContextGS ours (PSNR / MB) | size Δ ours | size Δ paper (ContextGS vs HAC rows) |
|---|---|---|---|---|
| bicycle | 25.08 / 33.32 | 25.18 / 25.38 | **−24 %** | −21 % |
| garden | 27.14 / 24.93 | 27.20 / 20.65 | **−17 %** | −17 % |
| counter | 29.33 / 7.59 | 29.33 / 6.40 | **−16 %** | −13 % |

The absolute size offset follows the same scene pattern as HAC's: bicycle has the largest gap, counter the smallest.
This points to the shared cause recorded in the HAC report. More anchors survive under the runtime's update order
and gsplat statistics, and the context model itself is not the cause. The batch rules do not allow iterating on
this: PSNR is within tolerance, and a fix would change the shared runtime.

## Fidelity gaps (kept; see plan.md §Fidelity gaps)
1. The rasterizer is gsplat, not the Scaffold fork. `visible_filter` was re-implemented.
2. The background is white; the official mip360 runs use black.
3. Views are sampled randomly with replacement. The loss uses valid-convolution SSIM.
4. The runtime steps Adam before `after_backward`. The official code densifies first and skips the last step.
5. Global weights are carried in the per-anchor `cgs_blob` field, because `evaluate()` does not restore component
   state.
6. The codec is the HAC CUDA arithmetic coder for all streams. The official code uses torchac and compressai rANS for
   the hyperprior. The probability models are the same, and the expected size difference is well under 1 %.
7. Only mip360's voxel size (0.001) is registered.
8. Runs need `--recipe 3dgs`, all SfM points, and `--steps 30000`.

## Reproduce
```bash
T=/home/sejain/splatify/tools/ssenv
cd reproductions/contextgs/plugin && $T pip install -e . && $T ss-verify contextgs --device cuda
python3 tools/bench.py launch --exp contextgs-mipnerf360-1600 --methods contextgs --dataset mipnerf360-1600 --scenes bicycle garden counter --steps 30000
$T python reproductions/contextgs/plugin/report_size.py runs/contextgs-mipnerf360-1600/contextgs/mipnerf360-1600-*
```
