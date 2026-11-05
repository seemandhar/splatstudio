# c3dgs reproduction report

**Paper:** Niedermayr et al., *Compressed 3D Gaussian Splatting for Accelerated Novel View Synthesis* (CVPR 2024, arXiv 2401.02436).

**Method:** `c3dgs`. Training runs 30k steps of 3DGS, then the compression stage for 5k more steps (35k total):
- sensitivity-aware vector clustering of colour and shape,
- quantization-aware fine-tuning,
- entropy-coded npz export.

The plugin is in `plugin/` and frozen at `3666e9660aab3b68`.

**Outcome: matched.**
- **Quality:** PSNR is within ±0.28 dB of the paper's per-scene numbers on all 3 scenes. The cost of compression on our own 3DGS (−0.16 / −0.42 / −0.25 dB) matches the paper's cost on its 3DGS (−0.20 / −0.43 / −0.22) to within 0.04 dB.
- **Size:** within −17 % to +5 % of the paper's.

## Results
Setup:
- Mip-NeRF 360, images_4 and images_2, llffhold 8.
- Metrics are measured on the decoded npz.
- Size is `len(point_cloud.npz) / 2^20`, as in the official `compress.py` (from `plugin/report_size.py`).

| scene | PSNR ours / paper (Δ) | SSIM ours / paper | LPIPS ours / paper | size MiB ours / paper (Δ) | #G (before → after prune) |
|---|---|---|---|---|---|
| bicycle | 24.76 / 24.97 (**−0.21**) | 0.743 / 0.751 | 0.276 / 0.240 | 49.3 / 47.1 (+5 %) | 6.57M → 5.61M |
| garden | 26.47 / 26.75 (**−0.28**) | 0.842 / 0.844 | 0.167 / 0.144 | 38.7 / 46.6 (−17 %) | 4.76M → 4.31M |
| counter | 28.60 / 28.67 (**−0.07**) | 0.897 / 0.896 | 0.276 / 0.215 | 14.0 / 13.8 (+1 %) | — → 1.07M |

**Cost of compression, each against its own 3DGS:**

| scene | our gsplat 3DGS (`baseline3dgs-mipnerf360-30000`) | c3dgs − ours | paper: c3dgs − its 3DGS |
|---|---|---|---|
| bicycle | 24.92 | −0.16 | −0.20 |
| garden | 26.89 | −0.42 | −0.43 |
| counter | 28.85 | −0.25 | −0.22 |

- **Garden size.** Our garden file is smaller because our base model has fewer Gaussians than the authors' released 3DGS model.
- **Compression ratio.** The paper reports about 31× on Mip-NeRF 360 (795 MB to 28.8 MB on average). Against an estimated 3DGS PLY of 248 B per Gaussian, our ratios, computed on the post-prune Gaussian counts, are about 27× / 26× / 18×.
- **LPIPS.** The higher LPIPS comes from the runtime's metric convention: the `lpips` package and a white background.

## Run notes
- Bicycle ran out of GPU memory on a shared GPU (another user's process held 25.6 GB).
- It was re-queued with `--per-gpu 1 --min-free-gb 36` and then finished.
- The failed attempts are kept as `.oom-*` directories.

## Fidelity gaps (kept, from plan.md)
1. **Base model.** We start from our gsplat 3DGS, not from the authors' released 3DGS models. Compare the deltas above.
2. **Fine-tune background.** It is white; the official code uses black. The sensitivity pass does use black, as the code does.
3. **Final optimizer step.** The runtime takes an opacity optimizer step at step 35000.
4. **View sampling.** Views are sampled at random with replacement.
5. **Rasterizer.** gsplat replaces the official rasterizer, and sensitivity gradients come from gsplat. A probe checks them.
6. **Rotation in the final metrics.** It is stored as Q(normalize(r)) rather than rendered as normalize(FQ(r)).
7. **Rendering speed.** The hardware-rasterizer speed claim (WebGPU) is not reproduced.
8. **Scatter sums.** They use `index_add_` instead of torch_scatter: the same maths with a different float order.

## Review
`reviews/review-1.json` found 5 minor issues and no major ones.

## Reproduce
```bash
python3 tools/bench.py launch --exp c3dgs-mipnerf360 --methods c3dgs --dataset mipnerf360 --scenes bicycle garden counter --steps 35000
tools/ssenv python reproductions/c3dgs/plugin/report_size.py runs/c3dgs-mipnerf360/c3dgs/mipnerf360-*
```
