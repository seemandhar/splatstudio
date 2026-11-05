# RDO-Gaussian: reproduction report

**Paper:** *End-to-End Rate-Distortion Optimized 3D Gaussian Representation* (ECCV 2024, arXiv 2406.01597).

**Official code:** `710ebab8`, read only. It was never executed.

**Method:** `rdo-gaussian`, using the paper's highest-rate setting (λ_GS = 0.0005, λ_SH = 0.005).
- Plugin: `plugin/` (`ss_rdo_gaussian`), frozen at `a209d95e0112acc2`.
- Entropy coder: vendored CompressAI rANS, a CPU C++ extension.

**Outcome: matched.**
- **PSNR:** within 0.30 dB of the paper's per-scene numbers on all 3 scenes. The loss relative to our own 3DGS baseline closely matches the loss the paper reports relative to its 3DGS (≤0.07 dB apart on each scene).
- **Size:** within ±15 % of the paper's.

## Results

**Setup:**
- 30k steps on `mipnerf360-1600`, the official `-r -1` loader: PIL resize to width 1600.
- llffhold 8.
- Evaluated on the decoded bitstream, which is what the official `render.py` evaluates.

**Sources:**
- Runs: `runs/rdo-gaussian-mipnerf360-1600/rdo-gaussian/`.
- Sizes: `plugin/report_size.py`, which measures the real rANS bitstream. MB = 1e6 bytes.

| scene | PSNR ours / paper (Δ) | SSIM ours / paper | LPIPS ours / paper | size MB ours / paper (Δ) | #G (prune ratio) |
|---|---|---|---|---|---|
| bicycle | 24.65 / 24.88 (**−0.23**) | 0.725 / 0.734 | 0.307 / 0.266 | 49.6 / 43.1 (+15 %) | 3.95M (36 %) |
| garden | 26.48 / 26.78 (**−0.30**) | 0.828 / 0.838 | 0.191 / 0.148 | 34.3 / 38.8 (−12 %) | 2.65M (42 %) |
| counter | 28.40 / 28.51 (**−0.11**) | 0.902 / 0.902 | 0.271 / 0.205 | 8.7 / 7.9 (+10 %) | 0.60M (49 %) |

**Size breakdown (MB):**

| scene | VQ indexes | positions (fp16) | codebooks | opacity logits |
|---|---|---|---|---|
| bicycle | 24.72 | 23.69 | 1.04 | 0.15 |
| garden | 17.16 | 15.89 | 1.06 | 0.15 |
| counter | 3.89 | 3.59 | 1.05 | 0.15 |

**Compared with each paper's own 3DGS baseline.** Our baseline is `baseline3dgs-mipnerf360-1600-30000`: gsplat 3DGS on the same images.

| scene | our baseline | ours − baseline | paper − its 3DGS |
|---|---|---|---|
| bicycle | 24.93 | −0.28 | −0.23 |
| garden | 26.93 | −0.45 | −0.52 |
| counter | 29.04 | −0.64 | −0.59 |

The rate–distortion trade-off reproduces to within 0.07 dB per scene. Our baseline used 6.17M / 4.55M / 1.19M Gaussians, which at roughly 248 B per Gaussian is about 1530 / 1130 / 294 MB. That makes our models about 31× / 33× / 34× smaller.

LPIPS is higher across scenes. This is the runtime's metric convention (the `lpips` package with a white background); PSNR and SSIM agree with the paper.

## Fidelity gaps (kept; see plan.md)
1. The runtime applies one Adam step at step 30000; the official code skips the last step.
2. Shared runtime differences:
   - white background instead of black;
   - views sampled at random with replacement;
   - SSIM with valid convolution;
   - gsplat rasterizer;
   - the `lpips` package.
3. k-means and reactivation randomness differ from the official run.
4. Scale codewords of exactly zero are clamped at 1e-12 instead of −inf. This is essentially unreachable.
5. An attribute with no coded symbols is stored as an empty rANS string, where the official encoder asserts.
6. Run time: VQ assignment every 10 steps makes training several times slower than gsplat.

## Review
`reviews/review-1.json` raised one major issue: a documentation error about the eval mode. It was corrected, and decoded-bitstream eval was confirmed to be the official protocol.

## Reproduce
```bash
python3 tools/bench.py launch --exp rdo-gaussian-mipnerf360-1600 --methods rdo-gaussian --dataset mipnerf360-1600 --scenes bicycle garden counter --steps 30000
tools/ssenv python reproductions/rdo-gaussian/plugin/report_size.py runs/rdo-gaussian-mipnerf360-1600/rdo-gaussian/mipnerf360-1600-*
```
