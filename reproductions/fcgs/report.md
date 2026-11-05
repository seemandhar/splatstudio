# FCGS: reproduction report

**Paper:** *Fast Feedforward 3D Gaussian Splatting Compression* (ICLR 2025, arXiv 2410.08017).

**Official code:** `31e59a46`, with the official pretrained codec weights. The plugin is in `plugin/` (`ss_fcgs`) and uses custom CUDA (the official grid and arithmetic-coder kernels).

FCGS is **feed-forward**: it compresses an already-trained 3DGS model in one pass, with no training. We therefore applied it with `plugin/apply_fcgs.py` (λ = 1e-4, the high-rate setting) to our finished gsplat baselines in `runs/baseline3dgs-mipnerf360-30000/`.

**Outcome: matched.** The decoded-model PSNR is within ±0.37 dB of the paper's per-scene numbers on all 3 scenes. Sizes are not comparable yet: they use a stand-in coder for positions (see below).

## Results
Mip-NeRF 360 (images_4 / images_2), llffhold 8 test split. Outputs are in `results/applied-<scene>.json`. Sizes are in MiB (bits / 8 / 2^20, the official unit).

| scene | PSNR ours / paper (Δ) | SSIM ours / paper | LPIPS ours / paper | size MiB ours / paper | input #G |
|---|---|---|---|---|---|
| bicycle | 24.70 / 24.95 (**−0.25**) | 0.747 / 0.739 | 0.250 / 0.248 | 114.0* / 106.0 | 6.59M |
| garden | 26.83 / 27.20 (**−0.37**) | 0.857 / 0.852 | 0.135 / 0.126 | 91.6* / 120.2 | 4.76M |
| counter | 28.82 / 29.07 (**−0.26**) | 0.906 / 0.913 | 0.262 / 0.185 | 23.8* / 23.4 | 1.22M |

\* **Size caveat (superseded, see the GPCC re-measurement below).** The paper codes positions with MPEG GPCC (`tmc3`), and building it was blocked in this environment. We used an octree xyz coder as a stand-in. Its xyz stream is 13.2 / 10.1 / 2.7 MiB. All other streams use the official networks and coder.
- Garden is smaller than the paper's because our input model has fewer Gaussians: 4.76M against the 5.62M the paper's 3DGS has.
- `apply_fcgs.py --xyz-coder gpcc` will re-measure the sizes once `$FCGS_TMC3` points at a tmc3 build.

**Size breakdown (MiB):**

| scene | xyz (octree) | mask | colour features (fea+feq) | geometry (geo) |
|---|---|---|---|---|
| bicycle | 13.15 | 0.75 | 54.98 | 45.10 |
| garden | 10.13 | 0.54 | 48.35 | 32.55 |
| counter | 2.74 | 0.14 | 13.07 | 7.85 |

**Cost of compression on the same input model** (our gsplat 3DGS, rendered on gsplat):

| scene | input PSNR | after FCGS | Δ |
|---|---|---|---|
| bicycle | 24.92 | 24.70 | −0.22 |
| garden | 26.89 | 26.83 | −0.06 |
| counter | 28.85 | 28.82 | −0.03 |

The paper's Mip-NeRF 360 average goes from 27.52 (3DGS) to 27.39 (FCGS high-rate), a cost of −0.13 dB. Our per-scene costs (−0.03 to −0.22) bracket that. The input models compress by about 13× / 12× / 12×, measured against a 62-float PLY.

Most of the remaining gap to the paper's absolute PSNR therefore comes from the input model, not the codec. The paper does not publish its per-scene 3DGS inputs, so this split cannot be checked exactly per scene.

## Fidelity gaps (kept)
1. **GPCC is unavailable**, so an octree xyz coder stands in. This affects size only; the decoded model is identical.
2. **Our base 3DGS differs**: gsplat with a white background and fewer Gaussians. FCGS's pretrained networks were trained on parameter statistics from official 3DGS models.
3. **Evaluation differs**: the SplatStudio metrics use a white background, no 8-bit rounding and valid-padding SSIM.
4. The codec was not retrained on DL3DV-GS; the official weights are used by design.
5. The decoded model is rendered with gsplat.
6. Official code bug: the encoder slices latents with a shape that does not match the decoder. The vendored code applies a fix with the same maths as the decoder's path; the reviewer checked it operator by operator (`reviews/review-1.json`).

## Review
`reviews/review-1.json` raised 1 major issue: the reported size mixed in the stand-in xyz coder without labelling it. Sizes here are labelled accordingly. It also raised 4 minors.

## Reproduce
```bash
T=/home/sejain/splatify/tools/ssenv
for s in bicycle garden counter; do
  $T python reproductions/fcgs/plugin/apply_fcgs.py --run runs/baseline3dgs-mipnerf360-30000/gsplat/mipnerf360-$s \
     --lmd 1e-4 --out reproductions/fcgs/results/applied-$s.json --skip-original
done   # bicycle needs a GPU with ~40 GB free
```


## GPCC re-measurement (real MPEG TMC13 `tmc3`)
Positions were re-encoded with the official GPCC coder: MPEG TMC13 release-v14.0, built at `~/models/tmc13/src/build/tmc3/tmc3`, run with the official FCGS flags (`apply_fcgs.py --xyz-coder gpcc`). Outputs are in `results/gpcc-<scene>.json`.

| scene | PSNR (unchanged) | size MiB, GPCC | size MiB, octree stand-in | paper | Δ vs paper | xyz stream MiB, GPCC / octree |
|---|---|---|---|---|---|---|
| bicycle | 24.70 | **113.56** | 113.99 | 106.01 | +7.1 % | 12.72 / 13.15 |
| garden | 26.83 | **91.21** | 91.57 | 120.15 | −24.1 % | 9.77 / 10.13 |
| counter | 28.82 | **23.71** | 23.81 | 23.39 | +1.4 % | 2.63 / 2.74 |

- **What GPCC changes:** only the position stream, which shrinks by 0.1–0.4 MiB per scene. The decoded model and its PSNR are identical, so the octree stand-in had been accurate to within 0.5 %.
- **Why sizes differ from the paper:** the remaining size differences follow the input model's Gaussian count. Garden is 4.76M here against 5.62M in the official 3DGS model; bicycle is 6.59M. The differences do not come from the codec.
- **Fidelity gap 1 is closed:** GPCC is now available and used.
