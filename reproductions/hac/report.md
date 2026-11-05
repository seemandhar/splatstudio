# HAC — reproduction report

**Paper:** *HAC: Hash-grid Assisted Context for 3D Gaussian Splatting Compression* (ECCV 2024, arXiv 2403.14530).

**Official code:** YihangChen-ee/HAC @ `110dcf0a`. It was read but never executed.

**Method:** `hac`, plugin `plugin/` (`ss_hac`), frozen at `a45cce4af18b4702`. This uses custom CUDA: the binary
hash grid and the arithmetic coder are built with torch cpp_extension.

**Outcome: matched (PSNR).** The decoded-model PSNR is within ±0.15 dB of the paper's per-scene Table numbers
(λ = 0.004, the low-rate setting) on all 3 scenes. The compressed sizes are 5–21 % **larger** than the paper's.

## Results (30k steps, mipnerf360-1600 = PIL width 1600, llffhold 8, decoded model)
Runs are in `runs/hac-mipnerf360-1600/hac/mipnerf360-1600-<scene>/`. Sizes come from `plugin/report_size.py`, which
writes `hac_size.json`. They use the official bit formula, the real arithmetic-coded streams, and MB = bits / 8 / 2^20
(the official unit).

| scene | PSNR ours / paper (Δ) | SSIM ours / paper | size MB ours / paper (Δ) | anchors coded |
|---|---|---|---|---|
| bicycle | 25.08 / 25.05 (**+0.03**) | see test.json / 0.742 | 33.32 / 27.54 (+21 %) | 1.07M |
| garden | 27.14 / 27.28 (**−0.14**) | — / 0.842 | 24.93 / 22.69 (+10 %) | 0.75M |
| counter | 29.33 / 29.35 (**−0.02**) | — / 0.911 | 7.59 / 7.26 (+5 %) | 0.26M |

The mean SSIM over the 3 scenes is 0.831, against a paper mean of 0.832 over the same 3 scenes. The mean LPIPS is
0.247.

**Size breakdown (MB):**

| scene | anchor | feat | offsets | scaling | masks | hash | mlps |
|---|---|---|---|---|---|---|---|
| bicycle | 6.13 | 13.65 | 7.55 | 4.45 | 1.26 | 0.12 | 0.16 |
| garden | 4.32 | 10.60 | 5.61 | 3.26 | 0.87 | 0.11 | 0.16 |
| counter | 1.51 | 3.34 | 1.19 | 1.07 | 0.26 | 0.07 | 0.16 |

The encode takes 10–15 s and the decode 20–36 s per scene.

**Size gap.** The extra size grows with scene size, and every stream grows with it (anchor, feat, offsets). That
points to more anchors surviving, not to a weaker entropy model. The paper does not report anchor counts, so this
cannot be checked directly. Likely causes are gaps 2, 4 and 5 below: gsplat's visibility and statistics, sampling
with replacement, and Adam stepping before growing and pruning. All of these change the anchor growing/pruning
statistics. Per the batch rules this was not iterated on, since PSNR is within tolerance. A fix grounded in the paper
would mean aligning the update order in the runtime, which is a shared runtime change.

**Baseline comparison** (`baseline3dgs-mipnerf360-1600-30000`, gsplat 3DGS, same images and protocol):

| scene | baseline PSNR / #G | HAC − baseline | HAC size vs 3DGS PLY (~248 B/G) |
|---|---|---|---|
| bicycle | 24.93 / 6.17M | **+0.15** | 33.3 MB vs ~1530 MB (~46x) |
| garden | 26.93 / 4.55M | **+0.21** | 24.9 MB vs ~1130 MB (~45x) |
| counter | 29.04 / 1.19M | **+0.29** | 7.6 MB vs ~294 MB (~39x) |

HAC beats our 3DGS on all three scenes at roughly 39-46x compression. This agrees in direction with the
paper's claim that HAC exceeds 3DGS quality on Mip-NeRF 360 at a small fraction of its size. The PLY sizes
are estimates: Gaussian count x 248 bytes (59 float32 values), not measured files.

## Fidelity gaps (kept; see plan.md §Fidelity gaps)
1. Resolution is matched through the `mipnerf360-1600` conversion.
2. The rasterizer is gsplat instead of the Scaffold fork. `visible_filter` is re-implemented to match the official
   anchor prefilter.
3. The background is white; the official mip360 runs use black.
4. Views are sampled randomly with replacement. SSIM uses valid convolution (runtime convention).
5. The runtime steps Adam before `after_backward`; the official code densifies first.
6. Global weights travel in a per-anchor `hac_blob` field, because `evaluate()` does not restore component state
   (TODO-runtime).
7. Only the mip360 voxel size (0.001) is registered.
8. The codec is the official CUDA arithmetic coder, in memory. The paper used torchac.

## Review
`reviews/review-1.json` raised 2 majors, both fixed before the benchmark:
1. The anchor `visible_filter` prefilter.
2. The 1.6K protocol data.

It also raised 6 minors, which were documented.

## Reproduce
```bash
T=/home/sejain/splatify/tools/ssenv
cd reproductions/hac/plugin && $T pip install -e . && $T ss-verify hac --device cuda
python3 tools/bench.py launch --exp hac-mipnerf360-1600 --methods hac --dataset mipnerf360-1600 --scenes bicycle,garden,counter --steps 30000
$T python reproductions/hac/plugin/report_size.py runs/hac-mipnerf360-1600/hac/mipnerf360-1600-*
```
