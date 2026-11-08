# FSGS reproduction report

**Paper:** *FSGS: Real-Time Few-shot View Synthesis using Gaussian Splatting*, ECCV 2024, arXiv 2312.00451.
**Official code:** github.com/VITA-Group/FSGS (Inria license, research use). It was read, never run, and the plugin is ported from it (cited).
**Plugin:** `plugin/` (method `fsgs`, module `ss_fsgs`). No custom CUDA.

**Outcome: matched.** The 8-scene mean on LLFF 3-view at 1/8 resolution is **20.50 PSNR vs. the paper's 20.31 (+0.19 dB)**.

## Results
Setup:
- 10k steps, `--recipe 3dgs`, all initial points kept, one seed.
- Dataset `llff-fewview3-r8-fsgs`, test split (every 8th image).
- The paper reports averages only, with no per-scene tables in the main text, supplementary or rebuttal. The full 8-scene mean is therefore the comparison.
- Runs: `runs/fsgs-llff3/`.

| | fern | flower | fortress | horns | leaves | orchids | room | trex | **mean PSNR** | SSIM | LPIPS |
|---|---|---|---|---|---|---|---|---|---|---|---|
| ours (fsgs) | 22.21 | 20.85 | 23.24 | 20.19 | 17.85 | 16.33 | 21.68 | 21.61 | **20.50** | 0.703 | 0.236 |
| paper FSGS (Table 1) | | | | | | | | | **20.31** | 0.652 | 0.288 |
| paper 3D-GS (Table 1) | | | | | | | | | 17.43 | 0.522 | 0.321 |

**SSIM and LPIPS are better than the paper's** (+0.051 SSIM, −0.052 LPIPS). The likely cause is differences in metric implementations, which this reproduction did not check:
- SSIM here is the runtime's valid-convolution SSIM. The official code uses the Inria `ssim`.
- LPIPS-VGG here comes from the `lpips` package. The official code uses `lpipsPyTorch`.

Other factors could also contribute: the rasterizer and the initial point cloud (see deviations). Treat the SSIM/LPIPS margins with caution. Only PSNR is directly comparable.

No gsplat baseline or ablation was run on this dataset. The gain over 3D-GS is therefore not verified; only the absolute PSNR is compared. In the 2k-step smoke test on horns, FSGS reached 19.65 vs. gsplat's 18.35 (`evidence/test.json`).

## Contributions → components
- **Proximity-guided unpooling** → `fsgs-density`.
  - Exact 3-NN search, using the mean squared neighbour distance.
  - Unpooling for steps < 2000 when the distance exceeds 5·extent and max scale > extent.
  - Extra split rule when the distance exceeds 10·extent.
  - Otherwise the official densify/prune.
  - Opacity reset to 0.05 at steps 2001, 5001 and 8001.
- **Pseudo views** → `fsgs-pseudo`.
  - 10k RegNeRF random LLFF poses.
  - Loss is 1 − Pearson against DPT run on the rendered pseudo view, with the gradient flowing through DPT.
  - Weight 0.5, ramped over 500 steps, active for steps 2001–9499.
- **Monocular-depth Pearson** → `fsgs-depth`.
  - Minimum of two Pearson variants against DPT depth of the training image.
  - Weight 0.05, dropping to 0.001 from step 9502.
- **Renderer** → `fsgs-renderer`.
  - Accumulated depth, black background, SH degree raised every 500 steps.
  - Straight-through opacity cap at 0.999, so gradients flow as in the Inria rasterizer.

**Verification:**
- `evidence/verify.json`: 8/8 probes pass (`declared_checks_passed`).
- 5 unit tests pass.
- `evidence/test.json`: smoke test, plus a pseudo-view mechanism run (`tools/smoke_pseudo.py`).
- Not separately reviewed (official-code port).

## Deviations from the paper / official code (kept)
1. **Depth estimator:** DPT-Hybrid, as in the official `depth_utils.py`. The paper's Table 5 names DPT-Large for the main row (20.31; Hybrid scores 20.29).
   - MiDaS code is the Aug 2024 master, with timm 0.6.12.
2. **Initial cloud:** COLMAP 3.9.1 MVS of the 3 training views. The authors' COLMAP version is unknown, and their released clouds were not used.
3. **New dataset key `llff-fewview3-r8-fsgs`:** `llff-fewview3-r8` with the official LLFF `images_8` PNGs, as loaded by FSGS's `--images images_8`. Built by `tools/prepare_llff_fsgs.py`.
4. **Code over paper:** the code's choices were followed where they differ from the paper (see `paper.md` §7):
   - unpooling thresholds use the squared distance, 5·extent and 10·extent;
   - pseudo poses are random (not interpolated between cameras);
   - the pseudo-view weight is 0.5.
5. **Runtime:**
   - views are sampled with replacement (the official code pops a shuffled stack);
   - the optimizer steps before densification (the official code densifies first).
6. **Rasterizer:** gsplat instead of the Inria rasterizer:
   - alpha clamp 0.999 vs. 0.99;
   - opacity-aware radii;
   - the straight-through gradient emulation.
7. **Metrics:** SSIM and LPIPS implementations, as described above.
8. **Seeds:** one seed per scene, so results are not averaged over seeds.
