# MS-GS (#98, arXiv 2311.17089) — build report (pre-benchmark)
- **Status:** ready for benchmark.
- **Method:** `ms-gs` (plugin `plugin/`, module `ss_ms_gs`).
- **Official code:** JokerYan/MS-GS @50e5040 and MS-GS-rasterizer @e7a2b4f, followed. Where the paper and code disagree the code is used (see paper.md).

## Implementation
- **C1 pixel coverage:** the minimum half-extent of the undilated 2D Gaussian at the 1/255 level set. Fused CUDA kernel `csrc/coverage.cu`, with a torch reference.
- **C2 aggregate-and-insert** at iterations 1010/1020/1030 for levels 4×/16×/64×:
  - chosen: level-0 Gaussians with S < 1 px on any training view;
  - positions contracted with 2 − 1/x;
  - voxel size 0.02·r/4;
  - raw parameters averaged per voxel;
  - scale × 2/clip(S̄, 0.25, 2).
- **C3 selective rendering:** cull if (S/S_min < 0.5 and S < 2) or S/S_max > 2. The ranges are updated at each Gaussian's own level with 0.95/1.05 decay. Culled Gaussians get opacity 0, so gsplat gives them radius 0.
- **C4 schedule:**
  - one resolution per epoch: 1× with p = 0.75, otherwise the least-trained of {2,3}, {4,5}, {6,7}; 2× never;
  - loss ×0.1 off 1×;
  - density control and opacity reset on level-0 Gaussians only;
  - position LR over 30k, then constant;
  - 40k iterations.
- **Fields:** `ms_level`, `ms_max_px`, `ms_min_px`, stored in the checkpoint.

## Protocol
- **Not mipnerf360-ms:** MS-GS does not use the MTMT protocol.
- **Base:** the 3DGS `-r -1` loader (width 1600).
- **Scales:** 2^i up to 128×, with multi-scale training; tables report 1/4/16/64×.
- **New dataset `mipnerf360-msgs`** (tools/make_multiscale_msgs.py; garden and counter):
  - train = 1× frames, with lower scales rendered by the plugin;
  - test = 1–64×, PIL bicubic from full res;
  - 128× is below the loader minimum and is not evaluated.

## Verification
- ss-inspect is clean, 8 unit tests pass, ruff is clean, and ss-verify passes.
- Coverage vs float64: 9e-7.
- CUDA kernel vs torch: 4e-4, with 0 mask flips.
- Aggregation vs an independent loop: 5e-7.
- Training targets vs official PIL images: 53–84 dB.

## Smoke test (garden, 2000 steps, PSNR at 1×/4×/16×/64×)
| | 1× | 4× | 16× | 64× | Gaussians | time |
|---|---|---|---|---|---|---|
| ms-gs | 21.98 | 21.66 | 21.11 | 23.06 | 895k | 125 s |
| gsplat | 22.97 | 20.62 | 16.26 | 14.51 | 778k | 160 s |

smoke-1 crashed on the 128× target resize; this is fixed (CPU resize) and the run is kept.

## Benchmark request
- `ms-gs` 40k (exp ms-gs-msgs) and `gsplat` 30k (exp ms-gs-msgs-3dgs) on mipnerf360-msgs, garden and counter.
- **Targets:**
  - garden 27.16 / 23.99 / 26.41 / 24.79 (paper 3DGS 27.27 / 20.42 / 16.74 / 14.92);
  - counter 29.17 / 26.77 / 23.44 / 24.59 (paper 3DGS 29.15 / 24.81 / 17.94 / 14.32).

## Fidelity gaps
1. Training targets at 2×–128× are a CPU antialiased bicubic resize of the 1600-px frame, not PIL from full res (≥53 dB apart).
2. **Runtime differences:**
   - views are sampled with replacement within the official per-epoch resolution sequence;
   - densification runs after the optimizer step;
   - white background (official: black);
   - gsplat rasterizer instead of the MS-GS fork;
   - the SH degree steps one iteration late.
3. 128× is not evaluated.
4. Runtime SSIM (valid convolution) and LPIPS (lpips VGG).
5. **Code values used where the paper differs:**
   - S_rel^max 2.0 (paper 1.5);
   - insertion threshold 1 px (paper 2);
   - voxel sizes 0.01/0.02/0.03 (paper 0.02/0.03/0.04);
   - only level-0 Gaussians are aggregated;
   - one-sided contraction.

## Review-1 resolution (orchestrator)
- **Verdict:** approved, with 0 critical, 0 major and 5 minor issues.
- **Confirmed against the official code:**
  - The protocol, and test images byte-identical to the official PIL resize at 1/4/64×.
  - llffhold 8.
  - The targets (supp. Tab. 360_per_scene).
  - C1–C4 line by line, including the per-Gaussian range updates and selection at eval.
- **Gaps added:**
  - gsplat culls opacity < 1/255 and uses narrower radii than the fork's 3σ, which changes which Gaussians count as visible for the range updates.
  - The voxel grid anchoring is assumed to match Open3D-ML VoxelPooling; ml3d is not installed to check.
  - The README example uses --ms_train_max_scale 6, while train_all.py uses 7; we use 7.
  - The CPU target resize makes timings incomparable with the paper's.
  - The official per-level grad/radii accumulators are inert (the 20 px prune never fires), so they are skipped.
- **Benchmark:**
  - ms-gs-msgs: ms-gs at 40k on garden and counter.
  - ms-gs-msgs-3dgs: gsplat at 30k, the paper's single-scale 3DGS row.
  - Summarise with `tools/multiscale_summary.py --dataset mipnerf360-msgs --scales 1,4,16,64`.

## Control: single-scale gsplat (ms-gs-msgs-3dgs, 30k) — matches the paper's 3DGS row
| PSNR | 1× | 4× | 16× | 64× |
|---|---|---|---|---|
| garden ours | 26.94 | 21.24 | 17.47 | 15.55 |
| garden paper 3DGS | 27.27 | 20.42 | 16.74 | 14.92 |
| counter ours | 28.99 | 24.90 | 18.15 | 14.36 |
| counter paper 3DGS | 29.15 | 24.81 | 17.94 | 14.32 |

The control is within 0.9 dB of the paper's 3DGS at every scale, which supports the mipnerf360-msgs protocol. The ms-gs runs (40k) are still pending.

## Benchmark result (ms-gs-msgs, 40k) — PARTIAL
PSNR per scale (ours / paper):

| scene | 1× | 4× | 16× | 64× | mean of 4 |
|---|---|---|---|---|---|
| garden | 26.79 / 27.16 (−0.37) | 23.90 / 23.99 (−0.09) | 24.70 / 26.41 (**−1.71**) | 26.24 / 24.79 (**+1.45**) | 25.41 / 25.59 (−0.18) |
| counter | 28.70 / 29.17 (−0.47) | 26.60 / 26.77 (−0.17) | 24.73 / 23.44 (**+1.29**) | 27.36 / 24.59 (**+2.77**) | 26.85 / 25.99 (+0.86) |

- **1× and 4×:** within ±0.5 dB on both scenes.
- **16× and 64×:** off by 1.3–2.8 dB, in both directions:
  - garden is too low at 16× and too high at 64×;
  - counter is too high at both.
  So the per-scale numbers do not match, even though garden's 4-scale mean does.
- **Gain over the single-scale control:** large, as in the paper.
  - At 64×: +10.7 (garden) and +13.0 (counter) dB, vs the paper's +9.9 / +10.3.
  - At 16×: +7.2 / +6.6, vs +9.7 / +5.5.
- **Final Gaussians:** 4.22M (garden), 1.07M (counter).
- **Not explained.** The test images are byte-identical to the official ones and the control matches the paper's 3DGS within 0.9 dB, so the difference lies in how the coarse levels are produced and selected. Candidates, none of them tested:
  - Selection uses gsplat visibility (opacity < 1/255 culled, narrower radii), which changes the per-Gaussian S_min/S_max ranges.
  - The voxel grid anchoring is assumed to match Open3D-ML.
  - Single seed: the paper's own 16× > 64× ordering on garden and 64× > 16× on counter suggests the low scales are noisy.
- No grounded fix was found, so no iteration was made.
