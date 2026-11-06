# GS-IR: build report (plan id 91)
**Status:** built and smoke-tested; benchmark pending.
- **Paper:** CVPR 2024, arXiv 2311.16473.
- **Official code:** lzhnb/GS-IR @ e5a030b, read in full and not executed.
- **Plugin:** `plugin/` (`ss_gsir`), with methods `gs-ir` (40k steps) and `gs-ir-smoke`.

## What was built (all three stages run inside one ss-train run)
1. **Initial stage, steps 1–30000:**
   - 3DGS training plus a per-Gaussian normal.
   - The normal is supervised by the normal derived from the linearly interpolated depth (L1 ×1 + edge-aware TV ×5).
   - Normals are rasterized on detached geometry, as in the official backward.
2. **Baking at step 30000**, inside the density component:
   - a 256³ probe grid over ±16;
   - each probe renders a 6-face 256² depth cube with gsplat (near 0.2);
   - threshold 0.4, then 16 SH coefficients, then dilation.
3. **Decomposition, steps 30001–40000:**
   - albedo, roughness and metallic;
   - a 256² cube-map light with mips, diffuse convolution, a GGX pre-filter (own CUDA kernel) and a BRDF LUT;
   - 64³ SH irradiance volumes;
   - occlusion recovery (own CUDA kernel);
   - losses: L1 shading, material TV, roughness/metallic prior, light TV.

**Global state:** kept in a density-owned bank (`bank.py`), exported to `density_state["gsir"]`. ss-eval restores it, with a Gaussian fingerprint check, and reports the PBR metrics.
**Evaluation:** reproduces the official render.py quirks (roughness += T, padded normal map, no roughness remap).

## Checks
- ss-inspect is clean, 15/15 unit tests pass and ruff is clean.
- ss-verify passes 10 probes:
  - linear depth 2.333 vs volumetric 1.75;
  - plane normal 0.03°;
  - no normal-loss gradient into geometry;
  - wall visibility 0.21 / 0.97;
  - GGX filter matches a numpy reference to 1e-7;
  - the constant-environment PBR closed form holds;
  - irradiance trilinear matches reference loops;
  - the stage switch works.

## Smoke test (garden, 2000 steps, stage switch at 1500)
| run | test PSNR / SSIM / LPIPS | time |
|---|---|---|
| gs-ir-smoke (PBR render) | 21.22 / 0.503 / 0.466 | 609 s (baking 276 s) |
| gsplat baseline (SH render) | 23.00 / 0.643 / 0.418 | 356 s |

The PBR image is below the SH baseline, as expected: the paper reports garden −1.69 dB vs 3DGS.

## Benchmark
- garden (paper 25.72) and counter (paper 26.22) at 40k steps.
- The official protocol is `-i images_4/images_2 -r 1`, so the baseline is baseline3dgs-mipnerf360-30000 (garden 26.89, counter 28.85).
- Main comparison: our drop vs the paper's drop (−1.69 / −2.48 dB against its 3DGS rows 27.41 / 28.70).
- Compare PSNR only; LPIPS is not comparable.

## Fidelity gaps
1. **Rasterizer conventions** (gsplat instead of the official fork):
   - alpha clamp 0.999 vs 0.99;
   - training near plane 0.01 vs 0.2;
   - stage-1 background white vs black.
2. **No nvdiffrast:** the install was refused by the permission check. Its cube texture ops are re-implemented in PyTorch (padded seams, approximated corners) and were not compared against nvdiffrast.
3. **Runtime conventions:**
   - Views are sampled with replacement.
   - The optimizer steps on the final iteration.
   - Dilation is deterministic.
4. **Metrics:** PSNR is computed on whole-image MSE, while the official code averages per-channel PSNR.
5. **Official-code quirks kept on purpose:**
   - diffuse = albedo·cube, with no (1−m)/π;
   - z-depth used as the ray distance;
   - irradiance box ±1.5 vs occlusion box ±16;
   - roughness/metallic prior 0.001.
6. **Not measured:** TensoIR normal, albedo and relighting metrics.

## Review-1 resolution (orchestrator)
Approved, with 0 critical, 0 major and 7 minor issues. Every stage was checked against the official code:
- the stage-1 normal supervision;
- baking with the README's Mip-NeRF 360 settings, in COLMAP world units;
- shade.py composition, LRs and loss weights;
- the eval path (PBR, no tone map).
- **nvdiffrast substitute:** face selection, UV mapping and mip selection match nvdiffrast; only the corner texels are approximated. The reviewer estimates the PSNR effect at well under 0.05 dB, but it was not measured against nvdiffrast.
- **PSNR bias:** the official PSNR is the mean of per-channel PSNRs, so ours is biased about 0.05–0.15 dB low.
- **Other minors:**
  - The border-pixel normal gradient is not dropped.
  - The train vs eval path is selected by grad mode.
  - Half-pixel offset in the depth back-projection.
- **Benchmark:** gsir-mipnerf360 (garden, counter, 40k). Compare our drop vs baseline3dgs-mipnerf360-30000 with the paper's drop (−1.69 / −2.48).

## Benchmark result (gsir-mipnerf360, 40k, PBR render) — PARTIAL
| scene | ours | paper | Δ | our baseline | paper 3DGS | drop ours / paper |
|---|---|---|---|---|---|---|
| garden | 24.92 / 0.780 / 0.199 | 25.72 | **−0.80** | 26.89 | 27.41 | −1.97 / −1.69 |
| counter | 25.86 / 0.841 / 0.327 | 26.22 | −0.36 | 28.85 | 28.70 | −2.99 / −2.48 |

- **Counter:** within ±0.5 dB.
- **Garden:** 0.80 dB short. Correcting for the known PSNR-metric bias (ours is about 0.05–0.15 dB low against the official per-channel mean) leaves it at about −0.65, still outside.
- **Drop below 3DGS:** about 0.3–0.5 dB larger than the paper's on both scenes.
- **Candidates** (not tested):
  - the PyTorch cube-texture replacement for nvdiffrast, with corner texels approximated (the reviewer estimated ≪0.05 dB);
  - the white stage-1 background;
  - gsplat's 0.999 vs 0.99 alpha clamp in the normal/depth renders.
- **No grounded fix:** none was identified, so there was no iteration.
