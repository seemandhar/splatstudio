# 3D Gaussian Splatting (Kerbl et al., SIGGRAPH 2023) — plan #24

**Outcome: matched.** Built-in native config `gsplat` (SplatStudio gsplat backend with the 3DGS recipe:
`--recipe 3dgs`, camera-extent scene scale, 3-NN initial scales, opacity reset every 3000 steps before 15k —
the runtime fix for gsplat 1.5.x's reset bug, see CLAUDE.md). No new plugin: this entry benchmarks the
existing config against the paper. Paper numbers: `tools/reference_results.json` (3DGS supplementary per-scene table).

Runs: `runs/baseline3dgs-mipnerf360-30000`, `runs/baseline3dgs-tandt-30000`, `runs/baseline3dgs-db-30000`
(30k steps, llffhold8, images_4 outdoor / images_2 indoor). These are the shared baselines used by every other reproduction.

| dataset | scene | ours | paper | Δ |
|---|---|---:|---:|---:|
| Mip-NeRF 360 | bicycle | 24.92 | 25.25 | −0.33 |
| | flowers | 21.63 | 21.52 | +0.11 |
| | garden | 26.89 | 27.41 | **−0.52** |
| | stump | 26.23 | 26.55 | −0.32 |
| | treehill | 22.28 | 22.49 | −0.21 |
| | room | 31.52 | 30.63 | **+0.89** |
| | counter | 28.85 | 28.70 | +0.15 |
| | kitchen | 31.34 | 30.32 | **+1.02** |
| | bonsai | 31.89 | 31.98 | −0.09 |
| | **average** | **27.28** | **27.21** | **+0.07** |
| Tanks&Temples | truck | 25.21 | 25.19 | +0.02 |
| | train | 21.52 | 21.10 | +0.42 |
| | **average** | **23.36** | **23.14** | **+0.22** |
| Deep Blending | drjohnson | 28.76 | 28.77 | −0.01 |
| | playroom | 29.80 | 30.04 | −0.24 |
| | **average** | **29.28** | **29.41** | **−0.13** |

All three dataset averages and 10 of 13 scenes are within ±0.5 dB. The three outliers: garden is 0.52 dB low;
room and kitchen are ~1 dB **above** the paper (indoor scenes are known to have improved in later 3DGS releases and
gsplat versions; the paper's indoor numbers are from the original release).

SSIM: 0.812 / 0.849 / 0.903 vs paper 0.815 / 0.841 / 0.903. LPIPS is systematically higher (0.259 / 0.207 / 0.321
vs 0.214 / 0.183 / 0.243) — the known runtime LPIPS offset (`lpips` package VGG, white background, valid-SSIM;
see `reproductions/_calibration/`).

Known differences from the original code: gsplat rasteriser instead of the Inria one, white background,
random-with-replacement view sampling, SSIM with valid convolution, LPIPS-VGG from the `lpips` package.
