# GOF (plan #57) — build report

**Status:** plugin built and verified, Mip-NeRF 360 benchmark pending (`bench_request.json`).

- Method `gof` (`reproductions/gof/plugin`, module `ss_gof`). Official code
  autonomousvision/gaussian-opacity-fields @5245b20; its `diff-gaussian-rasterization`
  (ray–Gaussian intersection, plane normals, median depth, distortion, abs-gradient channel,
  `integrate` opacity-field kernel) is vendored unmodified in `ss_gof/csrc` (Inria/MPII licence)
  and JIT-built via `tools/ssenv` (~70 s first build).
- Contributions → components: ray-tracing renderer (C1), `raster.opacity_field` (C2),
  `gof-reg` loss = 100·distortion + 0.05·depth-normal from step 15000 (C3, C4), GOF density
  (quantile abs-gradient metric, resampled clones, min_opacity 0.05, Mip-Splatting 3D filter,
  filter-aware reset; screen-size prune omitted = official quirk) (C5).
- Probes (all pass, `evidence/verify.json`): kernel RGB/normal/depth/alpha/distortion within
  5e-5 and gradients within 3e-4 rel. of an independent float64 ray tracer; gsplat projection
  differs by 0.042; opacity field = o·G(min(t,t*)) within 4e-6; distortion weights detached;
  plane normal error 3e-6; densify metric changes clone set; recipe/schedule checks.
- Smoke (garden, 2000 steps): gof 22.25 / 0.651 / 0.374 (1.58 M G) vs gsplat 22.99 / 0.642 /
  0.419 (0.84 M G). −0.75 dB PSNR but better SSIM/LPIPS; training loss lower (0.088 vs 0.105).
  Regularisers forced on at step 1000 (`evidence/smoke_reg.py`): finite, L_reg 0.024→0.011.
- Benchmark: paper gives only outdoor/indoor averages (Table 4: 24.82 / 30.79 PSNR; its 3DGS row
  24.64 / 30.41) → all 9 mipnerf360 scenes at 30k, official images_4/images_2 = our protocol.

## Fidelity gaps
1. Views sampled with replacement; runtime optimizer step precedes densification.
2. Tetrahedral mesh extraction (CGAL Delaunay + binary-search marching tetrahedra) not ported →
   DTU Chamfer / TnT F1 not reproduced.
3. Decoupled appearance (TnT only) not ported (not used for Mip-NeRF 360).
4. 3D-filter coefficient in log space (same value, NaN-safe).
5. Evaluator SSIM (valid conv) / `lpips` package, shared with baseline.
6. Official CUDA quirk kept: distortion backward omits the 1/(1−T)² normalisation.
7. Watch item: early (2k) PSNR deficit with ~2× Gaussians; check at 30k against the paper's
   +0.18/+0.38 dB gain over 3DGS.

## Review-1 resolution (orchestrator)
- Major: benchmark launched with `--max-gaussians 20000000` and GPUs with ≥36 GB free (GOF reaches ~1.9× gsplat's Gaussian count; bicycle would exceed the default 12M cap).
- The 2k smoke deficit (22.25 vs 22.99) is not evidence either way: GOF's xyz LR follows the official 30k schedule (~1.18e-4·extent at step 2000) while the 2k gsplat smoke decays over 2000 steps (74× smaller at the end).
- Added gaps: order densify → reset → filter update (official densify → filter → reset); runtime takes the Adam step before densify/reset (official effectively skips it on those steps); SH degree one step later; training ~5× slower than gsplat.

## Benchmark results
**Outcome: matched (absolute, both groups).** Outdoor and indoor PSNR are within ±0.5 dB of the paper. The indoor gain over 3DGS is not reproduced (see the comparison table below).

**Setup**
- 30k steps on all 9 `mipnerf360` scenes (images_4 / images_2).
- Decoded with the 3D filter on a black background.
- Runs are in `runs/gof-mipnerf360/gof/`.
- The paper reports group averages only.

**Per scene** (our baseline is `baseline3dgs-mipnerf360-30000`, gsplat 3DGS on the same images):

| scene | PSNR | Δ vs our baseline | #G |
|---|---|---|---|
| bicycle | 25.36 | +0.44 | 5.75M |
| flowers | 21.61 | −0.02 | 3.61M |
| garden | 27.33 | +0.43 | 4.58M |
| stump | 26.90 | +0.67 | 5.04M |
| treehill | 22.45 | +0.18 | 4.73M |
| room | 30.58 | −0.94 | 1.18M |
| counter | 28.60 | −0.25 | 0.83M |
| kitchen | 30.76 | −0.58 | 1.20M |
| bonsai | 31.53 | −0.36 | 1.09M |

**Group averages against the paper**

| group | PSNR ours / paper (Δ) | SSIM ours / paper | LPIPS ours / paper | our 3DGS | gain ours / paper |
|---|---|---|---|---|---|
| outdoor (5) | **24.73 / 24.82 (−0.09)** | 0.743 / 0.750 | 0.236 / 0.202 | 24.39 | +0.34 / +0.18 |
| indoor (4) | **30.36 / 30.79 (−0.43)** | 0.916 / 0.924 | 0.243 / 0.184 | 30.90 | −0.53 / +0.38 |

The paper's 3DGS row is 24.64 outdoor and 30.41 indoor.

**Reading the results**
- **Outdoor:** GOF's gain over 3DGS reproduces, and is in fact larger here than in the paper.
- **Indoor:** the absolute numbers match, but the gain does not. Our gsplat indoor baseline (30.90) is already 0.49 dB above the paper's 3DGS row (30.41), while GOF lands near the paper's own number.
- **Likely cause of the indoor gap:** the indoor deficit is in line with the shared runtime differences (gap list above), which affect gsplat 3DGS and GOF differently. GOF renders with its own rasterizer on a black background, while the baseline uses gsplat on white.
- **No tuning:** nothing was tuned, and no iteration was run. A runtime-level fix would be needed, not a change grounded in the paper.
- **Gaussian count:** bicycle ends at 5.75M, so the raised 20M cap was not needed in the end.
