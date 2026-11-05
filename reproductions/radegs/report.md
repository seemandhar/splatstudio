# RaDe-GS (plan #59) — build report

**Status:** plugin built and verified; Mip-NeRF 360 benchmark pending (`bench_request.json`).

- Method `radegs` (`reproductions/radegs/plugin`, module `ss_radegs`). Official code
  HKUST-SAIL/RaDe-GS **pinned at 2d4bc08 (2024-06-21)**, the last commit of the arXiv-v2 formulation
  (centre-ray-cosine depth, distortion + normal-consistency). Later commits change the formulation
  (Jul 2024: per-pixel cosine, distortion removed), fix a backward bug (5f5cd3b, Dec 2025) and add
  PGSR-style multi-view regularisation (c7c56cc, 2026); none of them is used. Rasterizer vendored
  unmodified in `ss_radegs/csrc` (Inria/MPII licence, glm 5c46b9c0), JIT-built via tools/ssenv (~70 s).
- Contributions → components: rasterized depth (C1) + normal (C2) = `radegs-raster` renderer;
  L_d/L_n (C3) = `radegs-reg` (100·edge·dist/α² + 0.05·(0.4/0.6 expected/median normal error), from
  15k); GOF densification + Mip 3D filter = `radegs-density` (official code is GOF's verbatim; the
  screen-size prune is a no-op as in the official code and is not applied; world-size prune kept).
- Probes (all pass, `evidence/verify.json`): kernel depth slope = cos θc·t* of an exact float64
  ray/Gaussian maximum (rel. err 1e-7; constant centre depth err 1.0, per-pixel cosine 0.06 fail);
  normal = intersection-surface normal (1−cos 1e-14); all 6 outputs and gradients vs an independent
  differentiable float64 renderer (≤3e-5); distortion formula + detached weights; loss schedule;
  GOF densify metric/recipe.
- **Official quirk kept:** the kernel's backward of the normal map w.r.t. means is inexact (rel. err
  ≈3; FD of the kernel forward matches the reference) → affects L_n gradients after 15k only.
- Principal point: kernel assumes cx=W/2, cy=H/2; local mipnerf360 conversions have cx = W/2 exactly;
  renderer refuses other cameras.
- Smoke (garden, 2000 steps): radegs 22.19 / 0.651 / 0.374 (1.60 M G, 569 s) vs gsplat 22.99 / 0.642 /
  0.419 (0.84 M G, 145 s) — same pattern as the GOF port (22.25), explained there by the official 30k
  xyz-LR schedule vs the baseline's decay over --steps. Regularisers forced on at 1000
  (`evidence/smoke_reg.py`): finite, L_reg 0.029 → 0.010, PSNR 22.04.
- Benchmark: paper gives outdoor/indoor averages only (Table 3: 25.17 / 30.74; its 3DGS row 24.64 /
  30.41) → all 9 mipnerf360 scenes at 30k, images_4/images_2 (GOF protocol; RaDe-GS gives no script).

## Fidelity gaps
1. No mesh extraction / DTU Chamfer / TnT F1 (the paper's main metrics). Full-view DTU (2DGS
   preprocessing) not available locally; TSDF fusion + evaluation not ported.
2. Decoupled appearance (DTU/TnT only) not ported.
3. Mip-NeRF 360 resolution protocol not stated by RaDe-GS (later README default = 1600-px width).
4. Paper vs code resolved to code: w_n 0.05 (paper text 5), NDC/edge/α² distortion, 0.4/0.6 normal mix.
5. Runtime: views sampled with replacement, Adam step before densification, SH degree one step
   later, 3D-filter coefficient in log space, evaluator SSIM/LPIPS shared with the baseline.
6. Training ~4× slower than gsplat; ~1.9× more Gaussians → bench uses --max-gaussians 20M, ≥36 GB GPUs.

## Review-1 resolution (orchestrator)
- Approved (0 critical/major).
- **Resolution protocol confirmed: `mipnerf360` (images_4 outdoor / images_2 indoor).**
  - The paper's Table 3 "3D GS" row (24.64 / 30.41) equals the 3DGS paper's per-scene images_4/2 PSNR averages (24.643 / 30.407).
  - GOF's `run_mipnerf360.py` runs `-i images_{4|2}` without `-r`, and no rescale triggers below 1600 px.
  - Fidelity gap 3 (unstated protocol) is downgraded accordingly.
- Additional documented items:
  1. Opacity reset at 3000/6000/9000/12000 uses the 3D filter from before densification (official order: densify → recompute filter → reset). Negligible.
  2. The pinned kernel carries the upstream `dL_dconic`/`conic_opacity` argument bug fixed later in 5f5cd3b (`rasterizer_impl.cu:530`). It is kept as-is, because the paper's numbers came from this kernel; it is nearly harmless with kernel_size 0.
  3. The normal→means backward quirk (relative error ~4 vs finite difference) is kept unmodified for the same reason.
