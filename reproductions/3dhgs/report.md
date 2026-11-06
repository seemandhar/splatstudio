# 3D-HGS (plan #78) — build report

**Status: ready for benchmark** (not benchmarked). Paper: arXiv 2406.02720 (CVPR 2025). Official
code lihaolin88/3DHGS @ dde1bd5; its rasterizer is vendored unmodified (Gaussian-Splatting licence).

## Port
- Plugin `plugin/` (`splatstudio-method-3dhgs`, method `3dhgs`; also `3dhgs-db-paper` with normal LR 0.3).
- State: `opacities` = α1, field `opacity2` = α2 (lr 0.05), `hgs_normal` (raw, init 0.6, lr 0.003).
- Renderer: official CUDA kernel, JIT-built; black background; eval renders use the official
  test path (half-aware tile rectangles, "efficient rasterizer").
- Density: official densify_and_prune for two opacities (until 20k, prune max-half < 0.01, reset 0.02,
  clone 1−√(1−α) on the copy only, opacity LR ÷1.4 per 5k). Loss: 0.8 L1 + 0.2 D-SSIM.

## Evidence
`evidence/verify.json`: all 8 probes pass. α1=α2 render = independent float64 3DGS compositor
(2e-7) and ignores the normal; single-Gaussian half-plane alpha = numerical z-quadrature (1.5e-3);
backward = float64 autograd (rel ≤1e-6, all fields) for opacity ≤0.45.
Smoke (garden, 2k steps): 3dhgs 23.55 dB / 0.868M vs gsplat 23.03 dB / 0.845M.

## Benchmark request
mipnerf360 garden + counter (per-scene, supp. tab:all), 30k, vs `baseline3dgs-mipnerf360-30000`.
Paper: garden 27.50 (3DGS 27.30, +0.20), counter 29.84 (28.70, +1.14). Ours baseline: 26.89 / 28.85.

## Deviations / quirks (all recorded)
1. Official quirks kept: camera-space normal gets the view translation added; forward-only power<−4.8
   skip → ~1% gradient mismatch for high-opacity tails; normal LR 0.003 everywhere (paper says 0.3 on DB),
   no normal LR decay (paper says ÷1.4).
2. Clone rescale computed stably and logit clamped to ±40 (official float32 gives ±inf, inert).
3. Runtime: optimizer step before density (LR decay one step late), SH switch one step late,
   sampling with replacement, SSIM valid conv. Screen-size prune inert as in official code.

## Review-1 resolution (orchestrator)
The review approved the port with no critical or major issues. The benchmark is launched on garden and counter (`mipnerf360`).

Corrections and additions from the review:
- **Backward accuracy:** errors against float64 autograd reach 5.1e-4 (means), under a gate of 1e-3. The earlier claim of ≤1e-6 was wrong.
- **Densification window:** the code densifies until step 20000. The paper text implies the 3DGS default of 15000.
- **Optimizer step on densify iterations:** the official code applies no optimizer update on its ~195 densification iterations; the runtime does.
- **Official scorer:** it scores unclamped float renders, while the runtime clamps. The expected effect is under 0.05 dB.

## Benchmark result (3dhgs-mipnerf360, 30k, images_4/2) — MATCHED
| scene | ours | paper | Δ | gsplat baseline | paper 3DGS | gain ours / paper |
|---|---|---|---|---|---|---|
| garden | 27.62 / 0.867 / 0.119 | 27.50 | +0.12 | 26.89 | 27.30 | +0.73 / +0.20 |
| counter | 29.73 / 0.914 / 0.252 | 29.84 | −0.11 | 28.85 | 28.70 | +0.89 / +1.14 |
- Both scenes are within ±0.5 dB.
- The garden run OOMed once because other users shared the GPU; the retry completed.
