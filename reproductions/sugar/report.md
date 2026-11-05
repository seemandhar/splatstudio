# SuGaR (plan #56): reproduction report

**Outcome: matched.** R-SuGaR-15K (1M vertices) test PSNR is within ±0.25 dB of the paper on all 3 scenes: garden −0.25, bicycle −0.15, counter −0.19.

## Results
Setup:
- `mipnerf360-1920`: full-resolution images PIL-resized to a 1920-px max side, which is SuGaR's own `load_gs_cameras`.
- llffhold 8.
- 30000 steps, one run per scene.
- Runs are in `runs/sugar-mipnerf360-1920/`.
- Baseline: `baseline3dgs-mipnerf360-1920-30000`, i.e. gsplat 3DGS on the same images.

| scene | PSNR ours / paper (Δ) | SSIM ours / paper | LPIPS ours / paper | #G | our 3DGS | SuGaR − our 3DGS | paper: SuGaR − its 3DGS |
|---|---|---|---|---|---|---|---|
| garden | **25.11 / 25.36 (−0.25)** | 0.768 / 0.775 | 0.255 / 0.220 | 1.60M | 26.62 | −1.51 | −2.05 |
| bicycle | **22.99 / 23.14 (−0.15)** | 0.632 / 0.640 | 0.392 / 0.345 | 1.45M | 24.63 | −1.64 | −2.11 |
| counter | **27.43 / 27.62 (−0.19)** | 0.889 / 0.891 | 0.304 / 0.234 | 1.39M | 28.86 | −1.43 | −1.08 |

Notes:
- **The paper's 3DGS row is at a different resolution.** It is the 3DGS paper's images_4/2 result (27.41 / 25.25 / 28.70), not 1920 px. The baseline deltas therefore mix protocols, and only the absolute SuGaR numbers are like for like.
- **Training time** on shared A6000s was 1.1 / 2.0 / 1.8 h (counter / bicycle / garden).
- **LPIPS** is higher here because the runtime uses the `lpips` package and a white background.

# Build notes

- **Paper:** arXiv 2311.12775.
- **Official code:** github.com/Anttwo/SuGaR, `sdf` pipeline. The README says this pipeline produced the paper's Mip-NeRF 360 results.

## Method
`sugar` is one 30k-step run:

| Steps | Stage |
|---|---|
| 1–7000 | 3DGS |
| 7001–15000 | Coarse SuGaR: entropy loss, prune at 9000, SDF R + R_Norm from 9000 with 16-NN |
| after 15000 | Level-set mesh extraction: λ = 0.3, 10M points, foreground/background Poisson depth 10, 1M triangles each |
| 15001–30000 | Mesh-bound refinement (= R-SuGaR-15K) |

`sugar-smoke` compresses the stage boundaries and is for smoke tests only.

## Implementation
- **Mesh-bound Gaussians** are built inside the components. The renderer derives them from three inputs:
  - learnable vertices, with their own Adam and the official LR schedule;
  - a complex 2D rotation field;
  - 2 in-plane scales.
- **Checkpoint and mesh:** base fields are synced from the mesh every step, so ss-eval needs no mesh (probe: max diff 0). The mesh is saved in `density_state`.
- **Open3D:** Open3D 0.19 and its pure-Python import dependencies live in `~/models/sugar-pylib`. That path is appended to `sys.path` only if `import open3d` fails.

## Checks
- `ss-inspect`: no issues.
- 7 unit tests pass.
- ruff clean.
- ss-verify: `declared_checks_passed` (9/9 probes).

## Smoke run
Garden, images_4, 4000 steps, at most 2000 per stage:

| | Test PSNR / SSIM / LPIPS |
|---|---|
| SuGaR | 22.44 / 0.636 / 0.336 |
| gsplat | 24.19 / 0.735 / 0.331 |

- **Mesh:** 823k vertices and 1.61M faces.
- **Gap:** −1.75 dB. For scale, the paper's R-SuGaR-2K is −3.85 dB against its own 3DGS on garden.
- **Time:** 19 min on a busy A6000.

## Fidelity gaps
1. **Level-set extraction:** uses alpha-normalised gsplat expected depth (α > 0.5) and the K nearest neighbours of each back-projected point. This is a variant: the official `use_gaussian_depth` branch uses un-normalised depth with a −1 background, and the paper used the default PyTorch3D diamond z-buffer. The effect on PSNR is likely small, since refinement re-optimises.
2. **Background:** white. The official code uses black.
3. **View sampling:** with replacement. The official code shuffles once per epoch.
4. **Rasterizer:** gsplat instead of Inria's. A straight-through opacity cap at 0.999 restores the gradients gsplat would otherwise drop.
5. **Metrics:** the SSIM and LPIPS implementations differ.
6. **Gradient safeguard:** a sqrt floor at d = 1 keeps the gradient finite.
7. **Resolution (review-1 major, corrected):** the paper's SuGaR numbers are at max side 1920 (SuGaR `load_gs_cameras`, `cameras.py:16,113-117`), not 1600. We benchmark on `mipnerf360-1920` against a gsplat baseline at the same resolution. In the official pipeline the 7k vanilla-3DGS stage runs at 1600 (`-r -1`) and mesh extraction uses 1600-px camera sizes. Our single run uses 1920 throughout, which is a remaining gap. The paper's 3DGS row is at images_4/2.
8. **Shared host:** CPU threads are capped at 16. This affects speed only.
9. **Adam state at the 7000 boundary:** it carries over from 3DGS into the coarse stage. The official code builds a fresh optimizer.
10. **Colour initialisation:** level points and initial vertex colours come from a white-background render; the official render is black.
11. **Train-only 3DGS stage:** the official `train_full_pipeline.py` trains the 3DGS stage without `--eval`, so test views are seen. The port trains on training views only, which is the correct protocol.
12. **Minor numerics:** `max_z` is detached in the depth background, and the coarse stage runs 8000 iterations instead of 8001.

## Benchmark request
- **Data:** mipnerf360-1920 (corrected from 1600 after review-1), scenes garden, bicycle and counter.
- **Run:** 30000 steps with `--recipe 3dgs --num-gaussians 10000000 --max-gaussians 12000000`.
- **Paper per-scene PSNR** (supplementary, 1M vertices): 25.36 / 23.14 / 27.62. That is −2.05 / −2.11 / −1.08 dB against the paper's 3DGS row.
- **Comparison:** set our gap to our own baseline against the paper's gap.

*Report saved by the orchestrator; the build subagent could not write it.*
