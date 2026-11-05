# PGSR (#58): build report (benchmark pending)

**Status:** built, verified and smoke-tested. No paper numbers are claimed yet.

## Plugin
`ss_pgsr` provides two methods:
- `pgsr`: `run_mip360.py` settings (exposure off, abs-grad 0.0002).
- `pgsr-tnt`: `run_tnt.py` settings.

**Official code:** zju3dv/PGSR `de24f1a`. The plane rasterizer is vendored unmodified (Inria and ZJU non-commercial licences) and JIT-built.

## Components

| Component | Role and settings |
|---|---|
| `pgsr-flatten` | L_s, λ1 100 |
| `pgsr-plane` | Plane depth, normal and distance |
| `pgsr-single-view` | λ2 0.015, active for step > 7000 |
| `pgsr-mv-geo` | λ4 0.03, φ < 1, w = exp(−φ) |
| `pgsr-mv-ncc` | λ3 0.15, 7×7 patches, 102,400 pixels |
| `pgsr-rgb` | Exposure compensation, gated on SSIM and step > 1000 |
| `pgsr-density` | See below |

`pgsr-density` settings:
- `percent_dense` 0.001.
- AbsGS split for radius > 20, at most 50k per step.
- 6M Gaussian cap; cull below opacity 0.005.
- World-size prune after step 3000. The screen-size prune is a no-op, as upstream.
- Trim Gaussians observed in fewer than 2 training views, every 1000 steps.
- Opacity reset every 3000 steps.

**Neighbour graph:** at most 8 neighbours per view, with angle under 30° and distance between 0.01 and 1.5, sorted by distance (the code's order). It is built from the training split only.

## Verification
- `ss-inspect`: no issues.
- 11 unit tests pass; ruff is clean.
- `ss-verify` passes for both methods (10 probes).
- The official kernel's plane depth matches an analytic oblique plane to within 7e-7.

## Smoke test (garden, 2k steps)

| run | PSNR / SSIM / LPIPS | Gaussians | time |
|---|---|---|---|
| `pgsr` | 22.91 / 0.671 / 0.339 | 1.95M | 101 s |
| early regularisers (from step 500) | 22.42 / 0.664 / 0.347 | 1.82M | 526 s |
| gsplat baseline | 22.99 / 0.642 / 0.419 | 0.84M | 203 s |

Once the multi-view losses are active, each step costs about 5× more.

## Benchmark request
- **Target:** mipnerf360, all 9 scenes, 30k steps. Mip-NeRF 360 has averages only:

  | | Indoor | Outdoor | All |
  |---|---|---|---|
  | Paper PSNR | 30.36 | 24.76 | 27.25 |
  | Paper gain over 3DGS | −0.63 | +0.52 | — |

- **Protocol change by the orchestrator:** the official code PIL-resizes full-resolution images (`-r 4` outdoor, `-r 2` indoor). The benchmark therefore uses `mipnerf360-pilr` for all 9 scenes, against a gsplat baseline on the same images (`baseline3dgs-mipnerf360-pilr-30000`). This replaces images_4/2 JPGs.

## Fidelity gaps
1. **Update order and sampling:** the runtime takes the optimizer step before densifying, and samples views with replacement.
2. **`ncc_scale`:** values other than 1 are unsupported. `pgsr-tnt` uses 1.0 instead of 0.5; this has no effect on Mip-NeRF 360.
3. **Geometry evaluation:** no TSDF mesh, DTU Chamfer or TnT F1. Local DTU data is only the FatesGS 3-view release.
4. **Seed and metrics:** a different seed, and the SplatStudio SSIM and LPIPS implementations.
5. **Adam on replaced parameters:** the official code skips the Adam step for parameters replaced on densify, trim and reset iterations; the runtime steps them.
6. **SH schedule:** the SH degree rises one step later than in the official code.
7. **Photometric loss:** `pgsr-rgb` replaces the base photometric loss.
8. **Test images:** the density reloads every frame into host RAM and keeps only the training split. Test images are never used, but they are briefly loaded (about 5.7 GB of host RAM on bonsai).

*Report saved by the orchestrator; the build subagent could not write it.*
