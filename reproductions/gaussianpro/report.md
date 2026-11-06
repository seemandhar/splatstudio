# GaussianPro (plan #79): build report

**Status:** the plugin is built and all probes pass. The 2000-step smoke run is still in progress, so the benchmark is pending.

- **Plugin:** `plugin/` (`ss_gaussianpro`), method `gaussianpro`, against official GaussianPro.
- **Vendored code:** the propagation CUDA (`submodules/Propagation`, MIT) and simple-knn are vendored unmodified and JIT-built.
- **Renderer:** a second gsplat pass renders camera-space depth and the shortest-axis normal.
- **Density:**
  - Baseline clone, split, prune and opacity reset.
  - Plus the official PatchMatch propagation every 50 steps, between steps 1000 and 12000.
  - Each view uses 4 neighbours, chosen by shared COLMAP points.
  - Propagated depth is kept only where it is multi-view consistent: 2 px, 1 %, at least 2 views.
  - New Gaussians are inserted where the abs-rel depth error exceeds a threshold that anneals from 1.0 to 0.8.
- **Losses:**
  - Normal loss, 0.001 + 0.001.
  - Flatten loss, weight 100.
  - Opacity-masked photometric loss.

## Probes
`ss-verify` reports `declared_checks_passed`.
- **Synthetic plane:** depth median error is 0.05 % and normal error 1.1°, starting from ±20 % noise.
- **Plane with a hole:** all 25 inserted Gaussians land in the hole, and none are inserted when there is no hole.

## Benchmark request
- **Scenes:** `mipnerf360` room and garden, 30k steps, against `baseline3dgs-mipnerf360-30000`.
- **Paper targets:**
  - room 31.98, against its 3DGS at 31.71 (+0.27);
  - garden 28.70, against its 3DGS at 28.77 (−0.07).
- **How to read the comparison:** the paper's 3DGS was retrained on the authors' own COLMAP re-run, which is not released. Their garden 3DGS scores 28.77, ours 26.89. Compare the gain over the baseline, not absolute PSNR.

## Open concerns
- **Normal loss:** it logged exactly 0.0 in every row up to step 1250 of the smoke run.
- **Speed:** training is much slower than gsplat.

## Fidelity gaps
1. **Rasterizer:** gsplat with a straight-through 0.999 opacity cap, instead of the Inria rasterizer. White background and valid-convolution SSIM.
2. **Propagation timing:** it runs at the end of the step rather than the start.
3. **Neighbour graph:** regenerated from shared COLMAP points, because the authors' pair files were not released.
4. **Propagation window end:** 12000, the code default.
5. **Official-code quirks kept:**
   - the kernel runs 6 iterations (the paper says 3);
   - normals are initialised randomly;
   - new Gaussians are inserted fully opaque (logit 20);
   - colours for new Gaussians come from the previous step's image;
   - depth caches go stale;
   - every insertion resets the densification statistics.

*Saved by the orchestrator from the build hand-back.*

## Review-1 resolution (orchestrator)
Approved, with 0 critical and 0 major issues. The benchmark is launched on room and garden (`mipnerf360`).

**Normal loss logged as 0.0: a logging artifact.**
- The port propagates at the end of a step, and the logged rows land on propagation steps.
- A direct 600-step check showed every render of a propagated view had a non-zero normal loss (about 0.0025).

**Smoke deficit (−1.52 dB at 2k steps): explained by official mechanisms.**
- The main cause is the opacity-masked photometric loss before step 1000 (`--depth_loss` in demo.sh).
- Gaussians at step 1000 by configuration:

  | configuration | Gaussians |
  |---|---|
  | default | 218k |
  | without the opacity mask | 327k |
  | without the flatten loss | 222k |
  | gsplat baseline | 292k |

- The flatten loss and the statistics resets match the official code.
- Judge the 30k gain over our baseline; the paper reports +0.27 on room and −0.07 on garden.

**Choices recorded.**
- The propagation window ends at 12000, the code default. demo.sh uses 6000 and the paper says nothing.
- demo.sh's learning-rate changes are not used, because the paper states it uses the 3DGS hyper-parameters.

**Speed.**
- Propagation costs about 0.93 s per call, inherent to the method.
- The masked photometric loss adds 10–25 ms per step through host syncs. This can be optimised later; the maths would not change.

## Benchmark result (gaussianpro-mipnerf360, 30k, images_4/2) — MATCHED (room absolute; garden on the pre-registered gain criterion)
| scene | ours | paper | Δ abs | our baseline | paper 3DGS | gain ours / paper |
|---|---|---|---|---|---|---|
| room | 31.72 / 0.918 / 0.295 | 31.98 / 0.927 / 0.192 | −0.26 | 31.52 | 31.71 | +0.20 / +0.27 |
| garden | 26.82 / 0.860 / 0.128 | 28.70 / 0.890 / 0.107 | **−1.88** | 26.89 | 28.77 | −0.07 / −0.07 |

- **Room:** within ±0.5 dB absolute.
- **Garden:** not comparable in absolute terms. The paper's garden rows (3DGS 28.77 and Ours 28.70) were trained on the authors' unreleased re-run COLMAP, 1.9 dB above any 3DGS on the public data. This was recorded in bench_request.json before the run, with the gain as the criterion. Our garden gain equals the paper's exactly (−0.07).
- **Caveat:** the absolute garden number is a −1.88 dB miss, kept visible here; the verdict relies on the gain criterion for that scene.
