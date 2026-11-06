# Octree-GS — reproduction report (build stage; benchmark pending)

**Paper:** *Octree-GS: Towards Consistent Real-time Rendering with LOD-Structured 3D Gaussians* (TPAMI 2025,
arXiv 2403.17898). **Official code:** city-super/Octree-GS @ `9db64fe` (read only, never run).
**Method:** `octree-gs` (Our-Scaffold-GS variant), plugin `plugin/` (`ss_octree_gs`).

## Status
Built and verified. `ss-inspect` reports no issues, the 9 unit tests pass and ruff is clean. `ss-verify` passes all
9 probes, including octree-init, lod-selection, progressive-training, anchor-growing and blob-eval.
Smoke run: native garden, 2000 steps. The method reaches PSNR 23.55, SSIM 0.737, LPIPS 0.267. The gsplat baseline
with the same arguments reaches 23.00, 0.643, 0.418. The run has 411k anchors (levels 0–5: 55.6k / 106k / 142k /
99k / 8.4k / 0) and renders 985k neural Gaussians per view. `ss-eval` in a fresh process equals the in-process
validation.

## Benchmark request
`bench_request.json`: mipnerf360-1600 (the official `-r -1` protocol), scenes garden and counter, 40k steps.
The paper reports per-scene results for Our-Scaffold-GS:
- garden: PSNR 27.67, 1475k Gaussians per view
- counter: PSNR 30.19, 409k Gaussians per view

The paper's own 3D-GS row gives garden 27.45 and counter 29.22. Our gsplat baseline on the same images (30k steps)
gives garden 26.93 and counter 29.04.

## Design
Anchors are the native population. The per-anchor fields are feat, offset, scaling, octree level and LOD bias ΔL.
The Scaffold MLPs and the octree scalars live in a density-owned bank. They are packed into `ogs_blob` every step,
so that ss-eval can render without the bank. This is the HAC / ContextGS design. Scaffold's `visible_filter` is
re-implemented in torch.

simple-knn is vendored unmodified (Inria licence) and JIT-built for the initial anchor scaling. The octree is built
in `set_training_cameras`, because it needs the training cameras.

## Fidelity gaps (see plan.md)
1. Rendering uses gsplat (with a 0.999 opacity straight-through cap) instead of the Scaffold fork of the Inria
   rasterizer.
2. The background is white; the official runs use black.
3. Training views are sampled randomly with replacement, and the loss uses valid-convolution SSIM.
4. The runtime steps Adam before the density update.
5. The global state is transported through the blob.
6. Only the mip360 preset is registered.
7. Where the paper and the code disagree, the port follows the code:
   - LOD selection uses `round` without opacity blending.
   - The visibility threshold τ_v is adaptive (the mean view frequency).
   - Next-level growing and ΔL updates stay active during progressive training. The cause is the official
     `~self.progressive` quirk.
   - `update_until` is 25000.
8. Our 30k-step baseline is compared with the paper's 40k-step rows.

## Review-1 resolution (orchestrator)
**Review outcome:** approved, with 0 critical and 0 major issues. The benchmark is launched: garden and counter on `mipnerf360-1600`, 40k steps.

**Secondary reference.** The official README reports these numbers for the released code and checkpoints, close to the paper's table:

| scene | PSNR | Gaussians per view |
|---|---|---|
| garden | 27.69 | 1344k |
| counter | 30.30 | 457k |

**Judging the result.**
- Our 30k gsplat baseline sits 0.52 dB below the paper's 40k 3D-GS row on garden.
- Garden and counter will therefore be reported both as absolute PSNR against the paper and as gain over our baseline.
- Adding the paper's gain over its own baseline (+0.22 on garden, +0.97 on counter) to our baseline gives implied targets of about 27.15 (garden) and 30.01 (counter).

**Corrected wording.** The initialisation uses the vendored simple-knn `distCUDA2`, not an exact 3-NN.

## Benchmark result (octree-gs-1600, 40k steps, mipnerf360-1600) — MATCHED
| scene | ours PSNR | paper (Octree-Scaffold) | Δ | gsplat baseline (30k) | paper 3DGS (40k) | #GS/view ours vs paper |
|---|---|---|---|---|---|---|
| garden | 27.36 | 27.67 | −0.31 | 26.93 | 27.45 | 1350k vs 1475k |
| counter | 29.76 | 30.19 | −0.43 | 29.04 | 29.22 | 306k vs 409k |
Both scenes are within ±0.5 dB. Gain over our baseline: +0.43 / +0.72 dB. The paper's gain over its 3DGS row is +0.22 / +0.97 (that baseline was trained for 40k steps, ours for 30k).
Anchors: garden 720k (6 levels), counter 212k (5 levels). FPS: 13.6 / 21.6 (shared GPU, not comparable to the paper's FPS).
Counts: runs/octree-gs-1600/octree-gs/*/octree_gs_counts.json. Summary: results/summary.md.
