# DET-GS (arXiv 2508.04099) — reproduction report

**Outcome: not matched.** Plugin `det-gs` (`plugin/ss_det_gs`). No official code exists; the paper says it will be released after publication.

## Results (Mip-NeRF 360, 30k, llffhold8, test split; `runs/det-gs-mipnerf360`)
| scene | ours | paper | gap | our gsplat | ours − gsplat | paper − its 3DGS |
|---|---|---|---|---|---|---|
| garden (images_4) | 26.51 | 28.09 | −1.58 | 26.89 | −0.38 | +0.36 |
| counter (images_2) | 28.42 | 30.07 | −1.65 | 28.85 | −0.43 | +0.95 |
Final Gaussian counts: 4.45M (garden) and 1.13M (counter). Training took 47 and 49 min. The paper's 3DGS row
(27.73 / 29.12) is above the 3DGS paper (27.41 / 28.70).

## Implementation
- **Base:** 3DGS. The gsplat renderer is unchanged apart from passing the camera through. 3DGS density.
  Photometric loss 0.8·L1 + 0.2·D-SSIM.
- **C1, hierarchical depth (Eq. 4–7):**
  - Prior: Depth Anything V2-Large, precomputed offline into dataset copy `mipnerf360-detgs` (hard links, same protocol_id).
  - Hard depth with ω=0.95; gradient reaches the means only.
  - Patch (γ=0.1) and image-level (η=1) normalisation; patch size U[5,20]; error-tolerant L2 with margin 0.2; every 5 steps.
- **C2, edge-aware depth smoothness (Eq. 8–11):** Canny 20/200 non-edge mask, cross 3×3 masked mean, β=0.1.
- **C3, RGB-guided TV (Eq. 12–15):** τ_edge=1e-2, τ_smooth=1e-4, ϕ=0.8.
- **Probes:** each contribution has a probe (independent loops, mask/schedule/gradient-leak checks). ss-verify passes.

## Deviations and assumptions (fidelity gaps)
1. **Unspecified values, taken from DNGaussian's code** (this is DNG's Global-Local Depth Normalization):
   - δ = 1e-2·std of the map, ω = 0.95.
   - Error tolerance 0.2 (DNG's code default; DNG also uses 0.01 and 2.5e-4).
   - λ = 3DGS's 0.2.
   - Depth active for the whole run.
2. **Depth space:** rendered camera-z vs −disparity (DNG-consistent). The paper uses ray distance ‖μ−o‖.
   A `space: disparity` option exists.
3. **Optimiser step:** the depth loss is summed into the shared Adam step. DNGaussian takes a separate step.
4. **TV normalisation:** averaged over channels.
5. **Edge-loss depth:** uses the soft depth of Eq. 3 in COLMAP units. Its gradient is only ~1% of the photometric one.
6. **Logging:** logged det-gs-depth values come from depth steps only, so they overstate the average (≈ logged/5).
7. **Runtime differences:** random-with-replacement view sampling, white background, valid-SSIM.

## Gap analysis (`gap-analysis-1.md`)
Garden, 7k steps, 20 held-out *training* views, one seed per variant:
- gsplat 25.18; full det-gs 24.82.
- Removing one loss: no-depth 25.08, no-TV 24.99, no-edge 25.03.
- Disparity space 24.91; DNG-style separate depth step 24.95.

No variant reaches the baseline, and every regulariser is neutral-to-harmful in our runtime. The DA-V2
normalised depth loss stays at 0.25–0.43 for the whole run (the prior conflicts with the photometric geometry), and
its mean-gradient is 3.6–5.1× the photometric one. No paper- or DNGaussian-grounded change is likely to close
≥0.5 dB, so no iteration was run. The paper's gains probably depend on unreported details of the unreleased code.

## Weak / failed runs kept
- Smoke runs: smoke-1 (tolerance 2.5e-4, 23.01), smoke-2 (0.2, 22.85), smoke-500.
- All 7 diagnostic runs are kept in `evidence/diag/`.
