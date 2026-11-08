# ConeGS gap analysis 1 (garden), 2026-11-07

Bench `runs/conegs-mipnerf360-conegs` (30k steps, 1M budget, seed 0).

| scene | ours | paper ConeGS-1M | abs gap | our gsplat (unbudgeted) | 3DGS paper | baseline offset | delta gap* |
|---|---|---|---|---|---|---|---|
| garden | 26.35 | 27.42 | −1.07 | 26.89 (4.76M) | 27.41 | −0.52 | −0.55 |
| room | 32.15 | 32.30 | −0.15 | 31.52 (1.73M) | 30.63 | +0.89 | −1.04 |

\*delta = (ours − our gsplat) − (paper ConeGS − Inria 3DGS paper). The paper's own 3DGS row is
budgeted at 1M (garden 26.70, room 31.62), so it is not comparable to our unbudgeted baseline.
Room's match therefore owes a lot to its favourable runtime offset (the Inria room number 30.63 is
low). Garden has a −0.52 runtime/protocol offset (calibration report) on top of the method gap.
Our protocol matches the official one: images_4 JPEGs, llffhold-8, 7-scene set. The
`-pilr` gsplat baseline on garden is 27.35, but the official ConeGS loads images_4.

## Evidence (train views only)
- **iNGP is healthy.** Ray PSNR on garden reaches 25.9 at 20k iterations; full train-view PSNR
  is 25.8 (room 31.5). The median depth agrees with the gsplat baseline's depth on train views
  (median relative error 1.3%), and only 0.8% of pixels are ≥20% too short (floaters). Paper
  ablations (a)/(b) show iNGP length barely matters. Not the cause.
- **Init distribution.** 46% of the 1M init points lie in the contracted background (nerf radius
  > 1), compared with 57% of SfM points. Background points start at a median 3-NN scale of 0.031
  versus 0.0074 in the foreground. This is expected for an outdoor scene, and the paper's
  ablation (g) puts the whole init at ≤0.3 dB.
- **Densification dynamics differ.** Garden prunes and refills 4k–14k per 100 steps (2.19M total
  over training); room does 19k–170k (29.2M). The paper's appendix figure (garden, 1M budget,
  data parsed from `gaussians_number_analysis.pgf`) shows prune = add ≈ 10k at 2.1k, 18k at
  4.2k, 22k at 6.2k, 17.6k at 10.3k and 6k at 23k. Ours is about 0.6× that.
- **Background is outdoor-specific.** With the runtime's white background, 15.5% of garden
  train pixels have alpha < 0.99 in the final model (room 2.8%). The white background fills
  bright sky/background cheaply, so fewer Gaussians fade and get pruned and recycled, and the
  error map is built on the white composite. The official code uses black
  (`white_background: false`).
- **7k-step diagnostic, same init and seed, train views every 10th:**

  | background | prune at 2k / 4k / 6k | train-view PSNR |
  |---|---|---|
  | white | 5.6k / 11.3k / 14.5k | 27.50 |
  | black (`conegs-blackbg`) | 9.4k / 14.4k / 19.0k | 27.90 (+0.40) |

  The black-background prune/refill counts sit much closer to the paper's figure.
- **Ruled out:**
  - Opacity saturation: only 0.02% of Gaussians have opacity > 0.999, so the gsplat
    backward-drop pitfall does not apply.
  - Opacity penalty: code-exact (signed mean, 2e-4; probe).
  - SH: all bands from step 1.
  - LRs: identical to the code.
  - Means-LR extent: 4.92, the 3DGS getNerfppNorm of the train views (same as official).
  - Ray count: 201/step, matching the code formula. The paper's figure accumulates about 267
    per step, an unexplained small difference; it does not limit refill, since added = pruned.
  - Unused config keys in the code: checked.
- **fp32 vs fp16 iNGP:** not testable without tcnn. Given the healthy depth and colour above it
  is unlikely to matter much.

## Recommendation (iteration 1, post-hoc)
Switch to the official black background: method `conegs-blackbg`, which renders and evaluates
over black and so also changes the error map. It is grounded in the official config. Expected
effect: positive on garden; the train-view gain at 7k is +0.4 dB, and the size of the test effect
at 30k is unknown, so ≥0.5 dB is not guaranteed. The remaining ~0.5 dB on garden is the runtime
offset already seen for the gsplat baseline.
