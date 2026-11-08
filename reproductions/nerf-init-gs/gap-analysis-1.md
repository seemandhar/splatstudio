# Gap analysis 1 — nerf-init-gs (30k benchmark `runs/nerf-init-gs-mipnerf360`)

## Gap table (test PSNR, 30k)
| scene | ours | paper | abs gap | our gsplat | paper 3DGS re-run | baseline offset | delta gap (method) |
|---|---|---|---|---|---|---|---|
| garden | 25.98 | 27.37 | −1.39 | 26.89 | 27.29 | −0.40 | **−0.99** |
| room | 30.79 | 31.63 | −0.84 | 31.52 | 31.43 | +0.09 | **−0.93** |

**Classification: method gap**, about −1 dB on both scenes. The paper improves on its own 3DGS by
+0.08 and +0.20 dB; we are −0.91 and −0.73 dB below ours.

## Evidence
### 1. Train views at 30k (`tools/train_view_eval.py`, every 4th training view)
| run | train PSNR | Gaussians | scale > 0.1 |
|---|---|---|---|
| garden nerf-init-gs | 30.53 | 4.55M | 3.9% |
| garden gsplat | 29.96 | 4.76M | 3.0% |
| room nerf-init-gs | 34.29 | 3.14M | 12.0% |
| room gsplat | 34.18 | 1.73M | 3.6% |

Training views fit as well as or better than the baseline, while test views are worse. The deficit
is therefore a **generalisation/geometry gap** (floaters, large Gaussians), not under-fitting.

### 2. Loss magnitudes and Gaussian growth (30k `training.jsonl`)
| step | garden photo / λ·depth | room photo / λ·depth | room G (ours / base) |
|---|---|---|---|
| 1 | 0.38 / 5.68 | 0.35 / 7.13 | 0.5M / 0.11M |
| 1000 | 0.13 / 0.079 | 0.06 / 0.54 | 1.05M / 0.20M |
| 3000 | 0.09 / 0.011 | 0.04 / 0.031 | 3.96M / 0.71M |
| 7000 | — / 0.0003 | — / 0.001 | 3.30M / 1.22M |

- The depth term dominates the photometric term for the first ~1k (garden) and ~2k (room) steps.
- In room, the mean depth error at step 1000 is 1.7 world units. Room NeRF depths reach 350–470
  units through the windows.
- Room densifies explosively: 5.6× the baseline at 3k. This is the "excessive number of Gaussians"
  that paper Sec. 3.4 warns about.

### 3. Train-views-only holdout ablation, 7k steps
The holdout copies keep the training views only (the llffhold8 test views are removed), and every
8th training view is held out. The NeRF and the init were rebuilt from the inner training views.
Scripts: `tools/make_holdout.py`, `tools/diag.sh`; results in `evidence/diag/`.

| held-out PSNR / SSIM / LPIPS | garden | room |
|---|---|---|
| gsplat (SfM) | **25.92** / .801 / .257 | 29.64 / .900 / .367 |
| nerf-init-gs-nodepth | 22.90 / .793 / .253 | **29.99** / .902 / .354 |
| nerf-init-gs (world units, as benchmarked) | 25.16 / .787 / .273 | 29.59 / .894 / .374 |
| nerf-init-gs-nerfunits (depth L1 × nerfacc scale ≈ 0.23) | 24.79 / .787 / .273 | 29.85 / .899 / .364 |

- In room, the ordering matches the paper (no-depth > depth ≥ SfM).
- In garden, the NeRF init alone loses 3 dB on held-out views, although training PSNR is higher
  (26.53 vs 26.24). Depth distillation recovers about 2.3 dB of that, but not all of it.

### 4. Init point quality (`tools/init_stats.py`)
| cloud | points within 1 world unit of a training camera | 1st percentile of nearest-camera distance |
|---|---|---|
| garden NeRF init | 2.7% (13.5k points) | 0.57 |
| garden SfM | 0.009% | 1.65 |
| room NeRF init | 2.2% | 0.72 |
| room SfM | 0.4% | 1.26 |

- These near-camera points are NeRF floaters. In novel views they become floaters, and the
  3DGS pruning does not remove them because they help the training views.
- The 5k fp32 iNGP reaches 24.1 dB on garden training rays and 22–25 dB on test views (review m5).
- The paper says its 5k NeRF takes about 30 s (tcnn fp16); ours takes 171 s. The paper reports no
  NeRF PSNR, so the quality gap cannot be quantified.

### 5. D_NeRF units (review m3)
- **Paper:** no units are stated. Points are "loaded in exactly the same way as COLMAP point
  clouds", so the authors must have mapped NeRF samples into the COLMAP frame.
- **nerfacc 360 example:** `similarity_from_cameras` rescales poses, so nerfacc depth is in units
  of 0.249 (garden) or 0.227 (room) × world.
- **Author fork `Youlenda/gaussian-splatting@3dgs_rf`:** it uses TensoRF on Blender lego (shared
  coordinates) and has no depth loss and no scale convention (`train_rfgs_depth.py` only computes
  `depth_gs`). Nothing to ground on.
- **Ablation (table above):** nerfacc units give +0.26 dB on room and −0.37 dB on garden. That is
  not a consistent fix.

## Ranked hypotheses
1. **NeRF init floaters near the cameras (weaker NeRF than tcnn's).**
   - For: garden no-depth −3 dB held-out with higher training PSNR; 2.7% near-camera points; NeRF
     test PSNR 22–25 dB.
   - Against: room no-depth is fine.
   - Effect: most of the garden gap. Fix: a faster/better iNGP (tcnn fp16, unavailable in the
     shared env) or more NeRF iterations. Paper Table 2 shows 10k/30k barely change its numbers,
     so the paper's own NeRF must already be cleaner at 5k. No paper-grounded parameter change
     exists.
2. **Depth term scale and densification blow-up in room.**
   - For: 5.6× Gaussians and 12% large Gaussians; held-out depth −0.4 dB vs no-depth.
   - The paper's own room result also drops with depth (32.11 → 31.63).
   - The units question is unresolved and the nerfunits ablation is mixed, so this is not a
     grounded fix.
3. **Centre-z vs ray-conditional depth (gap 1).** Small, because λ is ~0 after 3–5k steps.

## Recommendation
No concrete paper- or nerfacc-grounded fix is expected to close ≥ 0.5 dB on both scenes. Report
the result as **not matched**.

The `nerf-init-gs-nerfunits` variant is kept in the plugin as a labelled post-hoc diagnostic. It
is not recommended for a benchmark.
