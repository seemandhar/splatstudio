# AGS-Mesh (plan #64): build report

**Status:** built and verified. The 30k benchmark on all 6 MuSHRoom rooms is pending (`bench_request.json`).

- **Paper:** arXiv 2411.19271 (3DV 2025).
- **Official code:** the `ags-mesh` method in maturk/dn-splatter main @ 97588b4, i.e. the paper's "Splatfacto + Ours" row.
- **Plugin:** `plugin/` (`ss_agsmesh`), with methods `ags-mesh` and `ags-mesh-rgb`. No custom CUDA.
  - **Reused from the DN-Splatter port** (copied; the DN-Splatter files are unmodified):
    - `density.py`;
    - `sidecar.py`, extended with the DNC mask;
    - `renderer.py`, extended with surface normals;
    - four probes: density, sidecar-init, gaussian-normals and scale-loss;
    - `tools/prepare_mushroom.py`.
  - **New:** `dnc.py`, `losses.py`, `probes_ags.py`, `tools/prepare_dnc.py`, `tools/prepare_rooms.sh`, `tools/smoke_compressed.py`, `tools/dnc_stats.py`.

## Contributions

| contribution | component | probe |
|---|---|---|
| C1 DNC: KNN-200 depth normals vs Omnidata at 10°; raw depth until step 7k, filtered after | `dnc.py` (offline mask) + `ags-depth` (0.2 × EdgeAwareLogL1) | dnc-filter, depth-schedule |
| C2 ANR: rendered-depth normals vs prior; edge mask before 15k, 0.1 rad mask after; Gaussian-normal L1; λ 0.1 after 7k | renderer `surface_normal` + `ags-normal` | normal-anr, surface-normals |
| base: scale loss, Splatfacto density, TSDF init, Gaussian normals | copied from DN-Splatter | copied probes |
| C4 adaptive TSDF + IsoOctree meshing | not implemented | — |

## Prior-filter validation
- **Synthetic probe:**
  - A consistent wall is kept 100 %.
  - Corrupted-depth and wrong-prior blocks are rejected 100 %.
  - Open3D normals match an independent PCA (|cos| ≥ 0.9999).
  - The probe fails if the filter is removed or the normals are wrong.
- **vr_room** (40 frames): 15 % of pixels are rejected.
  - By depth structure: 63 % rejected at depth discontinuities, 13 % on smooth depth.
  - By distance: 10 % rejected under 1 m, rising to 30 % beyond 4 m.
- **Other rooms**, rejected fraction:

  | coffee_room | honka | kokko | sauna | computer |
  |---|---|---|---|---|
  | 20 % | 15 % | 16 % | 14 % | 16 % |

## Tests
- `ss-inspect` is clean, 5 pytest tests pass, and ruff is clean.
- `ss-verify` reports `declared_checks_passed` (11 probes).
- **Smoke run** on vr_room, 2000 steps on a compressed schedule:

  | method | within PSNR | cross PSNR |
  |---|---|---|
  | `ags-mesh` | 23.24 | 21.58 |
  | photometric-only control (from the DN-Splatter smoke run) | 23.29 | 21.41 |

## Code vs paper (code used)
- EdgeAwareLogL1 instead of L1.
- τ_N = 0.1 rad instead of 10°.
- The DNC angle is taken between encoded normals, with a JPEG mask round trip (the MuSHRoom dataparser path).
- ANR surface normals use detached depth, so that term passes no gradient. Setting `surface_grad: true` gives the paper's version.
- No TV loss, and the normal loss starts after 7k.

## Fidelity gaps
1. **No meshing or mesh metrics.** These need the Faro ground-truth meshes. The 2DGS variant is not ported.
2. **6 rooms required.** The paper reports only the 6-room average (within 24.83 and cross 22.24, against Splatfacto's 24.47 and 21.66), so all 6 rooms are needed for a comparison.
3. **Regenerated inputs.** The TSDF init and Omnidata normals are regenerated, not the authors' files. 35–45 % of init points are zero-distance duplicates, handled as in the DN-Splatter port.
4. **Runtime differences.** gsplat 1.5.3 instead of 1.0; LPIPS-VGG instead of AlexNet; views sampled with replacement.
5. **Empty normal mask.** An empty normal-mask selection gives 0, where the official code gives NaN.
6. **ScanNet++ infeasible.** It needs a licence and is mesh-only.

## Review-1 resolution
**Review result:** 1 major issue, fixed. The port matches the official `ags-mesh` code and the README command line.

**Control (the major issue).** The paper's Splatfacto row is vanilla `ns-train splatfacto`, inherited digit-for-digit from the DN-Splatter paper's runs. The benchmark therefore uses `dn-splatter-splatfacto` on all 6 rooms as the control:
- isotropic kNN init;
- random quaternions;
- the num_downscales=2 resolution schedule.

`ags-mesh-rgb` is kept only as a labelled ablation with the priors turned off.

**Normal-pass background.** This is now (1, −1, −1) in the OpenCV frame, the DN-Splatter review fix. The copied `gaussian-normals` probe was updated to the official formula.

**ANR has no effect on training.** The official code detaches depth when computing the ANR surface normals, so that term carries no gradient. A null C2 effect on NVS is therefore expected; it is not a port bug.

**Within-split sizes.** COLMAP dropped 3 test frames in coffee_room and 1 in honka, so their within splits have 35 and 32 frames. The official split does the same.

**Smoke run.** The 2k compressed smoke run is a plumbing check only and says nothing about final quality.
