# DN-Splatter (plan #63): build report

**Status:** the plugin is built and verified. The 30k benchmark is pending (`bench_request.json`).

- **Paper:** arXiv 2403.17822 (WACV 2025).
- **Official code:** maturk/dn-splatter @ 97588b4, built on Nerfstudio 1.1.3 and gsplat 1.0.0.
- **Plugin:** `plugin/` (`ss_dnsplatter`), with three methods. There is no custom CUDA.
  - `dn-splatter`: the method.
  - `dn-splatter-splatfacto`: the paper's baseline control, i.e. vanilla `ns-train splatfacto` (eval/batch_run.py:186-207, `load_pcd_normals` False). It uses the same 1M sensor-depth points, isotropic log(mean 3-NN) scales with random quaternions (splatfacto.py:214-220), Splatfacto density control with the photometric loss, and the 1/4 → 1/2 → full resolution schedule.
  - `dn-splatter-rgb`: a priors-off ablation. It keeps DN-Splatter's oriented flat init and full-resolution training and drops the depth/normal/scale losses. It is **not** the paper baseline.
- **Data:** MuSHRoom iPhone `vr_room` (zenodo 10230733), converted to `~/datasets/mushroom/native/vr_room` by `tools/prepare_mushroom.py`. It uses the official splits, the official TSDF 1M-point init with Open3D normals, and Omnidata v2 normals.

| contribution | component | probe |
|---|---|---|
| C1: edge-aware log-L1 depth | dn-depth (effective weight 1.2, from the code) | depth-loss |
| C2: shortest-axis Gaussian normals | dn-render normal pass | gaussian-normals (CUDA) |
| C3: Omnidata normal L1 + TV | dn-normal (weights 1/1, from the code) | normal-loss |
| C4: min-scale loss | dn-scale | scale-loss |
| C5: sensor-depth init, oriented flat Gaussians | prep tool + `density.initialize` | sidecar-init |
| base: Splatfacto density control | splatfacto-density | splatfacto-density |
| control: Splatfacto isotropic init | splatfacto-density `sidecar_init: isotropic` | isotropic-init |
| control: Splatfacto resolution schedule | dn-render `num_downscales 2` + plugin photometric loss | resolution-schedule |

## Where code and paper differ (code used)
- **Loss weights:** 1.2 / 1 / 1 in the code, 0.2 / 0.1 / 0.1 in the paper.
- **Normal flipping:** normals are flipped toward the camera every step.
- **Normal-pass background:** the official white background lives in the OpenGL camera frame. In our OpenCV frame that is (1, −1, −1), fixed after review 1. Only pixels with partial alpha change (2–10 % of training pixels at 2k steps).
- **Densification:** Splatfacto's own, with the means learning rate not scaled.

## Tests
- `ss-inspect`: no issues for all three methods. pytest: 6 passed. ruff: clean. `ss-verify --device cuda`: `declared_checks_passed` for `dn-splatter`, `dn-splatter-splatfacto` and `dn-splatter-rgb` (`evidence/verify*.json`).
- **Smoke run** (2k steps on `vr_room`, before review 1, against the `dn-splatter-rgb` ablation):
  - Scores: `dn-splatter` 22.64 on test (within) and 20.95 on validation (cross); the ablation 23.29 / 21.41.
  - Why it does not count: a 2k run says nothing about 30k. The runtime compresses the means-LR decay into the given step count, densification gets only 15 refine calls, and the paper's own ablation shows the normal-TV term costing RGB while depth recovers it.
  - No bug was found. Projecting the init points gives a median depth ratio of 0.997–1.011, the sidecar viewmats match all 376 frames, and rendered normals reach cos 0.96–0.98 against Omnidata.
  - Only the 30k comparison against `dn-splatter-splatfacto` counts. The paper's +0.2 dB / +0.009 SSIM is within run-to-run noise, so a tie is a valid outcome.
- **Normals check:** the raw Omnidata png agrees with OpenCV depth normals (cosine 0.71–0.87), so the official comparison is self-consistent.

## Fidelity gaps
1. **gsplat version:** 1.5.3 instead of 1.0.0. The 1.0 projection radius is recomputed for the screen-size thresholds.
2. **Init:**
   - The TSDF init is regenerated with a fixed seed.
   - 35 % of the init points are zero-distance duplicates, whose log scale would be −inf. These are set to −1e4 instead.
3. **LPIPS:** VGG here, AlexNet in the paper, so LPIPS is not comparable.
4. **Evaluation split:**
   - Headline: test (42 test.txt frames, within).
   - Table 4 does not state its split. Every single-number MuSHRoom NVS table in the paper uses the within value: the main_sfm_init 1M row (24.58 / .8558) equals Table 1 left, and the ab_2dgs_vs_dn_splatter_mono_sensor caption says "within the train sequence".
   - Validation (186 short_capture frames, cross) is reported as secondary.
   - No union split: it is not an official metric, because dn_pipeline.py averages within_ and with_ separately even with `eval_mode='all'`.
5. **View sampling:** with replacement.
6. **Geometry:** no meshing or mesh metrics, so the geometry claims are unverified.
7. **Scope:** only one per-scene target. The 6-scene Table 3 average is not attempted.
8. **Splatfacto control:**
   - The 1/4 → 1/2 → full resolution schedule runs inside the renderer, which counts training renders (equal to the runtime step). Intrinsics and image size are divided by d, and the plugin photometric loss downscales the target with Splatfacto's `resize_image`.
   - It shares the base SSIM (valid convolution, not torchmetrics) and with-replacement view sampling with the other methods.
   - The random quaternions come from the runtime-seeded torch RNG, so they are not the authors' draw.


## Benchmark results (MuSHRoom iPhone vr_room, 30k steps)
**Outcome: matched on absolute PSNR. The gain over Splatfacto is not reproduced.**

Runs are in `runs/dn-splatter-vr_room/`. "Within" is the 42 frames in `test.txt` (the headline split, as in the paper). "Cross" is the 186 frames of the second trajectory.

| method | within PSNR / SSIM / LPIPS | paper within PSNR / SSIM | cross PSNR / SSIM | #G |
|---|---|---|---|---|
| **dn-splatter** | **26.87** / 0.875 / 0.301 | 26.8 / 0.879 (**+0.07**) | 23.77 / 0.822 | 974k |
| dn-splatter-splatfacto (vanilla Splatfacto control) | 27.69 / 0.892 / 0.260 | 26.6 / 0.870 | 23.69 / 0.830 | 991k |

**Reading the result**
- **Absolute numbers:** DN-Splatter reproduces the paper's NVS numbers (+0.07 dB PSNR, −0.004 SSIM).
- **Control vs the paper's baseline:** our vanilla-Splatfacto control is 1.09 dB above the paper's Splatfacto row. It uses isotropic kNN init, random quaternions and the num_downscales=2 schedule, and runs on the same data and runtime.
- **Gain over the control:**
  - Ours is −0.82 dB on the within split, against the paper's +0.2. The paper itself notes that its +0.2 dB is within noise.
  - On the cross split the two are about even (+0.08 dB).
- **Likely causes of the gap:**
  - Our runtime and gsplat 1.5.3 train Splatfacto better than the paper's Nerfstudio 1.1.3 / gsplat 1.0 setup.
  - DN-Splatter's depth and normal terms trade some RGB fidelity for geometry. The paper's own ablation shows this for the TV term.
- **Not tested:** the paper's geometry claim, which needs mesh metrics against the Faro scans.
- **No iteration:** no grounded fix was found; the method's absolute numbers already match.
