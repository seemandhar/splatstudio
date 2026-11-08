# Compact Neural Appearance Models for Efficient Gaussian Splatting (plan #124): MATCHED

arXiv 2609.05255 (Hahlbohm et al.).

- **Code:**
  - Official code: nerficg-project/efficient-gaussian-appearance (Apache-2.0).
  - PPISP: vendored from nv-tlabs/ppisp @a247368 (Apache-2.0) and JIT-built.
  - Faster-GS rasterizer, FusedAdam and Morton ordering: imported from reproductions/faster-gs.
- **Base pipeline:** the paper's per-scene Neural and SH rows both come from Faster-GS + MCMC (Beta-Splatting caps) + PPISP with 5k controller distillation.
  - This is not the Faster-GS main configuration we matched, which used 3DGS ADC and no PPISP.
  - We reuse the faster-gs kernels and port MCMC (from the official kernels) and PPISP.
  - Both appearance models ship, so we measure the paper's paired gain (Neural − SH on the same base).
- **Methods:** `neural-appearance-<scene>` and `neural-appearance-sh-<scene>`; the scene suffix selects the MCMC cap (garden 5.0M, room 1.5M).
- **Protocol:**
  - 30k training steps + 5k PPISP distillation with a frozen scene (`--steps 35000`).
  - M360 at images_4 / images_2 (1297×840, 1557×1038), black background, llffhold8.
  - Plain `ss-eval`: the MLP and PPISP are restored from the `na_blob` field.

## Result (PSNR, test split, 35k steps, single seed)
| Scene | Method | Ours | Paper (5-run mean) | Δ | Gaussians | Train time |
|---|---|---|---|---|---|---|
| garden | Neural | **28.05** | 28.06 | −0.01 | 4.87M | 87 min |
| garden | SH twin | 27.94 | 27.96 | −0.02 | | |
| room | Neural | **33.05** | 33.23 | −0.18 | 1.25M | 35 min |
| room | SH twin | 32.81 | 32.98 | −0.17 | | |

| Scene | Neural − SH, ours | Neural − SH, paper |
|---|---|---|
| garden | +0.10 | +0.10 |
| room | +0.25 | +0.25 |

- **Primary criterion:** Neural within ±0.5 dB per scene. Both scenes pass, so the outcome is **matched**.
- **Secondary criterion:** our gain over the same-pipeline SH twin equals the paper's gain exactly on both scenes.
- **Baseline used:** the paper has no 3DGS row, so the SH twin is the baseline. baseline3dgs-mipnerf360-30000 serves as a reference only.
- **Runs:** `runs/neural-appearance-garden` and `runs/neural-appearance-room`, plus the SH twin runs.

## Checks
- ss-inspect reports no issues; 7 unit tests pass; ruff is clean.
- ss-verify reports declared_checks_passed for all 6 probes:
  - decoder vs explicit formula and 3DGS eval_sh;
  - render vs a float64 dense compositor (2.4e-7);
  - direction gradient;
  - MCMC relocation vs the exact binomial sum;
  - schedules;
  - CUDA PPISP vs torch reference (6.9e-7);
  - blob round trip.
- **Distillation phase:** checked only by a shortened 300+100-step run (`evidence/distill-check.log`): the scene stays frozen, the controller is trained, and the reload matches the live render (1e-5).
- **Smoke test** (garden, 2k steps): Neural 22.95 / SH 22.75 / gsplat 22.99.
- **Review:** review-1 approved, with 0 critical, 0 major and 6 minor issues.

## Deviations / fidelity gaps
1. **PPISP wrapper reconstructed (review m1):**
   - The NeRFICG `PPISPWrapper` (frame mapping, num_cameras, activation ratio) is reconstructed from the nerficg `Optim/ppisp.py` that the faster-gs sources contain. It is not part of this paper's repository.
   - We assume one PPISP camera, which holds for M360, T&T and DB.
   - Training views are mapped to PPISP frames by matching the viewmat key (unique for every garden and room train view; no test view collides).
   - Every evaluation view is a novel view and uses the distilled controller (`frame_idx=-1`).
2. **fp32 MLP:** the MLP runs in fp32 PyTorch over all Gaussians (the paper's unfused reference path) instead of the fused fp16 tcnn kernel with loss scale 128. The paper reports equal quality for both paths.
3. **Single seed:** we ran one seed; the paper averages 5 runs (std 0.02–0.06 dB). The room Δ of −0.18 is within about 3σ of run noise plus the protocol gaps below.
4. **RNG:** tcnn's Xavier-uniform init and the Philox MCMC noise are matched in distribution only, not bit for bit.
5. **Runtime differences:**
   - The runtime SSIM (valid convolution) replaces fused-ssim in the loss.
   - Views are sampled with replacement.
   - Images come from the runtime's images_4/images_2 loader, not the NeRFICG scale factor.
6. **Minor implementation notes (review m2–m6):**
   - m2: one GPU→CPU viewmat sync per render.
   - m3: the module-global bank `ACTIVE` takes precedence over `na_blob` in-process; ss-eval runs in a fresh process, so this is safe.
   - m4: no probe covers the distillation phase; it is covered only by the evidence log.
   - m5: learning rates are zeroed after T while the runtime keeps rewriting the means lr; this is harmless because no gradient flows.
   - m6: bench needs one launch per scene.
7. **Unused memory:** `shN` is allocated but unused for Neural (memory only).
