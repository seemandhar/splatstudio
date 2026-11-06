# Group Training (plan #94) — build report
- **Paper:** *Faster and Better 3D Splatting via Group Training*, ICCV 2025, arXiv 2412.07608.
- **Official code:** Chengbo-Wang/3DGS-with-Group-Training, read and not run. It has no license file.
- **Status:** built, benchmark pending.

## Host
The per-scene numbers (supp. tab:supp_3dgs_MipNerf360, row "w/ OPS 0~30K") come from the vanilla-3DGS host with the released default GroupingParams. The paper's baseline is its retrained 3DGS*.

## Contributions → code (plugin/ss_group_training/density.py)
- **C1, subset rendering and updating:**
  - The held-out 40% is removed from params, optimizers and density state with gsplat ops.remove (official prune_points) and stored detached.
  - Only the sampled group is rendered, stepped and densified.
  - On merge, the held-out rows come back with zero Adam moments (official), and all density accumulators are zeroed (official). Nothing needs to be scattered back.
  - Probe gt-subset-state shows that held-out rows stay bit-identical.
- **C2, opacity-prioritised sampling:** multinomial(σ(o)/Σσ(o), 0.6N, no replacement) (Eq. 11).
- **C3, schedule:**
  - Regroup at 1000, 1500, …, 29000.
  - Merge without resampling at 14500 and 29000.
  - Higher-order SH is frozen at the merges, as in the official code.
  - Densification is otherwise the baseline's, applied to the subset.
- **Custom CUDA:** none.

## Verification
- ss-inspect is clean, pytest passes (5), ruff is clean, and ss-verify passes 7 probes.
- Sampling inclusion rates are 0.989 / 0.211 vs a numpy reference of 0.988 / 0.212 (random would be 0.6).

## Smoke test (garden, 2k steps, run concurrently on the same GPU)
| run | PSNR | SSIM | LPIPS | Gaussians | train s | peak GB |
|---|---|---|---|---|---|---|
| group-training-smoke2k | 22.49 | 0.598 | 0.446 | 838,548 | 158.5 | 1.13 |
| gsplat | 23.00 | 0.643 | 0.418 | 841,292 | 167.0 | 1.37 |

The −0.51 dB is expected at 2k: 40% of the Gaussians sit out from step 1000 to 2000.

## Benchmark
| scene | paper OPS 0~30K | paper 3DGS* | paper Δ | our baseline |
|---|---|---|---|---|
| garden | 27.418 | 27.397 | +0.02 | 26.893 |
| counter | 28.973 | 28.980 | −0.01 | 28.846 |

- **Paper efficiency:** training time ×0.65 / ×0.82 and models about 22% smaller.
- **What to compare:** PSNR gain vs the paper's gain, and our training-time and Gaussian-count ratios to gsplat.

## Fidelity gaps
1. **Runtime host differences shared with the baseline:**
   - white background;
   - no screen-size prune;
   - optimizer step before density.
2. **Wall time:** measured on shared A6000s, so it is noisy; compare ratios.
3. **Variants:** only OPS 0~30K is a method. Random sampling and 0~15K are config-only.
4. **3DGS version:** the exact 3DGS commit behind the official numbers is unknown.

## Review-1 resolution (orchestrator)
Approved, with 0 critical, 0 major and 5 minor issues. Confirmed against the official code:
- regroup and merge state handling;
- opacity sampling;
- the schedule and the SH freeze windows (14500, and 29000–30000);
- subset-only densification;
- resets touch only the subset;
- paper_results.json for all 9 scenes;
- nothing in the port erases the speed-up.
- **Official behaviours reproduced but previously undocumented:**
  - Cached Gaussians escape the opacity resets, so OPS favours them at the next regroup.
  - Refines at multiples of 500 run on one step of statistics, because the merge zeroed them.
- **Host version:** the official code runs on graphdeco 3DGS from before tmp_radii was added (before late 2024), consistent with the port.
- **Minor, not fixed:**
  - The final merge assumes total_steps = 30000.
  - The probe does not cover the reset/cache interaction.
  - The 2k smoke gap is not informative.
- **Benchmark:** group-training-mipnerf360 (group-training and gsplat, garden and counter, 30k). The gsplat run is a same-load timing reference.

## Benchmark result (group-training-mipnerf360, 30k) — PARTIAL (absolute); gains match
| scene | group-training | paper OPS 0~30K | Δ abs | gsplat same exp | shared baseline | paper 3DGS* | gain vs shared baseline ours / paper |
|---|---|---|---|---|---|---|---|
| garden | 26.78 / 0.860 / 0.132 | 27.418 | **−0.64** | 26.96 | 26.89 | 27.397 | −0.11 / +0.02 |
| counter | 28.92 / 0.908 / 0.261 | 28.973 | −0.06 | 28.95 | 28.85 | 28.980 | +0.07 / −0.01 |

| scene | training s, GT / gsplat (ratio) | paper ratio | Gaussians, GT / gsplat (ratio) | paper ratio |
|---|---|---|---|---|
| garden | 4180 / 2753 (1.52×) | 0.65× | 4.24M / 4.76M (0.89) | 0.78 |
| counter | 1287 / 1598 (0.81×) | 0.82× | 1.01M / 1.21M (0.84) | 0.79 |

- **Quality:** the gain is ≈0 on both scenes, as in the paper.
  - Garden's absolute miss comes from our gsplat garden baseline sitting 0.5 dB below the paper's 3DGS* (26.89 vs 27.40). Group Training itself is within 0.11 dB of that baseline.
  - Under the absolute ±0.5 dB rule, garden fails, so the verdict is partial.
- **Speed:** counter reproduces the paper's ×0.82 time ratio. Garden ran 1.5× slower than its gsplat pair. Wall time on shared GPUs is unreliable (the two runs were on different GPUs and loads), so the speed claim is inconclusive.
- **Model size:** about 11–16% fewer Gaussians, vs the paper's ~22%.
