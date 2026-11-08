# HGS: Hard Gaussian Splatting (plan #103, arXiv 2412.04826): reproduction report

**Outcome: matched.** On both benchmarked scenes our PSNR is within ±0.5 dB of the paper's per-scene value.

- **Method:** `hard-gs`, a native SplatStudio plugin in `plugin/ss_hard_gs`.
- **Official code:** none (github.com/GhiXu/HGS returns 404). The Scaffold-GS backbone is ported from the official
  Scaffold-GS code (@59c833b).
- **Benchmark:** experiment `runs/hard-gs-mipnerf360-pilr`. Settings: 30k steps, `--recipe 3dgs`, mipnerf360-pilr
  (3DGS-repo PIL resize -r4 / -r2, which is the paper's protocol, supp. §B).
- **Paper source:** per-scene numbers from supp. table `tab/mipnerf360.tex`.

## Results (test split, llffhold8)
| scene | method | PSNR | SSIM | LPIPS | anchors | train min |
|---|---|---|---|---|---|---|
| garden | **hard-gs (ours)** | **28.11** | 0.883 | 0.099 | 2.63M | 88 |
| garden | paper HGS | 28.33 | 0.884 | 0.086 | – | – |
| garden | scaffold-gs-hgsbase (ours, control) | 27.71 | 0.868 | 0.129 | 0.69M | 34 |
| garden | paper Scaffold-GS | 27.83 | 0.869 | 0.111 | – | – |
| garden | gsplat baseline (ours) / paper 3DGS | 27.35 / 27.74 | | | | |
| counter | **hard-gs (ours)** | **30.21** | 0.929 | 0.203 | 1.12M | 55 |
| counter | paper HGS | 30.26 | 0.928 | 0.149 | – | – |
| counter | scaffold-gs-hgsbase (ours, control) | 29.62 | 0.920 | 0.237 | 0.29M | 28 |
| counter | paper Scaffold-GS | 29.66 | 0.918 | 0.177 | – | – |
| counter | gsplat baseline (ours) / paper 3DGS | 29.03 / 29.11 | | | | |

How the results compare with the paper:
- **PSNR vs paper:** garden −0.22 dB, counter −0.05 dB.
- **HGS gain over the same Scaffold backbone:** +0.40 / +0.59 dB, against the paper's +0.50 / +0.60.
- **Anchor count:** about 3.8× Scaffold, consistent with the paper's storage ratio (HGS vs Scaffold-GS on T&T,
  0.73 vs 0.18 GB).
- **Training time:** about 2–2.6× Scaffold (the paper reports 3.2× on T&T).
- **SSIM:** matches the paper.
- **LPIPS:** our values are higher than the paper's, most clearly on counter. This holds for the Scaffold control
  too (0.237 vs 0.177), so it comes from the shared LPIPS implementation/protocol, not from HGS. Our relative LPIPS
  gain (−0.030 garden, −0.034 counter) is in line with the paper's (−0.025 / −0.028).

## Implementation
- **Backbone:** Scaffold-GS.
  - Each anchor has a 32-D feature, 10 offsets and a 6-D scaling.
  - MLPs take [feat, view dir] and decode the neural Gaussians.
  - Anchors are filtered by the fork's `visible_filter` before decoding.
  - Growing uses 3 voxel levels; pruning uses accumulated opacity.
  - Loss: 0.8 L1 + 0.2 D-SSIM + 0.01·mean(prod scale).
  - The engineering pattern comes from the matched HAC plugin.
- **PGHGS (Eq. 5–6):** a running top-k (k=3) of per-view NDC-scaled gradient norms for each neural Gaussian. It is
  hard if the k-th largest is ≥ λτ_i (λ=1).
- **REHGS (Eq. 7–8):**
  - |S_i| is the number of pixels whose maximum-weight Gaussian is i. It comes from a CUDA kernel reused from the
    Mini-Splatting plugin, which re-traverses gsplat's tile lists.
  - A Gaussian is a detection if |S_i| > 2e-4·W·H and the 3DGS SSIM map at its projected centre is < 0.7.
  - It is hard after ≥2 detections in an interval.
  - The SSIM check needs the ground-truth image, so the `hgs-scaling-reg` loss passes it to the density step.
- **Checks:** 8/8 mechanism probes pass in ss-verify, 9 unit tests pass, and review-1 approved the plugin with
  6 minor issues.

## Deviations and fidelity gaps
1. **The way hard Gaussians enter growing is guessed (no official code).** At Scaffold level i (τ_i = τ·2^i), a
   neural Gaussian is a candidate if any of these holds:
   - OG: `avg ≥ τ_i` and denom > 40;
   - PGHGS: `g'_3 ≥ λτ_i`;
   - REHGS: hard.

   Scaffold's random keep mask and voxel de-duplication then apply unchanged. Within this guess:
   - **Success mask:** hard candidates skip Scaffold's 40-view mask (review m1).
   - **Thresholds:** PGHGS uses the level-scaled threshold, while Eq. 6 literally says λτ_grad (review m3).
   - **Window:** the hard statistics cover one 100-step interval and are reset every 100 steps (review m2).
   - **≥2-views rule:** we need ≥2 hard *detections*; the text could also mean "hard once and visible in ≥2 views"
     (review m4).

   The matched gains and the 3.8× growth ratio suggest this guess is close, but it is not verified.
2. **Effi-HGS (§5.3) is not implemented.** The top-Q cap is a separate variant and not the reproduced row.
3. **Rasterizer:** gsplat (classic, eps2d 0.3) instead of the Scaffold diff-gaussian-rasterization fork.
4. **Training-loop differences:**
   - white background instead of Scaffold's black;
   - views sampled at random with replacement instead of a shuffled stack;
   - the runtime SSIM loss uses a valid convolution;
   - the runtime steps Adam before growing, while Scaffold grows before `optimizer.step()`.
5. **Weight transport:** the MLP weights reach ss-eval inside the per-anchor field `hgs_blob` (with a checksum),
   which needs ≥887 anchors.
6. **Voxel size per dataset:** the method default is the mip360 value 0.001. `hard-gs-tandt` (0.01) and `hard-gs-db`
   (0.005) exist but were not benchmarked.
7. **Fixed schedule length:** `total_steps` is fixed at 30000, so the LR schedules ignore `--steps` (review m5).
8. **Baseline comparability:**
   - The paper's 3DGS and Scaffold-GS rows were retrained by the authors on an RTX 3090.
   - Our gsplat baseline is 0.39 dB (garden) and 0.08 dB (counter) below the paper's 3DGS row.
   - Our Scaffold control is 0.12 dB and 0.04 dB below the paper's Scaffold-GS row.
9. **Coverage:** only 2 of 9 Mip-NeRF 360 scenes were run, and T&T and DB were not. This is the batch rule, so no
   dataset average is claimed.

## Smoke and weak runs (kept)
`evidence/smoke-1`: the first version accumulated the hard statistics over steps 501–1600 and jumped 5× in anchors at
the first growth step. I fixed this with the per-interval reset (`evidence/smoke-2`). See `evidence/test.json`.
