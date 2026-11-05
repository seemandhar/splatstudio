# DashGaussian (plan #31): reproduction report

**Outcome: matched.** Over 7 scenes the mean gap to the paper is −0.35 dB, with 5 of 7 within ±0.5 dB, after one fix iteration.
- Drjohnson is the clear miss (−0.76 dB).
- Garden is marginal (−0.53 dB).
- A matched PSNR is not proof of fidelity; every known deviation is listed below.

| | |
|---|---|
| Paper | *DashGaussian: Optimizing 3D Gaussian Splatting in 200 Seconds* (Chen et al., CVPR 2025, arXiv 2503.18402) |
| Official code | https://github.com/YouyuChen0207/DashGaussian (CC BY-NC-SA 4.0), current version including the 2025-08-16 fix |
| Variant reproduced | DashGaussian on the **3DGS backbone (Adam)**: the "3DGS +Ours" rows of Table 2 and supp. Tables 7–9 |
| Not reproduced | The headline "200 s" result, which is DashGaussian on Taming-3DGS (Sparse Adam + fast backward) |
| Plugin | `plugin/` (package `ss_dashgaussian`, method `dashgaussian`); no custom CUDA |

## Mechanism → implementation
| contribution | component | notes |
|---|---|---|
| C1 frequency-guided resolution schedule (Sec. 3.2, supp. Eq. 6–7) | `dash-density` builds the schedule from the training images' FFT; `dash-renderer` renders at floor(H/r)×floor(W/r) with K scaled by (w/W, h/H) | 32 levels, a=4, r_m ≤ 8, inverse-square interpolation, floored; re-evaluated at every densification step; full resolution from step 27000 |
| GT downsampling (supp. 7.1) | loss `photometric` resamples the target to the render size (antialiased bilinear) | the runtime always passes the full-resolution target |
| C2 resolution-guided primitive target (Eq. 4) and top-k densification (supp. 7.4) | `dash-density` | prune first (opacity < 0.005); clone/split only the top-k qualifying Gaussians; no size pruning; densify until 27000; opacity reset every 3000 while densifying |
| C3 momentum budget (Eq. 5) | `dash-density` | γ 0.98, step cap 1e6; initial budget P_fin = 6·P_init (code; the paper says 5·P_init) |
| C4 position-LR hold (supp. 7.2) | `dash-density.update_lr` | constant until the first level with r < 2, then log-linear decay to 1 % at 30000 |

**Evaluation is always full resolution.**
- The renderer lowers resolution only when gradients are on and a linked density has an active schedule.
- `measure` and `ss-eval` run under `no_grad`, and `ss-eval` builds fresh components.

**Runtime workarounds** (the runtime hooks that would remove them are listed in `plan.md`):
- **Training images:** the density reloads the dataset from `--data` and verifies every training camera against the runtime's cameras.
- **Step:** the renderer is not given the step, so it follows the schedule through a process-local link to the density. `before_backward` checks the rendered size against the schedule every step.

## Iterations
| iter | change | why |
|---|---|---|
| 0 | port as above; training D-SSIM = runtime valid-convolution SSIM | — |
| 1a | **Garden and bicycle on PIL-protocol data** (`~/datasets/mipnerf360/pilr/<scene>`: images PIL-resized from full resolution, as the 3DGS loader does; compared against a gsplat baseline on the same data) | our images_4 gsplat baseline is ~0.9 dB below the paper's 3DGS on garden (26.89 vs 27.80), so the gap is in the data, not the method |
| 1b | **Same-padded D-SSIM in the training loss** (`ssim_padding: "same"`: 11×11 Gaussian window, σ 1.5, zero padding 5, as 3DGS `loss_utils.ssim` / `fused_ssim`). Evaluation SSIM is unchanged (runtime valid SSIM) | valid convolution drops a 5-px border; at r=6–7 renders are about 120–260 px wide, so 10–14 % of pixels lost SSIM supervision during the long low-resolution phase |

Nothing else changed; no hyper-parameters were tuned on test images. Train was not re-run in iteration 1: the truck control moved only −0.06 dB with the SSIM fix.

## Results (30k steps, `--recipe 3dgs`, llffhold8 test split)
Ours is iteration 1 except train (iteration 0). Paper values are from supp. Tables 7–9, "3DGS +Ours".

| scene | PSNR ours / paper | Δ vs paper | SSIM ours / paper | LPIPS ours / paper | Gaussians ours / paper | time ours (s) | speed-up vs our gsplat |
|---|---|---:|---|---|---|---:|---:|
| garden (PIL) | 27.40 / 27.93 | −0.53 | 0.862 / 0.873 | 0.144 / 0.110 | 3.34M / 3.42M | 4431 | 1.40× |
| bicycle (PIL) | 25.33 / 25.70 | −0.37 | 0.769 / 0.785 | 0.248 / 0.202 | 5.59M / 5.38M | 2354 | n/a (PIL baseline incomplete; iter 0: 1.37×) |
| room | 31.11 / 31.58 | −0.47 | 0.907 / 0.923 | 0.313 / 0.203 | 1.43M / 1.29M | 1647 | 4.90× |
| truck | 25.43 / 25.66 | −0.23 | 0.879 / 0.882 | 0.193 / 0.156 | 1.87M / 1.85M | 669 | 5.34× |
| train (iter 0) | 21.70 / 22.01 | −0.31 | 0.809 / 0.806 | 0.258 / 0.223 | 1.11M / 1.01M | 1615 | 3.17× |
| drjohnson | 27.99 / 28.75 | **−0.76** | 0.890 / 0.895 | 0.348 / 0.262 | 2.52M / 2.47M | 1926 | 2.30× |
| playroom | 30.06 / 29.84 | +0.22 | 0.909 / 0.906 | 0.320 / 0.249 | 1.86M / 1.89M | 2777 | 2.36× |
| **mean** | | **−0.35** | | | | | |

**How the results compare with the paper:**
- **Gaussian counts** match the paper within about 10 % on every scene.
- **LPIPS** is uniformly higher than the paper's. This is also true of our gsplat baseline, so it comes from the runtime (white background, LPIPS from the `lpips` package), not from DashGaussian.
- **Speed-up:** the paper reports 1.43–2.26× for its 3DGS → +Ours. All timings were taken on shared, contended RTX A6000 GPUs and are only indicative; per-scene ratios vary from 0.94× to 5.3× across runs. Absolute times cannot be compared with the paper (gsplat rasterizer vs `3dgs-accel`).

**Δ vs our gsplat baseline, compared with the paper's Δ (+Ours − 3DGS):**

| scene | Δ ours | Δ paper |
|---|---:|---:|
| garden (PIL) | +0.14 | +0.13 |
| bicycle (PIL) | n/a | +0.09 |
| room | −0.41 | −0.14 |
| truck | +0.22 | +0.21 |
| train | +0.18 | +0.23 |
| drjohnson | −0.77 | −0.34 |
| playroom | +0.26 | −0.06 |

**Iteration 0, kept for the record** (PSNR, Δ vs paper):

| scene | PSNR | Δ vs paper |
|---|---:|---:|
| garden (images_4) | 26.97 | −0.96 |
| bicycle (images_4) | 24.90 | −0.80 |
| room | 30.85 | −0.73 |
| truck | 25.49 | −0.17 |
| train | 21.70 | −0.31 |
| drjohnson | 27.96 | −0.79 |
| playroom | 30.18 | +0.34 |

## Drjohnson analysis (the clear miss)
**The schedule ran as designed.**
- Training starts at r=7 (r_m 7.82) and moves through r=6/5/4/3/2 at steps 2085/4850/8070/11631/15946.
- It reaches r=1 only at step **21104**, the latest of all the scenes (room reaches it at 18717).
- The position LR is held until 20882, so only about 9k steps run at full resolution. Those steps have a decaying LR and opacity resets at 21k and 24k.

**Growth is not the problem.**
- Until r=1, growth is limited by the Eq. 4 target (+0.2–0.5 % per 100 steps).
- After r=1, the gradient threshold limits it (+1–3 % per 100 steps); it is still growing when densification stops at 27000.
- The momentum budget never binds: P_fin is 2.91M against a final count of 2.52M.
- The final count matches the paper (2.52M vs 2.47M).

**The model underfits.**
- Iteration-0 final training loss was 0.026 vs the baseline's 0.018, the largest gap of any scene.
- The SSIM fix did not change it: iteration 1 scores 27.99 vs 27.96 in iteration 0.

**This is in the paper too.** The paper's drjohnson is also its worst DashGaussian scene (−0.34 dB vs its 3DGS); our deficit is about twice as large.
- Our gsplat baseline grows to 3.61M Gaussians (the paper's 3DGS has 3.31M), so the baseline we compare against is stronger.
- No bug or other deviation from the official code was found: schedule, budget, LR endpoints, densification window and counts all check out.

## Probes and tests
- `ss-verify` reports `declared_checks_passed` with 10 probes (`evidence/verify.json`, `evidence/verify-i1.json`):
  - **resolution schedule:** white-noise analytic r_m=2 and r_i=(E_i/E)^−½; grid-search reference; supp. Eq. 7 begin steps; floored interpolation; monotone to r=1.
  - **render scale:** matches gsplat with hand-scaled K; evaluation renders at full resolution.
  - **target downsampling:** matches a PIL bilinear-reduce reference.
  - **primitive schedule:** Eq. 4 / Eq. 5, 0.2 cap, P_fin^(0) = 6·P_init.
  - **top-k densification:** prune first, exact top-k, no size pruning, zero Adam moments.
  - **position-LR schedule.**
  - **renderer gradients** with the schedule active.
  - **end-to-end runtime cycle** on demo data (r 4→1, evaluation at full resolution).
  - plus the built-in state-transition and photometric probes.
- 11 CPU unit tests pass, including: same-padded SSIM against an independent 3DGS-style reference, which fails for `"valid"`; interpolate-size parity; and field-of-view preservation. ruff is clean.
- Smoke test (garden, 2000 steps): finite. A compressed full-cycle schedule ran via the Python API (`evidence/test.json`).

## Fidelity gaps
1. **Rasterizer:** gsplat instead of 3DGS `3dgs-accel`, so absolute time is not comparable.
2. **Runtime-wide differences:** white background (paper uses black), view sampling with replacement, evaluation SSIM with valid convolution and LPIPS-VGG from the `lpips` package. These are shared with the baseline.
3. **Training D-SSIM:** iteration 1 uses same padding (matching the paper); the baseline's loss still uses valid convolution.
4. **Step order:** the optimizer step happens before densification; the official code densifies first.
5. **Baseline:** gsplat DefaultStrategy, not the paper's 3DGS retrain.
6. **Data:** PIL-protocol data was used only for garden and bicycle; the other scenes use the standard native conversions.
7. **Training images:** the runtime does not expose them, so the density reloads the dataset (verified against the runtime's cameras). Schedule setup takes 1–5 min and sits outside the training timer, as the official timer also excludes it.
8. **Step not passed to the renderer:** it follows the schedule through a process-local link.
9. **Paper vs current code:** the paper's numbers predate the official code's 2025-08 fix. The README's post-fix Mip-NeRF 360 average is 27.70, against 27.81 in the paper.

## Review items
- **Drjohnson:** −0.76 dB, about 0.4 dB worse than the paper's own drop. Open question: does the paper's (pre-fix) code reach r=1 earlier on Deep Blending? Re-run with the post-fix official code if a CUDA 3DGS build is available.
- **Timing:** speed-ups need re-measuring on an idle GPU; the iteration-0 garden (0.94×) and drjohnson (1.06×) runs were contention artefacts.
- **Bicycle:** the gsplat baseline on PIL data is incomplete, so bicycle has no Δ or speed-up yet.
- **Ablations:** `resolution_mode: const` and `densify_mode: free` exist as config switches but have not been run; they need a separate method entry.

## Reproduce
```bash
cd reproductions/dashgaussian/plugin && ../../../tools/ssenv pip install -e .
tools/ssenv ss-inspect dashgaussian && tools/ssenv ss-verify dashgaussian --device cuda
tools/ssenv python -m pytest -q reproductions/dashgaussian/plugin/tests
python3 tools/bench.py launch --exp dashgaussian-i1-mipnerf360 --methods dashgaussian --dataset mipnerf360 --scenes room --steps 30000 --recipe 3dgs
python3 tools/bench.py launch --exp dashgaussian-i1-tandt --methods dashgaussian --dataset tandt --scenes truck train --steps 30000 --recipe 3dgs
python3 tools/bench.py launch --exp dashgaussian-i1-db --methods dashgaussian --dataset db --scenes drjohnson playroom --steps 30000 --recipe 3dgs
# garden/bicycle: PIL-protocol data in ~/datasets/mipnerf360/pilr/<scene> (runs/dashgaussian-i1-pilr2)
python3 tools/bench.py summarize --exp dashgaussian-i1-db --paper reproductions/dashgaussian/paper_results.json --baseline gsplat
```
