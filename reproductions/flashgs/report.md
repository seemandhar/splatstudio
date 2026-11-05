# FlashGS (plan id 73) — build report (pre-benchmark)

**Feasibility**: the official release (InternLandMark/FlashGS@cdfc4e40, MIT) is forward-only
(no backward kernels; uint8 output). FlashGS cannot train; the paper itself renders models
trained by unmodified 3DGS. Reproduced as a **render-only** plugin:
- `flashgs` method: gsplat while gradients are enabled (training identical to baseline), vendored
  FlashGS kernels under `no_grad` (final validation, `ss-eval`).
- `plugin/apply_flashgs.py`: paper protocol on finished baseline runs — quality (gsplat / original
  3DGS / FlashGS, SplatStudio metrics + parity PSNR) and same-GPU speed at native/1920/3840 widths.
- CUDA vendored unmodified (FlashGS MIT; graphdeco diff-gaussian-rasterization@59f5f77e as the
  paper's reference, Inria licence), JIT via `tools/ssenv`.

**Verification**: ss-inspect clean; 9 unit tests; ss-verify 7/7 probes passed (parity 51 dB vs
3DGS on fixture, model conversion = upstream loadPly; 0 missing visible tile pairs, pairs 8342
vs 19064 3DGS-AABB; fixture 1080p 8.8× vs 3DGS).
Smoke garden 2k: flashgs 23.11 dB vs gsplat 23.02 dB.
Truck smoke, 3 test views, shared A6000 at ~100% util — **same-GPU ratios, not the paper's 3090**:
PSNR gsplat 26.03 / 3DGS 26.01 / FlashGS 26.11 (uint8 truncation alone: 26.07); speedup vs
3DGS 2.3× native, 3.8× 1080p (paper 3.76), 6.3× 4K (paper 7.01); vs gsplat 2.6/3.3/3.1×;
key/value pairs −59/−72/−80 %.

**Fidelity gaps**: busy shared A6000; test views rather than the paper's frame sets; 1080p/4K from
rescaled intrinsics (T&T 1920×1071); gsplat-trained white-bg models; FlashGS gets precomputed
cov3D (its API); memory claim not measured; MatrixCity/Rubble unavailable; upstream
empty-right-edge-tile write bug counted (`edge_wrap_tiles`), not patched.

Benchmark: `bench_request.json` (evaluation-only, truck/train/drjohnson/playroom).


## Review-1 resolution (orchestrator)
**Major issue, fixed in the comparison mapping (no rerun needed).** The paper's Deep Blending "1080p" column is actually the native resolution: the draft table at `output.tex:833` lists Playroom 1264×832 and DrJohnson 1332×876. Those two scenes are therefore compared against our `speed.native`, and T&T against `speed.1080p`.

**Match criteria, fixed in advance:**
- **Quality:** |ΔPSNR(FlashGS − 3DGS)| ≤ 0.05 dB, with parity ≥ 40 dB.
- **Speed:**
  - Compare only same-GPU `avg_speedup`, the mean of per-frame ratios, which is how the paper computes it.
  - It must be at least 0.8× the paper's value at the equivalent resolution.
  - 4K must beat 1080p, and DrJohnson must show the largest gain.
- **Reliability:** run on a GPU lightly used by others, and rerun once to record the spread.

**Minor issues, recorded:**
- There is no `3dgs_uint8_trunc` quality row. Truncating to uint8 alone shifts PSNR by +0.04 dB.
- Backends are timed in a fixed order.
- Only test views are timed; upstream times every camera.
- Correction to the wording "bit-identical training": it is the same code path. gsplat itself is not deterministic.


## Benchmark results (evaluation on finished 30k gsplat baselines)
**Outcome: partial.**

**Setup.**
- Speedups are against the original 3DGS rasterizer, on the same GPU (GPU 0, 0–5 % load from other users at the start).
- Each figure is `avg_speedup`, the mean of per-frame ratios over the test views, as in the paper.
- The paper's numbers are on an RTX 3090; ours are on an RTX A6000.
- Outputs are in `results/<scene>.json`.

| scene | PSNR 3DGS / FlashGS (Δ) | min-view parity | speedup at paper "1080p" (ours / paper, ratio) | speedup at 4K (ours / paper, ratio) |
|---|---|---|---|---|
| truck | 25.209 / 25.231 (+0.022) | 45.5 dB | 3.59 / 3.76 (0.95) [1920 wide] | 6.33 / 7.01 (0.90) |
| train | 21.309 / 21.341 (+0.032) | 43.1 dB | 3.66 / 4.20 (0.87) [1920 wide] | 6.57 / 7.48 (0.88) |
| drjohnson | 28.816 / 28.795 (−0.022) | 47.3 dB | **4.21** / 6.18 (**0.68**) [native 1332×876] | 8.95 / 9.99 (0.90) |
| playroom | 29.805 / 29.639 (**−0.166**) | **27.1 dB** | **3.40** / 4.92 (**0.69**) [native 1264×832] | 7.25 / 7.86 (0.92) |

**Against the criteria fixed in advance (review-1).**
- **Quality:** 3 of 4 scenes pass |ΔPSNR| ≤ 0.05 dB.
  - Playroom fails at −0.166 dB.
  - The failure comes from one test view, whose parity with 3DGS is 27.1 dB against a scene mean of 48.9 dB.
  - The known upstream right-edge tile bug is ruled out: 0 such tiles were hit.
  - The likely cause is a culling or intersection difference in the vendored official kernel for that view. It was not investigated further, and the kernel was not patched.
- **Speed ≥ 0.8× the paper:**
  - Passes at 4K for all 4 scenes (0.88–0.92×) and at 1080p for T&T (0.87–0.95×).
  - Fails at Deep Blending's native resolution, the paper's "1080p" column: 0.68–0.69×. At 1920 wide the same scenes give 5.76 and 4.79 (0.93× and 0.97×), but the review mapped this column to native resolution, so 1920 wide does not count.
  - A plausible cause is that small frames are dominated by fixed per-frame costs on the A6000. The camera copy inside the timed call (≈0.12 ms) weighs more at low resolution.
- **Trend:** 4K speedup is higher than 1080p for every scene, and DrJohnson shows the largest gain. Both match the paper.
- **Spread:** only one run was made. The review recommended a second run to measure spread; it has not been done.

**Bottom line.** The speed claim reproduces at high resolution, and quality is unchanged on 3 of 4 scenes. On the two Deep Blending scenes at native resolution, the speedup is about 30 % smaller than the paper's, and playroom has one view where FlashGS's output deviates.
