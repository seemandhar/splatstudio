# EfficientGS (plan #32): reproduction report

**Outcome: matched.** Mip-NeRF 360, all 9 scenes, 30k steps: PSNR **26.99 vs paper 27.38 (−0.39 dB)**, within ±0.5. Final Gaussian count is **0.30×** our gsplat baseline.

- Paper: *EfficientGS: Streamlining Gaussian Splatting for Large-Scale High-Resolution Scene Representation*, arXiv 2404.12777.
- Official code: github.com/wenkaier/EfficientGS @ 1f70babe (`train.sh` flags).
- Plugin: `reproductions/efficientgs/plugin` (`efficientgs`, freeze sha fbbd1e1c1962b0bb).
- Runs: `runs/efficientgs-mipnerf360/efficientgs/mipnerf360-<scene>/`; baseline `runs/baseline3dgs-mipnerf360-30000/gsplat/`.

## Means vs the paper (Mip-NeRF 360, 9 scenes, test split)
| | PSNR | SSIM | LPIPS |
|---|---|---|---|
| EfficientGS, ours | **26.99** | 0.811 | 0.262 |
| EfficientGS, paper (Table 1) | 27.38 | 0.817 | 0.216 |
| gsplat baseline, ours | 27.28 | 0.812 | 0.259 |
| 3DGS*, paper (authors' retrain) | 27.48 | 0.812 | 0.222 |

**Change vs our own baseline: −0.29 dB PSNR, vs the paper's −0.10 dB against its own 3DGS\*.** That leaves −0.19 dB that the baseline calibration does not explain.

SSIM is level with the baseline (−0.001; paper +0.005). LPIPS is +0.003 (paper −0.006). Our absolute LPIPS is higher than the paper's for both methods. That is the shared runtime/LPIPS calibration offset (gsplat 0.259 vs 3DGS\* 0.222), not something specific to EfficientGS.

The paper reports dataset averages only, so all 9 scenes were run. No per-scene paper numbers exist.

## Per-scene results
| scene | PSNR | Δ vs gsplat | SSIM | LPIPS | gsplat PSNR / SSIM / LPIPS | Gaussians (ours / gsplat) | ratio |
|---|---|---|---|---|---|---|---|
| bicycle | 25.18 | +0.26 | 0.772 | 0.233 | 24.92 / 0.757 / 0.243 | 1.870M / 6.590M | 0.28 |
| flowers | 21.35 | −0.28 | 0.617 | 0.332 | 21.63 / 0.608 / 0.359 | 1.459M / 3.858M | 0.38 |
| garden | 26.59 | −0.30 | 0.851 | 0.152 | 26.89 / 0.860 / 0.131 | 1.242M / 4.765M | 0.26 |
| stump | 26.42 | +0.19 | 0.771 | 0.258 | 26.23 / 0.761 / 0.258 | 1.396M / 4.506M | 0.31 |
| treehill | 22.18 | −0.10 | 0.627 | 0.349 | 22.28 / 0.627 / 0.361 | 1.878M / 5.074M | 0.37 |
| room | 31.14 | −0.38 | 0.915 | 0.303 | 31.52 / 0.916 / 0.298 | 0.368M / 1.727M | 0.21 |
| counter | 28.47 | −0.38 | 0.897 | 0.281 | 28.85 / 0.907 / 0.261 | 0.312M / 1.216M | 0.26 |
| kitchen | 30.31 | −1.03 | 0.913 | 0.182 | 31.34 / 0.926 / 0.162 | 0.380M / 1.416M | 0.27 |
| bonsai | 31.30 | −0.59 | 0.935 | 0.270 | 31.89 / 0.942 / 0.258 | 0.351M / 1.374M | 0.26 |
| **mean** | **26.99** | **−0.29** | 0.811 | 0.262 | 27.28 / 0.812 / 0.259 | sum 9.26M / 30.53M | **0.30** (mean of ratios 0.29) |

Breakdown by scene type:
- Outdoor (5 scenes): mean Δ −0.05 dB, 0.32× Gaussians.
- Indoor (4 scenes): mean Δ −0.60 dB, 0.25× Gaussians.

## Gaussian counts and storage vs Table 4 (resolves the review's major item)
The review asked whether the 1.59× count at 2k steps meant over-densification or a scale error. The 30k data settles it: the early growth is transient.

| stage | ours / gsplat (9-scene mean) | paper Table 4 storage ratio vs 3DGS* |
|---|---|---|
| after selective densification, step 14999 (before the first prune) | **0.75×** (0.63–0.81) | SD only: 544/742 = **0.73×** |
| after the 3 top-K prunes, step 30k, Gaussian count | **0.30×** (0.21–0.38) | SD+GP: 259/742 = 0.35× |
| with sparse SH: estimated storage at 30k | **0.12×** | SD+GP+SOI: 98/742 = **0.13×** |

How the storage estimate is computed:
- Per Gaussian: 14 floats for geometry, opacity and DC colour, plus 3·((d+1)²−1) floats for its SH degree d.
- Final degree histograms are read from the checkpoints. For example, garden ends at degree 0/1/2/3 = 937k/43k/36k/227k; roughly 74% of Gaussians stay at degree 0 and roughly 18% reach degree 3.
- The baseline is 59 floats per Gaussian.

Our checkpoints still store all 15 SH coefficients for every Gaussian (plus Adam state), so the on-disk ratio equals the count ratio (0.30×). The 0.12× is what a sparse export would take.

Both ratios match the paper's ablation closely. Selective densification reduces the count the way the paper claims. Pruning is slightly more aggressive than SD+GP's 0.35×, but the paper's 259 MB is a storage figure, not a count.

## Event log (from the benchmark runs)
- **Densification**: steps 600–14900 with S_g ≥ 0.0007; opacity reset at 3k, 6k, 9k, 12k.
- **Top-K pruning (K=1)**: three passes; the first removes 49–66% of Gaussians.

| scene | 15000 | 18000 | 21000 |
|---|---|---|---|
| bicycle | 4.68M → 2.27M | → 2.00M | → 1.87M |
| flowers | 2.95M → 1.68M | → 1.53M | → 1.46M |
| garden | 2.98M → 1.46M | → 1.31M | → 1.24M |
| stump | 3.47M → 1.78M | → 1.52M | → 1.40M |
| treehill | 4.08M → 2.22M | → 1.99M | → 1.88M |
| room | 1.39M → 0.47M | → 0.40M | → 0.37M |
| counter | 0.91M → 0.40M | → 0.34M | → 0.31M |
| kitchen | 1.12M → 0.47M | → 0.41M | → 0.38M |
| bonsai | 0.96M → 0.44M | → 0.38M | → 0.35M |

- **SH allocation** at the end of step 15499 (official iteration 15500).
- **Sparse SH increments** at 16000, 17000 and 18000. Each selects 20% of Gaussians (+1 rounding by the official ≥-threshold rule).
  - Garden histograms by degree 0/1/2/3: 1.171M/0.293M/0/0, then 1.142M/0.057M/0.264M/0, then 1.125M/0.049M/0.040M/0.250M.
  - At step 18000 the increment runs before that step's prune, matching the official order.
  - All scenes show the same pattern: the high-error Gaussians are chosen again each time.

## Indoor drop (hypotheses only; no runs made for this)
Indoor scenes lose about 0.6 dB (kitchen −1.03) while outdoor scenes are at −0.05. Possible causes:

1. **Pruning is strongest indoors.** The first prune removes 56–66% indoors vs 49–55% outdoors, and indoor ends at 0.25× vs 0.32×. Detailed indoor surfaces (kitchen's shiny objects, bonsai's leaves) may need more than one dominant Gaussian per ray. The paper's K-ablation (Table 5) gains +0.07 dB from K=1 to K=3 on the average.
2. **Weight convention and the alpha cap.** Ranking uses α(1−α)·T, as the official code does, which demotes nearly opaque Gaussians. Under gsplat's 0.999 cap (official: 0.99), such a Gaussian scores about 1e-3 instead of 1e-2. Indoor scenes have many high-opacity surface Gaussians, so the wrong ones may be pruned more often here than in the official code. This is review minor item 1.
3. **Too little view-dependent colour indoors.** About 74% of Gaussians keep degree 0. Kitchen and bonsai are the most specular scenes, and the paper's own r_s ablation (Table 6) loses 0.08 dB at r_s = 0.2 vs 0.5.
4. **White background.** We train over white; the official code uses black. This changes Eq. 9's colour errors at silhouettes (the probe fixture showed edge errors dominating) and the background term of S_g. It is shared with the baseline only for the photometric part.
5. **Paper averages hide the split.** The −0.10 dB average could include a similar indoor/outdoor imbalance that the paper simply does not show.

Cheapest tests to try later: `top_k: 2`, alpha clamped at 0.99 only inside the statistics kernel, and `weight_after_T: false` on kitchen and bonsai.

## Probes (`ss-verify`: declared_checks_passed, 9/9)
- `egs-gradnorm-kernel`: the Eq. 6 CUDA statistic matches a brute-force autograd sum of per-pixel NDC gradient norms (relative error 7e-8). Its summed vectors equal gsplat's `means2d.grad` (1e-7). In a cancellation case S_g = 0.32 while the plain 3DGS statistic is 2e-9.
- `egs-selective-densify`: a Gaussian whose per-pixel gradients cancel is densified with Eq. 6 and not with the 3DGS statistic (Eq. 5) at the same threshold.
- `egs-stats-kernel`: top-K flags (K = 1, 2) and Eq. 9 weights match a brute-force blend exactly, for both weight conventions.
- `egs-topk-prune`: a hidden auxiliary Gaussian is removed at K=1 and kept at K=2.
- `egs-sparse-sh`: per-Gaussian SH colours equal gsplat's at each degree; masked SH bands get zero gradient; the Eq. 9 selection picks the miscoloured Gaussian and caps its degree at 3.
- `egs-schedule`: event steps, SH learning rate 5e-4, and the reset at 15500.
- Base probes pass: state transitions with the `sh_order` field, photometric gradient, renderer gradient.
- Unit tests: 6 passed; ruff clean.
- Smoke tests: garden 2k steps at 22.73 (gsplat 23.04). A 2k run with every event squeezed in ran cleanly.

## Fidelity gaps
1. Shared runtime differences: white background (official: black), random view sampling with replacement, valid-convolution SSIM, density step after the optimizer step, `lpips`-package VGG.
2. Rasterizer numerics follow gsplat: alpha cap 0.999 (official 0.99), and clamped Gaussians get zero position gradient. This interacts with the α(1−α)T weight (see the indoor hypotheses).
3. The Eq. 7/9 weight follows the official code, α·T after the transmittance update, not the paper's α·T. Pruning follows the code (15k/18k/21k), not the paper text (once after 15.5k).
4. Checkpoints keep dense SH coefficients, so the 0.12× storage figure is computed, not measured on disk. FPS and training time were not compared: the GPUs are shared and the timings are not meaningful.
5. The paper's headline large-scale results (Mill19, Foothill-Town; 60k steps, 4K–6K images) are not available and not reproduced. T&T and Deep Blending were not run.

## Review minor items (review-1.json)
- **Weight convention with the 0.999 cap**: documented above as a deviation and a possible cause of the indoor drop. Not changed, to avoid tuning after seeing test results.
- **Kernel precision**: the backward "behind" colour is computed as C_total − accumulated in float32, which can lose precision for nearly opaque front Gaussians. It only affects the densification statistic. Possible fix: double accumulation or a back-to-front pass. Open.
- **White background**: listed in gap 1 (affects S_g and Eq. 9 at silhouettes).
- **Probe coverage**: no end-to-end `after_backward` test of event order or of the tie rule (≥ threshold). The benchmark logs confirm both orders (increment before prune at 18000; histograms sum to n). Open.
- **paper_results.json completeness**: T&T/DB resources and Tables 5–6 are not transcribed. Open.

## Reproduce
    tools/ssenv pip install -e reproductions/efficientgs/plugin
    tools/ssenv ss-inspect efficientgs
    tools/ssenv python -m pytest -q reproductions/efficientgs/plugin/tests
    tools/ssenv ss-verify efficientgs --device cuda --output reproductions/efficientgs/evidence/verify.json
    python3 tools/bench.py launch --exp efficientgs-mipnerf360 --methods efficientgs --dataset mipnerf360 \
        --scenes bicycle flowers garden stump treehill room counter kitchen bonsai --steps 30000 --recipe 3dgs
    python3 tools/bench.py summarize --exp efficientgs-mipnerf360 \
        --paper reproductions/efficientgs/paper_results.json --baseline gsplat
    # squeezed-schedule smoke (all events within 2k steps)
    tools/ssenv python reproductions/efficientgs/plugin/smoke_schedule.py efficientgs \
        ~/datasets/mipnerf360/native/garden <out>
