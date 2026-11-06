# FisherRF — implementation spec

Paper: *FisherRF: Active View Selection and Uncertainty Quantification for Radiance Fields using
Fisher Information*, Wen Jiang, Boshu Lei, Kostas Daniilidis, ECCV 2024, arXiv 2311.17874.
Official code: github.com/JiangWenPL/FisherRF @ b747328 (`main` = the 3D Gaussian Splatting
branch; there is no separate branch). Sources: `source/latex/` (arXiv e-print), `source/paper.pdf`,
`source/code/` (official repo, read only), `source/orig-dgr/` (Inria diff-gaussian-rasterization
59f5f77, the base of the modified kernel, for diffing).

## 1. Method summary
- Fisher information of the radiance-field parameters w under a unit-variance Gaussian pixel
  likelihood equals J^T J (J = Jacobian of rendered pixels wrt w); approximated by its diagonal
  plus a prior λI (Laplace approximation).
- Expected information gain of a candidate view ≈ tr(H_cand H_train^{-1}) (upper bound of
  ½ log det(H_cand H_train^{-1} + I)); pick the argmax (next best view).
- Batch selection: greedy, adding each picked view's H to H_train before scoring the next.
- Diagonal Hessians are computed with a modified 3DGS CUDA rasterizer whose backward pass
  accumulates *squared* per-pixel gradients.
- Active training loop on 3DGS: start from 4 initial views; at a schedule, score all remaining
  training views, add the best, retrain; budget 20 views (10 in the 10-view experiment).

Contributions (scope of this reproduction):
- **C1 Fisher statistic rendering** (§3.1, Eq. 7–9, supp. "customized CUDA kernel"): diagonal
  of Σ_pixels ∇_w f ∇_w f^T for every Gaussian parameter, accumulated over views.
- **C2 EIG view selection** (§3.2 Eq. 6, §3.3 Alg. 1): score = Σ_k H_cand[k] / (H_train[k] + λ),
  argmax, greedy batch.
- **C3 Active view-selection training loop** (§4.1 "Experiment settings"): initial views,
  selection schedule, budget, retraining after each acquisition; random-selection baseline.
- Not reproduced (out of scope of plan id 92): active mapping (separate repo, Habitat),
  pixel-wise uncertainty / AUSE on the LF dataset (Eq. 10), Plenoxels and ActiveNeRF variants.

## 2. Equations (verbatim from sec/method.tex)
(1) `- \log p(\by|\bx, \bw) = \left(\by - f(\bx, \bw)\right)^T \left(\by - f(\bx, \bw)\right)`
(2) `\mathcal{I}(\bw) = - \mathbb{E}[\frac{\partial^2 \log p(\by|\bx, \bw)}{\partial \bw^2}] = \bH''[\by|\bx, \bw]`
(3) `\mathcal{I}[\bw^*; \{\bya\}|\{\bxa\}] = \mathbb{H}[\bw^* | D^{train}] - \mathbb{H} [\bw^*| \{\bya\}, \{\bxa\}, D^{train}]`
(4) `\frac{1}{2} \log \det (\bH''[\{\bya\}|\{\bxa\}, \bw^*]\; \bH''[\bw^*| D^{train}]^{-1} + I ) \leq \frac{1}{2} \tr( \bH''[\{\bya\}|\{\bxa\}, \bw^*]\; \bH''[\bw^*| D^{train}]^{-1})`
(5) `\argmax_{\bxa} \tr( \bH''[\bya | \bxa, \bw^*] \; \bH''[\bw^*| D^{train}]^{-1})`
(6)–(7) `\bH''[\by|\bx, \bw^*] = \nabla_\bw f(\bx; \bw^*) ^T \nabla_\bw f(\bx; \bw^*)`
(8) `\bH''[\by|\bx, \bw^*] \simeq \diag(\nabla_\bw f(\bx, \bw^*) ^T \nabla_\bw f(\bx, \bw^*)) + \lambda I`
Alg. 1 (batch): H_0 = H_train; for b=1..B: i = argmax_{x ∉ S} tr(H_x H_{b-1}^{-1}); S ← S∪{i};
H_b = H_{b-1} + H_i.

### What the official code actually computes (overrides the paper where they differ)
`active/H_reg.py`, `diff/cuda_rasterizer/backward.cu:renderCUDAFused` (`acc_func = atomicAddSquare`):
- Render with the modified rasterizer, call `pred_img.backward(gradient=ones)`. The fused kernel
  computes, per (pixel, Gaussian) pair, the full chain to means3D / SH / scale / rotation /
  opacity (activated) for dL/dpixel = 1 on all three channels, and accumulates the **square** of
  each per-pixel gradient. So per Gaussian parameter θ (activated space):
  `H[θ] = Σ_pixels (∂(R+G+B)_p/∂θ)^2` for geometry/opacity, `Σ_p (∂C_{p,ch}/∂sh_ch)^2` for SH.
- PyTorch then chains to the raw parameters **once (not squared)**: opacity logit gets
  `H[σ]·σ'(x)`, log-scale gets `H[s]·exp(x)`; means / SH identity.
- Parameters used: `capture()[1:7]` = xyz, f_dc, f_rest, scaling, rotation, opacity, minus
  `--filter_out_grad rotation` (default) → xyz, f_dc, f_rest, scaling(raw), opacity(raw).
- H_train = Σ over current training views; λ = `reg_lambda` = 1e-6 (paper "λI");
  score(c) = Σ_k H_c[k] · 1/(H_train[k] + λ); `I_acq_reg` False; `I_test` False.
- M=1: `select_single_view` streams candidates; M>1: greedy Alg. 1 with stored candidate H.
- Background = the training background (white for Blender, black for Mip-NeRF 360).

## 3. Hyper-parameters
| name | value | source |
|---|---|---|
| reg_lambda λ | 1e-6 (Blender), 1e-6 Mip360 seq1, 2 Mip360 seq4 | code active_train.py:argparse; scripts/mip360_seq4.sh |
| filtered params | rotation excluded | active_train.py `--filter_out_grad` default |
| init views | train idx 0 + 3 farthest-point (max-min camera-centre distance) | active/schema.py VNSeqMInplace |
| sequential schedule | N=20, M=1, 4 init, interval_epochs=100 → select at it=100·Σ|S| (400, 900, …, 18400) | schema.py V20Seq1Inplace |
| batch schedule | N=20, M=4, 4 init, interval 300 → 1200, 3600, 7200, 12000 | V20Seq4Inplace |
| 10-view | N=10, M=1, 2 init, interval 100 | V10Seq1Inplace |
| after each selection | **restore the initial Gaussians + fresh optimizer** (init.ckpt), base_iter = it−1 (LR, densify schedule restart) | active_train.py:120-126 (paper says "reset the opacity") |
| iterations | 30000 Blender (scripts/blender_seq*.sh), 20000 Mip360 (mip360_seq*.sh); paper: 30000 | scripts |
| sh_up_every | 5000 (absolute iteration; degree reset to 0 on restore) | active_train.py argparse (3DGS: 1000) |
| densify_until_iter | 10000 (cur_iter) | arguments/__init__.py (3DGS: 15000) |
| densify_from / interval / reset | 500 / 100 / 3000; extra opacity reset at cur_iter==500 if white bg | arguments, active_train.py:208 |
| grad thresh / min_opacity / percent_dense | 0.0002 / 0.005 / 0.01 | arguments, argparse |
| LRs | xyz 1.6e-4→1.6e-6 ×extent over 30000 (cur_iter), f 0.0025, f_rest /20, opacity 0.05, scale 0.005, rot 0.001 | arguments |
| λ_dssim | 0.2 | arguments |
| random baseline | `random.Random(seed=0).shuffle(sorted(candidates))[:M]` each round | active/rand_selector.py |
| seed | 0 | scripts |

## 4. Schedules
Effective iteration it = 1..T. If it ∈ load_its: select M views with the model trained up to
it−1, restore init state, base_iter = it−1. cur_iter = it − base_iter drives the xyz LR and
densification. SH degree +1 when it % 5000 == 0 (from the restored degree 0). Last round
(20 views) trains T − 18400 + 1 iterations (11601 for Blender T=30000; 1601 for Mip360 T=20000).
Optimizer step only for it < T.

## 5. Builds on
3DGS (Inria) train.py unchanged otherwise: L1 + 0.2 D-SSIM, clone/split/prune ADC, Adam.

## 6. Protocol
- Blender (Tables 2–4): 8 scenes, 100 training views = candidate pool, all 200 test views,
  800×800, white background, random 100k init (np seed 0), `--eval --white_background`.
- Mip-NeRF 360 (Table 5): 9 scenes (averages only), `-r -1` → width 1600 (paper "1066×1600"),
  images/, llffhold8 test split (candidate pool = llffhold8 train views), black background,
  20000 iterations (paper text says 30000).
- Metrics: test PSNR/SSIM/LPIPS(VGG, lpipsPyTorch) at the final iteration (training_report).
- No per-scene numbers anywhere (no supplementary in the e-print).

## 7. Ambiguities
- Paper "reset the opacity every time we select new views" vs code full restore of the initial
  Gaussians → code followed.
- Paper "30,000 iterations" vs Mip360 scripts 20000 → only Blender (30000 both) is benchmarked.
- `--I_acq_reg` is a bool flag used as a number in the batch branch (`cur_H + True`); off by default.
- `sh_up_every` 5000 is an active_train.py default (3DGS uses 1000).
- The Fisher kernel's chain rule mixes squared (in-kernel) and non-squared (autograd through
  sigmoid/exp) factors; reproduced as is.
