# Optimized Minimal 3D Gaussian Splatting (OMG): implementation spec

Lee, Ko, Park. NeurIPS 2025. arXiv 2503.16924. Official code: https://github.com/maincold2/OMG
(cloned read-only into `source/code`, HEAD of main on 2026-11-07). **The repository has no LICENSE
file**, so its code is read but not copied. The port re-implements the official numerics and cites
the files. The rasterizer and statistics kernels come from our own Mini-Splatting port
(`reproductions/mini-splatting`), which is based on the 3DGS / Mini-Splatting sources under the
Inria licence. The OMG code is itself built on Mini-Splatting (README: "Our code is based on
Mini-Splatting").

## 1. Method summary
- The base is Mini-Splatting (MS): 3DGS densification with blur split, depth re-initialisation every
  5k steps until 15k, then intersection-preserving importance sampling at 15k. The paper keeps
  this pipeline unchanged up to 15k.
- **C1 Local-distinctiveness (LD) importance scoring (§3.3).** At 20k the MS importance (I-bar) is
  multiplied by a power of the mean L1 difference in static feature T against two Morton-order
  "neighbours". The result is then pruned with a CDF threshold τ (XS 0.96 … XL 0.9999). τ is the
  only knob that sets the variant.
- **C2 Compact attribute representation / space feature (§3.1, Eq. 3–4).** From 15k, colour and
  opacity are no longer per-Gaussian. Each Gaussian keeps 3-D static features T and 3-D
  view-dependent features V. A tiny neural field F = MLP_s(γ(p)) (frequency encoding plus a 1-hidden
  64-wide MLP, 13 outputs) is concatenated with T or V. MLP_t and MLP_o (on [F,T]) produce the SH
  DC and the opacity; MLP_v (on [F,V]) produces the 45 higher-SH coefficients. Scale and rotation
  remain per-Gaussian.
- **C3 Sub-vector quantisation (SVQ, §3.2, Eq. 5–6).** In the last 1k iterations (≥29k):
  - scale is split into 3 sub-vectors of length 1 with 2^6 codes each;
  - rotation into 2 of length 2 with 2^9 codes each;
  - [T,V] into 3 of length 2 with 2^10 codes each.

  The codebooks are initialised by K-means once. The indices are then frozen, and only the
  codebooks are fine-tuned (Adam lr 1e-8, `code:train.py`/`apply_svq`).
- **C4 Post-processing / storage (§4.1, supp. "Effect of post-processings").**
  - Positions are cast to fp16, mapped to uint16 and coded with G-PCC (lossless).
  - SVQ indices are Huffman coded; codebooks and MLPs are stored in fp16.
  - Everything is packed into one LZMA file. The decoded model is a plain 3DGS (SH3) model.

## 2. Equations (verbatim from main.tex)
- Eq. 3–4 (`eq:static`, `eq:view`):
  `h^{(0)}_n = MLP_t(cat(T_n,F_n)), o_n = MLP_o(cat(T_n,F_n)), h^{(1,2,3)}_n = MLP_v(cat(V_n,F_n)), F_n = MLP_s(γ(p_n))`
- SVQ: `\hat z = q(z;M) = cat(C^{(1)}[i_1],…,C^{(M)}[i_M]), i_m = argmin_j ||z_m − C^{(m)}[j]||_2^2`.
- Base importance: `\bar I_i = Σ_ρ w_{i,ρ}` if Gaussian i is the max-weight contributor of at least
  one ray, else 0. In code this is MS `intersection_preserving`: the outdoor metric is
  Σ_views accum_weights/area_proj where area_max ≠ 0, the indoor metric is Σ accum_weights, and the
  result is zeroed where accum_area_max = 0.
- LD: `I_i = \bar I_i ( (1/D) Σ_{j∈N_i^K} ||T_i − T_j||_1 )^λ`, with λ = 2, K = 2 (Morton-order
  neighbours) and D = 3.
- CDF threshold (`init_cdf_mask`): sort I+1e-6 ascending and take the cumsum. The split is the first
  index where cumsum/sum > 1−τ. Keep I > vals[split].

## 3. Hyper-parameters (code `arguments/__init__.py` unless noted)
| name | value | source |
|---|---|---|
| iterations | 30000 | args |
| MS densify (from/until/every/grad/min-opacity) | 500/15000/100/2e-4/0.005 | args, train.py |
| depth reinit every / num_depth / num_max | 5000 / 3.5M / 4.5M | args |
| simp_iteration1 (sampling, factor 0.5) | 15000 | args |
| simp_iteration2 (LD prune) | 20000 | args |
| net_itr (MLPs + features introduced) | 15000 | args, supp. |
| svq_itr | 29000 | args, supp. |
| importance_thresh τ | XS 0.96 (default), S 0.98, M 0.99, L 0.999, XL 0.9999 | README, §4.1 |
| lambda_ld λ | 2.0 | args, supp. |
| D (feature dim) | 3 static + 3 view | supp., gaussian_model.construct_net |
| MLP_s (mlp_cont) | tcnn Frequency(16 freqs) → FullyFusedMLP 64, 1 hidden, ReLU, 13 out | construct_net |
| MLP_v / MLP_t(dc) / MLP_o | Network 16→64 (1 hidden, LeakyReLU) → 45 / 3 / 1 | construct_net |
| position input | contract_to_unisphere(xyz.detach(), aabb [-1,1]) | render_imp |
| net optimiser | Adam lr 0.01 eps 1e-15; LinearLR(0.01, 100 it) + MultiStepLR([1000,3500,6000], 0.33), stepped per it > 15000 | construct_net, train.py |
| feature lr (T, V) | 0.0025 | training_setup |
| scale / rotation lr | 0.005 / 0.001 | args |
| xyz lr | expon(1.6e-4 → 1.6e-6 ×extent, 30k); after 15k evaluated at it−15000+5000 | train.py |
| SH | degree 0 until 15k; +1 at 16k, 17k, 18k | train.py |
| SVQ scale / rot / app | len 1 ×2^6, len 2 ×2^9, len 2 ×2^10 | args, supp. |
| K-means | cuML KMeans(n_clusters, max_iter 1000, n_init 1) | gaussian_model.kmeans |
| codebook optimiser | Adam lr 1e-8 eps 1e-15, from 29000 | apply_svq, train.py |
| imp_metric | outdoor (M360 outdoor, T&T), indoor (M360 indoor, DB) | README |

## 4. Schedules (official train.py order inside iteration `it`)
1. LR update, SH up, and `apply_svq` at it == 29000, all before the render.
2. render_imp, loss (L1 0.8 + D-SSIM 0.2), backward.
3. At it == 30000: encode → comp.xz → decode(decompress=False), giving the plain decoded model.
4. MS densification (it < 15000) and depth reinit.
5. At 15000: intersection_sampling plus reinit, then construct_net (T = DC of the reinitialised
   points, V = 0, fresh Gaussian optimiser).
6. At 20000: intersection_preserving with the current MLP opacity, ld_scoring, prune, and a fresh
   Gaussian optimiser.
7. Optimiser steps run for it < 30000. The net steps when it > 15000; the codebooks step when
   it ≥ 29000.

## 5. Builds on
Mini-Splatting (ECCV 2024). The pipeline up to 15k is identical to MS. The MS 1 % CDF prune at 20k is
replaced by LD + τ. The photometric loss is unchanged. There is no opacity reset; MS does none.

## 6. Protocol
- Mip-NeRF 360 (9 scenes), T&T (truck, train), DB (drjohnson, playroom).
- `--eval` (llffhold 8), 30k iterations, black background (`white_background` False), RTX 4090.
- Resolution is not stated in the README. We take the standard 3DGS protocol: images_4 for outdoor
  scenes and images_2 for indoor scenes, matching our runtime.
- Metrics come from the official `metrics.py` (3DGS SSIM and LPIPS-VGG) on renders of the decoded
  `comp.xz` model.
- Sizes are reported in MB = 2^20 bytes of the LZMA file.

## 7. Ambiguities and code-vs-paper notes
1. **LD neighbours (code overrides paper).** `ld_scoring` builds `order = argsort(morton)`,
   `order_l = clamp(order−1, 0)` and `order_r = clamp(order+1, max)`, then compares `T[i]` with
   `T[order][order_l][i] = T[order[order[i]−1]]`. This is the permutation applied twice, not the
   spatial neighbours at rank(i)±1. We default to the official behaviour (`ld_neighbors:
   "official"`) and also implement the paper's intent (`"morton"`) as an option. The paper's
   table "Morton→KNN" shows a ±0.01 dB difference, consistent with the neighbour choice
   mattering little.
2. **The variant threshold for XL** is 0.9999 in the main text and README, but the supplement says
   0.999 for both L and XL. This is irrelevant for XS.
3. **tcnn fp16 MLPs.** We run fp32 MLPs whose weights are rounded to fp16 at encode. tcnn's
   `__sinf` on large arguments (frequency 2^15·π) is approximate, so its high-frequency features
   differ numerically. They are essentially hash-like in both implementations.
4. **G-PCC.** The official flags include `--cabac_bypass_stream_enabled_flag=1`. With our TMC13
   v14.0 build, streams encoded with this flag fail to decode ("bypass buffer exceeded"). The
   decodable stream (flag 0) is <1 % larger on test clouds. We report both sizes and decode the
   flag-0 stream; the coder is lossless either way. The official string also lacks a space
   (`--positionQuantisationEnabled=1--intra_pred_max_node_size_log2=3`); we keep it verbatim.
5. **Final step.** Officially there is no optimiser step at 30000. The runtime steps the
   per-Gaussian Adam before the density hook, so positions get one extra tiny step (lr ≈ 1.6e-6·extent).
6. **K-means.** We use our own torch K-means (k-means++ init, Lloyd until the centre shift
   < 1e-4 relative, max 1000 iterations) instead of cuML's scalable-k-means++.
