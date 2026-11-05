# Taming 3DGS: High-Quality Radiance Fields with Limited Resources — implementation spec

Mallick*, Goel*, Kerbl, Steinberger, Carrasco, De La Torre. SIGGRAPH Asia 2024. arXiv 2406.15643.
Sources: `source/latex/arxiv.tex` (LaTeX), `source/paper.pdf`, official code
`source/code/` = github.com/humansensinglab/taming-3dgs @ fd0f7d9 (main branch; rasterizer
submodule `submodules/diff-gaussian-rasterization`). **Code overrides paper where they differ.**

## 1. Method summary
- 3DGS (Inria code base) with the clone/split densification replaced by a *budgeted,
  score-guided* densification: at each densification step a target count from a parabolic
  growth curve is reached by multinomial sampling (without replacement) of clone/split
  candidates, weighted by a per-Gaussian score.
- The score sums, over 10 random training views, median-normalized per-Gaussian attributes
  (gradient, opacity, depth, radius, scale) and per-pixel accumulations (pixel count, distance
  to centre, saliency (L1 + edge map), blending weight), multiplied by the view's photometric loss.
- "High-opacity Gaussians" after 15k (abs activation; see §7 — the released code freezes opacity).
- Speed: per-splat-parallel backward, separate SH tensors, fused SSIM (all numerically
  equivalent), SH bands ≥1 updated by a separate Adam once every 16 iterations (changes math),
  sparse Adam (released, *not* used for the paper numbers: `--optimizer_type default`).
- Two configurations: *budget* (`Ours`, small counts, interval 500, multiplier mode) and *big*
  (`Ours (#3DGS)`, final count = 3DGS's count per scene, interval 100, final_count mode).

Contributions
- C1 Predictable growth curve with a user budget (§3.2 "Predictable Model Growth", Eq. 2).
- C2 Steerable score-based densification: saliency map (Eq. 3), score (Eq. 4, Alg. 1),
  multinomial resampling of clone/split candidates, score-guided partial pruning (§3.3).
- C3 High-opacity Gaussians from 15k (§3.4).
- C4 Batched SH updates every 16 iterations with separate SH Adam, SH lr / opacity lr changes (§4.2, §5).
- C5 Speed-only kernels: per-splat backward (§4.1), fused separable SSIM, separate SH tensors (§4.2), sparse Adam (code).

## 2. Equations
Growth curve (paper Eq. 2, as printed — garbled):
`A(x) = \frac{B-S-2N}{N^2}x^2 + 2x + B`
Code (`utils/taming_utils.py:get_count_array`, authoritative):
```
B = int(S * multiplier)  (mode multiplier)  |  B = budget (mode final_count)
N = (densify_until_iter - densify_from_iter) // densification_interval
k = 2 (B - S) / N ;  a = (B - S - k N) / N^2 = -(B - S)/N^2
values[x] = int(a x^2 + k x + S),  x = 0..N-1
```
At the j-th densification call (j = 0,1,..) the target is `values[j+1]`.

Saliency (Eq. 3): `S_v = 1_ROI ⊙ (λ1 L1(v, r_v) + λ2 E(v))`, λ1 = λ2 = 0.5 in the paper.
Code (`get_loss_map`): `S_v = 50 * minmax(mean_c |r_v - v|) + 50 * minmax(FIND_EDGES(gray(v)))`
(PIL `ImageFilter.FIND_EDGES` 3x3 Laplacian on the 8-bit grayscale GT; min-max normalized
per image). ROI = all ones.

Score (Eq. 4, Alg. 1): `S_g = Σ_{i∈views} F(∇_g, c_g^i, D_g^i, s_g^i, B_g^i, z_g^i, o_g, s_g)`.
Code (`compute_gaussian_score`):
```
norm_w(x) = w * x / median(x[x > 0])  on x > 0, 0 elsewhere (NaN -> 0)
g = norm_25(grad) + norm_100(opacity) + norm_5(depth) + norm_10(radius) + norm_25(prod(scale))
p = norm_50(dist_accum) + norm_10(saliency_accum) + norm_0.1(count) + norm_50(blend_accum)
S_g += 50 * P_i * (p + g)   for Gaussians visible (radii > 0) in view i
P_i = 0.8 L1 + 0.2 (1 - SSIM_fused)  of view i (current render)
```
Per-pixel accumulations are over contributing (alpha ≥ 1/255, before T cut-off) pixels:
count += 1, dist += ||μ2D − pixel||, saliency += S_v(p), blend += α T.
grad = 3DGS mean NDC-space positional gradient norm accumulated since the last densification.
depth = view-space z (0 if culled), radius = 3σ screen radius in px.

Densification (`gaussian_model.py:densify_with_score`):
```
q = ||grad|| >= 0.0002 ; clone = q & max(scale) <= 0.01 extent ; split = q & max(scale) > 0.01 extent
budget = min(T, n_clone + n_split + N_cur)
clone_budget = ((budget - N_cur) * n_clone) // (n_clone + n_split) ; same for split
clone: multinomial(score * clone, clone_budget, replacement=False) -> duplicate
split: multinomial(score * split, split_budget, replacement=False) -> 2 children, N(0, scale)
       samples rotated, scale / 1.6, same opacity; parents removed
prune_mask = opacity < 0.005  (| max(scale) > 0.1 extent | max_radii2D > 20  if step > 3000)
if densify_call < 27 and int(0.5 |prune_mask|) > 0:
   sel = multinomial(w, int(0.5 |prune_mask|), replacement=False),
         w[i] = 1/(1e-6 + score[i]) for i < len(score), 0 for the appended children
   remove prune_mask & sel
```
(The pruning weights index the *post-densification* order with pre-densification scores; the
max_radii2D test is a no-op because stats are zeroed by `densification_postfix` before it —
both inherited/introduced in the released code; reproduced as-is.)

High opacity (§3.4): after 15k, opacity activation `abs`, alpha clamped. Code
`modify_functions()`: `_opacity = abs(sigmoid(_opacity))` as a plain tensor that is no longer
the optimizer's parameter ⇒ opacities are frozen at their 15k values for the rest of training.

## 3. Hyper-parameters
| name | value | source |
|---|---|---|
| iterations | 30000 | arguments/__init__.py |
| densify_from / until | 500 / 15000 (`iteration > 500`, `< 15000`) | arguments, train.py |
| densification_interval | big: 100, budget: 500 | full_eval.py |
| budget (big, final count) | bicycle 5987095, flowers 3618411, garden 5728191, stump 4867429, treehill 3770257, room 1548960, counter 1190919, kitchen 1803735, bonsai 1252367, truck 2584171, train 1085480, playroom 2326100, drjohnson 3273600 | full_eval.py |
| budget (budget, multiplier) | outdoor 360: 15, indoor 360: 2, T&T: 2, DB: 5 | full_eval.py, paper §5.2 |
| cams | 10 random distinct training views per densification | train.py `--cams` |
| score weights | view 50, edge 50, mse 50, grad 25, dist 50, opac 100, depth 5, loss 10, radii 10, scale 25, count 0.1, blend 50 | train.py:`score_coefficients` (paper text: grad 50, count 0.1, dist 50, saliency 10, blend 50, depth 5, opacity 100, scale 25, λ=0.5) |
| grad threshold | 0.0002 | gaussian_model.py |
| percent_dense | 0.01 | arguments |
| min_opacity prune | 0.005 | train.py |
| big-point prune | world scale > 0.1 extent, screen 20 px (no-op), after step 3000 | train.py, gaussian_model.py |
| score-guided prune | 50% sample, first 27 densification calls only | gaussian_model.py |
| opacity reset | every 3000 while < 15000, to min(o, 0.01) | train.py |
| ho_iteration | 15000 | train.py |
| sh_lower | SH rest (bands 1–3) Adam step every 16 iters (grads accumulated) | train.py, README/full_eval `--sh_lower` |
| lr: means | 1.6e-4 → 1.6e-6 × extent | arguments |
| lr: sh0 | 0.0025 | arguments |
| lr: shN | shfeature_lr/20 = 0.005/20 = 2.5e-4 (paper: "four times, 0.001") | arguments, gaussian_model.py:training_setup |
| lr: opacity | 0.025 (half of 3DGS) | arguments, paper §5 |
| lr: scale / rot | 0.005 / 0.001 | arguments |
| loss | 0.8 L1 + 0.2 (1 − SSIM) (fused SSIM) | train.py |
| optimizer_type | default (dense Adam) for paper results | train.sh |
| background | black (no `--white_background`) | full_eval.py |

## 4. Schedules
- 1..15000: densification statistics accumulated every iteration; stats reset at each densification.
- Densification at iterations 600, 700, …, 14900 (big; 144 calls) or 1000, 1500, …, 14500 (budget; 28 calls).
- Opacity reset at 3000, 6000, 9000, 12000.
- Score-guided pruning only for densification calls 0..26.
- 15000: opacity frozen ("high opacity" switch).
- SH degree +1 every 1000 iterations; SH-rest optimizer steps at iteration % 16 == 0.

## 5. What it builds on
Inria 3DGS code (initialization from SfM, 3-NN scales, opacity 0.1, LR schedule, L1+D-SSIM,
opacity reset, clone/split operators). Only the selection of densified Gaussians, their number,
the pruning and the opacity/SH optimization changed.

## 6. Protocol
Mip-NeRF 360 (9 scenes; outdoor images_4, indoor images_2), T&T (truck, train), Deep Blending
(drjohnson, playroom); 3DGS llffhold-8 split; 30k iterations; PSNR/SSIM/LPIPS(VGG, `lpipsPyTorch`).
RTX A4500. **Only per-dataset averages are reported** (Table 1); per-scene numbers exist only
for the T&T ablation (budget mode; Table 2).

## 7. Ambiguities / paper–code disagreements
1. Growth curve formula: printed Eq. 2 is inconsistent; code parabola used.
2. Score weights: paper grad 50 vs code 25; code adds radius term (10) not in the paper; paper's
   λ1=λ2=0.5 vs code 50/50 (only relative weight matters inside saliency → but scale multiplies
   the saliency term). Code used.
3. High opacity: paper describes trainable abs-activated opacity possibly > 1; code freezes
   opacities at 15k (they stay in (0,1)). Code used (config `high_opacity: "freeze"`).
4. SH lr: paper says 4× (0.001) — code gives 0.005/20 = 2.5e-4 (2× 3DGS's 1.25e-4). Code used.
5. Pruning sampling indexes post-densification order with pre-densification scores; screen-size
   prune is dead code. Reproduced as in code.
6. `torch.multinomial` without replacement does not fail when fewer positive weights than samples;
   it fills with zero-weight indices (so the budget is always met). Reproduced.
7. Saliency L1 uses the first render; the second render (`pixel_weights`) is the same image —
   one render suffices.
