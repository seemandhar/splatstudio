# ResGS: Residual Densification of 3D Gaussian for Efficient Detail Recovery

Lyu, Cheng, Kang, Chen — ICCV 2025, arXiv 2412.07494. Official code:
https://github.com/yanzhelyu/ResGS (cloned at `source/code`, commit a0db69d, 2026-05-23).
Sources: `source/latex/sec/{3_method,4_experiments,X_suppl}.tex`, `source/paper.pdf`.

## 1. Method summary
- Replaces 3DGS clone/split with **residual split**: a selected Gaussian keeps its scale and
  position, gets its opacity multiplied by beta, and a down-scaled replica (scale / lambda_s) is
  added at a position sampled from the parent Gaussian.
- **Image-pyramid supervision**: L = 3 training stages on images downsampled by a Gaussian
  pyramid (cv2.pyrDown), 1/4 -> 1/2 -> full resolution.
- **Varying gradient threshold**: each Gaussian carries a non-learned fineness level l_i
  (initial 0, child = parent + 1). Each stage has K = 3 substages; at substage k, Gaussians with
  l_i < k get a lower threshold tau / alpha^(k - l_i).
- Built on **AbsGS** in the released code (absolute view-space gradients, periodic opacity
  reduction x0.8 instead of opacity reset, extended beyond densification).

Contributions
- C1 Residual split (Sec. 3.2.1, Eq. 4-5).
- C2 Image pyramid as supervision / coarse-to-fine resolution schedule (Sec. 3.2.2, Eq. 6-7).
- C3 Varying gradient threshold for coarse-Gaussian selection (Sec. 3.2.3, Eq. 8).
- (Pipeline, not claimed as novel) AbsGS gradient + extended periodic opacity reduction (Sec. 4.1, supp.).

## 2. Equations (verbatim)
Eq. 4 (residual split, new Gaussian G_j from G_i):
```
\mathbf{S}_j = \frac{1}{\lambda_s}\mathbf{S}_i,\quad \mathbf{R}_j = \mathbf{R}_i,\ SH_j = SH_i,\ o_j = o_i,\quad \boldsymbol{\mu}_j \sim \mathcal{N}(\boldsymbol{\mu}_i, \boldsymbol{\Sigma}_i)
```
Eq. 5: `o'_i = \beta o_i` (parent opacity, after activation).
Eq. 6-7: `\mathcal{I}_i = \{ P^v_i \}_{v \in V}`, `(H_i^v,W_i^v)=(H_L^v/2^{L-i},W_L^v/2^{L-i})`.
Eq. 8:
```
\tau_{k,i} = \tau if l_i \geq k;   \tau / \alpha^{k-l_i} if l_i < k
```
Code (`scene/gaussian_model.py:adjust_gaussian`): `thr = tau * update_value ** (level - cur_stage)`
for `level < cur_stage`; `update_value = stage_hr_factor ** (1/stage_split) = 2^(1/3)`; select
`grad_abs_mean >= thr` (abs gradient = accumulated `||grad[:,2:]||` / visible count).
Code residual split (`densify_residual_split`): child mean = `R(q_i) @ (s_i * eps) + mu_i`,
eps ~ N(0, I); child log-scale = `log(s_i / 1.6)`; child rotation/SH/opacity-logit copied from
parent *before* reduction; parent opacity `logit(sigmoid(o_i) * 0.3)` with the parent's opacity
Adam moments zeroed (`replace_tensor_to_optimizer_with_mask`); children get zero Adam moments
for all fields; child level = parent level + 1. Every selected Gaussian is residual-split
(no clone/split choice by scale, `percent_dense` unused). After the split, prune opacity < 0.005
(no screen/world size pruning). All stats reset afterwards.
Periodic opacity reduction (`reduce_opacity`): `o <- logit(sigmoid(o) * 0.8)`, opacity Adam
moments zeroed, every 500 iterations while `cur_resolution != 0` and `iteration < 25000`; from
`iteration >= 12000` it is preceded by pruning opacity < 0.005.

## 3. Hyper-parameters (official code overrides paper when they differ)
| name | value | source |
|---|---|---|
| iterations | 30000 | paper §4.1, `arguments/__init__.py` |
| tau (densify_grad_threshold, abs grad) | 0.00067 | paper §4.1, code |
| tau (3D-GS variant) / small variant | 0.00028 / 0.0016 | paper §4.1 (not released as defaults) |
| alpha | 2^(1/3) (stage_hr_factor 2, stage_split K 3) | paper, code |
| lambda_s (residual_split_scale_div) | 1.6 | paper, code |
| beta (opacity_reduce_weight) | 0.3 | paper, code |
| L, K | 3, 3 | paper |
| resolution_scales (pyrDown count per stage) | [2, 1, 0] | code |
| change_iter | [2500, 6000] | code (paper: stage 1 = 2500, stage 2 = 3500 iters) |
| warm_up_iter (no stats/densify after a stage change) | 500 | code (commented out in paper) |
| start_stat / update_from / update_interval / update_until | 500 / 700 / 100 / 12000 | code |
| min_opacity | 0.005 | code |
| opacity reduce factor / interval / until | 0.8 / 500 / 25000 | code (`reduce_opacity`, `prune_until`) |
| position_lr init / final / max_steps | 1.28e-4 / 1.28e-6 / 30000 (x extent) | code (3DGS: 1.6e-4) |
| feature_lr (sh0), shN | 0.002, 0.002/20 | code (3DGS 0.0025) |
| opacity_lr | 0.04 | code (3DGS 0.05) |
| scaling_lr | 0.006 | code (3DGS 0.005) |
| rotation_lr | 0.001 | code |
| lambda_dssim | 0.2 | code |
| SH degree | +1 every 1000 iters, max 3 | code |
| background | black | code (`white_background=False`) |

## 4. Schedules
Literal state machine in `train.py:training` (variables `warm_up_iter`, `cur_level`,
`change_split_level_iter`, `last_change_iter`, `next_change_iter`):
- init: level k = 1, last = 700, next = 2500, substage length = (next - last)//3 = 600.
- Stats accumulate when `500 < it < 12000` and `warm_up_iter == 0`; densify when also
  `it > 700` and `it % 100 == 0`.
- At `it in [2500, 6000]` (after densify of that iteration): resolution index +1, warm-up 500,
  k = 3 * res_index + 1, last = change_iter[res-1] + 500, next = change_iter[res] or 12000,
  substage length recomputed; the camera stack is refilled.
- End of every iteration: warm-up decremented; if warm-up is 0 and `700 < it < 12000`, the
  substage counter is decremented, and when it reaches 0, k += 1 and the counter is reset.
- Resulting renders: it 1..2500 at pyrDown^2 (1/4), 2501..6000 at pyrDown^1 (1/2), >6000 full.
- Opacity reduction at it = 2500, 3000, ..., 24500.

## 5. What it builds on
Released code = AbsGS-based 3DGS fork (`diff-gaussian-rasterization-abs`, AbsGS opacity
reduction). Loss: unchanged 3DGS (0.8 L1 + 0.2 D-SSIM, same-padded SSIM). Initialisation:
SfM points, 3-NN scales, opacity 0.1. No opacity reset; no clone/split; no size pruning.
The paper has three variants (Ours (AbsGS), Ours-Small (AbsGS), Ours (3D-GS)); the per-scene
numbers we target are **Ours (AbsGS)**, which is what the released defaults produce.

## 6. Protocol
Mip-NeRF 360 (9 scenes; images_4 outdoor, images_2 indoor, via `script.py -i`), T&T (truck,
train), DB (drjohnson, playroom). llffhold 8 (`--eval`). 30k iterations, black background,
LPIPS-VGG (`lpips` package in `metrics.py`? — train uses `lpips.LPIPS(net='vgg')`). Evaluation
renders test cameras at the original (finest) pyramid level (`getTestCamerasOrig`). RTX 3090.
Per-scene tables: supp. tab:ttdb_* and tab:mip_*.

## 7. Ambiguities
- Paper says stage 2 lasts 3500 iterations and does not mention the 500-iteration warm-up
  (commented out in LaTeX); code has it -> follow code.
- Learning rates differ from 3DGS in code only (not mentioned in paper) -> follow code.
- The substage counter only starts after iteration 700 and after warm-ups, so substages are
  not exactly even; follow the literal code loop.
- 3D-GS-variant (tau 0.00028, signed gradient) settings are not released; we reproduce the
  AbsGS variant.
- Image pyramid uses cv2.pyrDown (5x5 binomial, BORDER_REFLECT_101, ceil sizes) of the float
  image; cameras keep their FoV (intrinsics scaled with the image size).
