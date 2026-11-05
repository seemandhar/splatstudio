# AbsGS — implementation spec

arXiv 2404.10484 (ACM MM 2024). Official code: https://github.com/TY424/AbsGS (cloned to
`source/code/`, full history in `source/code_full/`; the only code release is commit
`084f94e` "init release", 2024-05-13; later commits only touch README/train.py cosmetics).

## 1. Method summary
- 3DGS densification uses the norm of the per-view **summed** screen-space positional gradient
  `g_i = Σ_j ∂L_j/∂μ_i`. Per-pixel terms of a large Gaussian have mixed signs and cancel
  ("gradient collision"), so large Gaussians in over-reconstructed regions never pass τ_p and
  never split.
- AbsGS sums the **absolute value of each component per pixel** before the norm
  ("homodirectional gradient" ĝ_i) and uses it **only for the split decision**; clone keeps the
  original 3DGS signed gradient and threshold.
- Because ĝ ≥ |g|, the split threshold is raised (0.0004 = "AbsGS-0004", main/best; 0.0008 =
  "AbsGS-0008", lighter). The scale threshold τ_S is lowered from 0.01 to 0.001.
- Everything else is 3DGS (losses, LRs, schedules, pruning).

Contributions
- **C1 Homodirectional view-space gradient** (§3.3, Fig. 2c, Eq. ĝ): per-pixel |∂L_j/∂μ_x|,
  |∂L_j/∂μ_y| summed, then L2 norm, averaged over views like 3DGS.
- **C2 Split driven by ĝ with a larger threshold; clone unchanged** (§4.1 Implementation Details):
  clone if ‖g‖ ≥ τ_clone=0.0002 and max scale ≤ τ_S·extent; split if ĝ ≥ τ_split and max scale
  > τ_S·extent.
- Training setting (not claimed as contribution): τ_S = 0.001 (§4.1 Baselines; Table 2 ablation).

## 2. Equations (paper notation)
3DGS average gradient (§3.1):
`\nabla_{\mu_i}L = \frac{\sum_{k=1}^M ||\frac{\partial L^k}{\partial \mu_{i}^k}||}{M}`,
M = number of views in which G_i is visible during the 100-iteration window.
Split condition (3DGS): `\nabla_{\mu_i}L > \tau_p \text{ and } \Sigma_i^{3d} > \tau_S`.
Per-pixel decomposition (§3.2): `g_{i,x} = \sum_{j=1}^m \frac{\partial L_j}{\partial \mu_{i,x}}`.
Homodirectional gradient (§3.3, Fig. 2c): `\hat g_{i,x} = \sum_j |\partial L_j/\partial \mu_{i,x}|`,
`\hat g_{i,y} = \sum_j |\partial L_j/\partial \mu_{i,y}|`, `\|\hat g_i\| = \sqrt{\hat g_{i,x}^2 + \hat g_{i,y}^2}`,
averaged over the M views exactly as above.
Code (`submodules/diff-gaussian-rasterization-abs/cuda_rasterizer/backward.cu:549-550`):
`atomicAdd(&dL_dmean2D.z, fabs(dL_dG*dG_ddelx*ddelx_dx))` (same for y) — the absolute value of
the full per-pixel screen gradient, in the same NDC-scaled units as the signed gradient.
`scene/gaussian_model.py:479-482`: `accum += ‖grad[:2]‖`, `accum_abs += ‖grad[2:]‖`, `denom += 1`
over visible Gaussians. gsplat `absgrad=True` computes the identical quantity (`means2d.absgrad`).

## 3. Hyper-parameters
| name | value | source |
|---|---|---|
| τ_clone (signed, clone) | 0.0002 | paper §4.1 ("clone follows 3D-GS"); `code:arguments/__init__.py:89` |
| τ_split (abs, split) | 0.0004 (AbsGS-0004); 0.0008 variant | paper §4.1/Table 1; `code:arguments/__init__.py:90` |
| τ_S (percent_dense) | 0.001 | paper §4.1; `code:arguments/__init__.py:83` |
| densify interval / from / until | 100 / 500 / 15000 | paper §4.1 "stop after 15k"; code `:84-88` |
| opacity reset | every 3000 to ≤0.01 | 3DGS; code |
| prune opacity | 0.005; world-size >0.1·extent after 3000 | code `train.py:129-130`, `gaussian_model.py:471-475` |
| iterations | 30000 | paper §4.1 |
| LRs, λ_dssim=0.2, SH schedule | 3DGS defaults | code `arguments/__init__.py:75-84` |

## 4. Schedules
Densification active for iterations (500, 15000), every 100; reset opacity every 3000;
big-Gaussian pruning after iteration 3000. No AbsGS-specific schedule.

## 5. Builds on
3DGS (official INRIA code base). Losses, renderer forward pass, LRs, SH schedule unchanged.

## 6. Protocol
Mip-NeRF 360 (9 scenes; images_4 outdoor / images_2 indoor as 3DGS), T&T truck/train,
DB drjohnson/playroom; 3DGS llffhold8 `--eval`; 30k iterations; black background
(`white_background` default False); LPIPS via the 3DGS `lpipsPyTorch` VGG. V100 GPU.
Per-scene tables in supplementary for all 13 scenes (SSIM/PSNR/LPIPS/memory).
Baseline rows: "3D-GS" = numbers copied from the 3DGS paper; "3D-GS*" = authors' retrain with
τ_S=0.001.

## 7. Ambiguities / code-vs-paper
- **Code release has extra defaults not used in the paper** (README: "minor modifications to
  improve performance in large scenarios. None of these modifications were utilized in our
  paper"): (a) SfM points filtered at load (`max_err=2, min_track_length=3`), (b)
  `use_reduce=True`: every 500 iterations (during densification) opacities clamped to ≤0.8
  (`reduce_opacity`), (c) weight-based pruning (`use_prune_weight=False` by default). We follow
  the paper (none of these); (b) is available as an off-by-default config `reduce_every`.
- Paper text writes `>`; code uses `>=` for both clone and split (immaterial).
- Which variant is "AbsGS": both are reported; AbsGS-0004 is the best and the code default →
  method `absgs`; `absgs-0008` registered as variant.
- Supplementary DB LPIPS for AbsGS-0004 S=0.001 (0.156/0.179) contradicts main-table 0.236.
- Relation to SplatStudio `gsplat-absgrad`: that config uses |·| gradients for **both** clone and
  split with one threshold 0.0008 and τ_S=0.01. AbsGS differs in all three: clone uses signed
  gradient @0.0002, split uses abs @0.0004, τ_S=0.001. Only the renderer (gsplat `absgrad=True`)
  is reusable unchanged.
