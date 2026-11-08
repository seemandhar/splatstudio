# LPM — Improving Gaussian Splatting with Localized Points Management (CVPR 2025 Highlight)

arXiv 2406.04251 · project https://happy-hsy.github.io/projects/LPM/ ·
**official code** https://github.com/Surrey-UP-Lab/GS-LPM (commit 7c06026, cloned to `source/code/`,
built on Inria 3DGS, Inria licence). LaTeX in `source/latex/`, PDF text in `source/paper.txt`.
Feature matching: SuperPoint + LightGlue (https://github.com/cvg/LightGlue, commit 5a650c4,
weights `superpoint_v1.pth`, `superpoint_lightglue_v0-1_arxiv.pth`, in `~/models/lightglue/`).

## 1. Summary and contributions
- Plug-in point-management for 3DGS: find 3D "error source zones" from a view's rendering error
  map + feature matches to a neighbouring view, then do localized densification and localized
  opacity reset there, on top of the usual ADC.
- **C1 Error-map generation** (§3.2 "Error map generation"; code `lpm/utils.py:get_errormap`).
- **C2 Cross-view region mapping** (§3.2 (i); `get_paired_views`, `region_matching.py`):
  SuperPoint+LightGlue matches to up to 3 neighbour views within an angle, precomputed offline.
- **C3 2D→3D zone projection** (§3.2 (ii); `zones_projection.py:region2zone_3d`): rays through the
  perimeter of the paired regions, pairwise closest points, axis-aligned bounding box = zone.
- **C4 Localized point addition** (§3.2 "Points manipulation" (1); `lpm.py:lpm_densify_and_*`):
  inside zones the clone/split gradient threshold is halved; an equal number of lowest-opacity
  points is pruned ("adaptive pruning ... to minimize model expansion").
- **C5 Localized reset of ill-conditioned points** (§3.2; `lpm.py:reset_localized_points`):
  high-opacity Gaussians in front of a zone (in the cone camera→zone) get opacity 0.01.

## 2. Equations / algorithms (no numbered equations in the paper; code is the spec)
3DGS rendering $c(p)=\sum_i c_i\alpha_i\prod_{j<i}(1-\alpha_j)$ unchanged.

**C1** (code): $\hat I = I/(\bar I+0.01)$, $\hat G = G/(\bar G+0.01)$,
$E=\sum_c|\hat I-\hat G|$; $M = E > \mathrm{quantile}_{0.4}(E)$; pad to multiples of 16 (zeros),
16×16 non-overlapping patches; patch significant iff $\sum M > 0.85\cdot256$; error map = significant
patches upsampled, cropped to H×W.

**C2**: connected components (8-conn., cv2) of the error map → bbox $(x_0,y_0,x_0+w,y_0+h)$.
For each, matched keypoints of the current view with $x_0\le x\le x_1, y_0\le y\le y_1$ →
$R_e$ = int bbox of those keypoints, $R'_e$ = int bbox of their matches in the referred view.
Skip pairs where either area is 0. Neighbour views: angle between the centre-pixel ray directions
$<$ `angle` and $>0$; 3 random of them (or the view itself if none); per training iteration one
of these pairs is drawn uniformly.

**C3**: rays $o+t d$ through pixel centres ($\text{ndc}=(2v+1)/S-1$) of the four borders of
$R_e$ and $R'_e$ (top row, bottom row $y_1-1$, left col, right col $x_1-1$); shorter list padded with
its first ray; rays paired by index. Closest points:
$t_a=\frac{((o_b-o_a)\times d_b)\cdot(d_a\times d_b)}{|d_a\times d_b|^2}$,
$t_b=\frac{((o_b-o_a)\times d_a)\cdot(d_a\times d_b)}{|d_a\times d_b|^2}$ (code writes it as
$-t_b$ with $-d_b$); zone = [min, max] over all $2K$ closest points.
(Paper text: cone radius = smallest circumscribed circle, zone = smallest enclosing sphere; code
uses AABBs — code wins.)

**C4** at densification steps (every 100, 500 < it < 15000), *replacing* 3DGS densify_and_prune:
if no zones → nothing (no densify, no prune, stats keep accumulating). Else, with $g$ = mean
view-space grad, $\tau=2\cdot10^{-4}$, $s_{\max}$ = max scale, $\rho=0.01\cdot$extent:
clone $= (g\ge\tau \wedge s_{\max}\le\rho) \vee (g\ge0.5\tau \wedge s_{\max}\le\rho \wedge x\in Z)$,
split (on the post-clone population, clones get $g=0$) analogous with $s_{\max}>\rho$, N=2, scale/1.6.
$n_{extra}$ = #clone ∧ $g<\tau$ + #split ∧ $g<\tau$. Prune: $\sigma<0.005$ ∨ logit ≤ $n_{extra}$-th
smallest logit (if $n_{extra}>0$) ∨ ($s_{\max}>0.1\,$extent if it > 3000). (Screen-size prune is a
no-op: `densification_postfix` zeroes `max_radii2D`.) $x\in Z$: strict inside any AABB.
Global 3DGS opacity reset (every 3000, < 15000) kept.

**C5** at it % `reset_interval` == 0, 500 < it < 15000, using this iteration's rendered view:
per zone: centre $b$, $r=\min(\text{extent}_{x,y,z})/2$, inside $=|x-b|\le r$; cone apex $o$
(camera centre), axis $(b-o)/|b-o|$, $h=|b-o|$, $\cos\theta=\cos(\arctan(r/h))$;
front $=\cos\angle(x-o,\text{axis})\ge\cos\theta \wedge |x-o|\le h \wedge (|x-o|\le h-r$ if $h>r)$.
Union over zones. $k=\lfloor0.7\,n_{inside}\rfloor$; threshold = $k$-th largest logit among inside;
reset = front ∧ logit ≥ threshold → logit := logit(0.01) (in place, Adam state untouched).

## 3. Hyper-parameters
| name | value | source |
|---|---|---|
| iterations | 30000 | §4, code |
| densify from/until/interval | 500 / 15000 / 100 | README command, `arguments/__init__.py` |
| densify grad threshold τ | 0.0002 | code |
| zone grad ratio | 0.5 | `lpm.py` `grad_ratio=0.5` |
| min opacity prune | 0.005 | `points_addition` |
| percent_dense | 0.01 | code |
| opacity reset interval (global) | 3000 | code |
| reset from/until | 500 / 15000 | README |
| reset_interval (localized) | 100 outdoor, 200 indoor | README (+ note) |
| neighbour angle | 45° outdoor, 90° indoor | README (+ note) |
| neighbours per view | 3 (`select_min_num`) | `utils.py` |
| reset top ratio | 0.7 | `reset_localized_points` |
| reset opacity | 0.01 | code |
| error quantile / patch / fill | 0.4 / 16 / 0.85 | `get_errormap` |
| error function | L1 ("diff") | code; tab:ab_error |
| SuperPoint | max_num_keypoints=None, resize 1024 | `lpm.py`, LightGlue default |
| LightGlue | depth_conf 0.9, width_conf 0.95 | `lpm.py` |
| all 3DGS LRs / λ_SSIM 0.2 | unchanged | code |

## 4. Schedules
Densify/LPM-add every 100 in (500, 15000); global opacity reset at 3000…12000; localized reset every
100/200 in (500, 15000); SH degree +1 per 1000.

## 5. Builds on
Vanilla 3DGS (also 2DGS/Mip/Pixel-GS, not reproduced here). Loss, rendering, LRs unchanged.

## 6. Protocol
13 static scenes (M360 ×9, T&T ×2, DB ×2), llffhold8, 30k, 3DGS `full_eval` resolution
(images_4 outdoor / images_2 indoor) → our `mipnerf360`. Black background. RTX 3090.
Reported: per-scene PSNR/SSIM/LPIPS in supp. tables (see `paper_results.json`).

## 7. Ambiguities
- Paper says "add new Gaussian points at the center of the 3D zone" for sparse zones; code does not
  (unused `add_in_center` argument). Follow code.
- Paper: reset-interval ablation (tab:lpm_reset_ini) indoor 100 → 31.03, 200 → 29.85; outdoor
  100 → 24.83, 200 → 25.12 — the *opposite* of the README advice (indoor 200/90°). Which setting
  produced the per-scene tables is not stated. We follow the README.
- Paper: cones / spheres; code: AABBs from closest points of perimeter rays. Follow code.
- Official LPM skips *all* densification at an interval where no zone is found.
- Matching is on full training images at training resolution (SuperPoint resizes to 1024 internally).
