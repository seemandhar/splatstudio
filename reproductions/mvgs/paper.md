# MVGS: Multi-view Regulated Gaussian Splatting for Novel View Synthesis — implementation spec

arXiv 2410.02103 (v-latest LaTeX in `source/latex/`, PDF `source/paper.pdf`).
Official code: https://github.com/xiaobiaodu/MVGS @ 36d073a (`source/code/`, a fork of Inria 3DGS;
Inria non-commercial research license headers, no separate LICENSE file). Project page:
https://xiaobiaodu.github.io/mvgs-project/.

## 1. Method summary
- Each iteration renders **M training views** and back-propagates the **sum** (not mean) of their
  3DGS photometric losses (multi-view regulated learning, MVRL).
- **Cross-intrinsic guidance (CIG)**: coarse-to-fine image pyramid (focal/principal point scaled by
  1/s), with more views per step at coarse levels: M_s = {48, 24, 12, 8} for s = {8, 4, 2, 1}.
- **Multi-view augmented densification (MVAD)**: halve the densification gradient threshold when
  the sampled views are strongly distinct.
- **Cross-ray densification (CRD)**: find the worst-loss window in every view, cast 4 corner rays,
  intersect corner rays across views, and clone all Gaussians inside the resulting 3D cuboids.

Contributions:
- C1 MVRL — multi-view summed loss (§3.1, Eq. 4, `train.py:86-112`).
- C2 CIG — coarse-to-fine intrinsics/resolution schedule with matched M (§3.2, Eq. 9, `run_360.py`).
- C3 MVAD — threshold halving (§3.3.1, Eq. 10, `train.py:135-148`).
- C4 CRD — cross-ray cuboid cloning (§3.3.2, supp. Alg. 1, `train.py:151-176`,
  `scene/gaussian_model.py:376-510`).

## 2. Equations (verbatim from `sec/method.tex`)
- 3DGS loss per view: $\mathcal{L} = (1-\lambda)\mathcal{L}_1 + \lambda\mathcal{L}_\text{D-SSIM}$, λ = 0.2.
- Eq. (gradient_chain): $\frac{\partial\mathcal{L}_{\text{MV}}}{\partial\theta} = \sum_{m=1}^M\frac{\partial\mathcal{L}_m}{\partial\theta}$
  ("we do not average the gradient but sum it"). Code: `total_loss += loss` then one backward.
- Eq. (CIG): $K'_s = \begin{bmatrix} f/s & 0 & c_x/s \\ 0 & f/s & c_y/s \\ 0&0&1\end{bmatrix}, s\in\{8,4,2,1\}$.
  Code realises it by loading the dataset at `-r 8/4/2` (PIL resize of the **full-resolution** images,
  FoV unchanged) and finally `-r -1` (width 1600).
- MVAD: $\hat{\beta} = \beta - \frac{\beta}{2} H\left(\frac{r_i}{\tau} - 1\right)$, τ = 1, H(0)=1.
  Code: `camera_t = C/‖C‖` (camera centre normalised to the unit sphere, world origin = COLMAP
  origin); `r = ‖t_i − t_j‖` for all ordered pairs i≠j of the step's M views; if **any** r > 1
  (strict `>` in code) then β̂ = β/2 = 0.0001, else β. Rotation similarity mentioned in the text is
  **not** used by the code.
- CRD (supp. Alg. 1 + code):
  1. per view m: L1 map `|gt − render|` [3,H,W]; window (h, w) = (20, 40) (rows, cols), stride = window,
     window starts `range(0, H−20, 20)` × `range(0, W−40, 40)`; take the window with the largest
     mean (first strictly-greater max, initial max 0). Box = [row, col, row+20, col+40].
  2. sort views by their max window loss (descending, stable).
  3. rays: origin = camera centre, direction through pixel centre (x+0.5, y+0.5) (`pix2ndc`),
     normalised, in world frame, computed at the current training resolution.
  4. for every sorted pair (i, j>i): intersect the 4 corresponding corner rays
     (top-left [r0,c0], bottom-right [r1,c1], bottom-left [r1,c0], top-right [r0,c1]); intersection =
     midpoint of the closest points of two lines (`t1 = ((o2−o1)×d2)·(d1×d2)/‖d1×d2‖²`,
     `t2 = ((o2−o1)×d1)·(d1×d2)/‖d1×d2‖²`); skip the pair if any ‖d1×d2‖ < 1e-6.
     **Code quirk**: `box1 = boxes[j]` with `j` enumerating `cams[i+1:]`, i.e. the second view's rays
     are cast from the box of sorted view `j` (not `i+1+j`).
  5. cuboid = axis-aligned min/max of the 4 midpoints; mask = Gaussians with
     `min < μ < max` on all axes (strict).
  6. `densify_and_clone`: clone = (‖ḡ‖ ≥ β̂ ∧ max scale ≤ 0.01·extent) ∨ in-cuboid (in-cuboid
     Gaussians are cloned regardless of size or gradient); then the usual 3DGS split with β̂;
     then prune opacity < 0.005 and, when iteration > 3000, world size > 0.1·extent
     (screen-size pruning uses `max_radii2D`, which `densification_postfix` has just zeroed — a
     no-op, as in Inria 3DGS).

## 3. Hyper-parameters
| name | value | source |
|---|---|---|
| λ_dssim | 0.2 | code `arguments/__init__.py` |
| M per stage (indoor m360) | 48, 24, 12, 8 | `run_360.py` MV_normal; supp. §A "M_s={48,24,12,8}" |
| resolution per stage (indoor) | `-r 8`, `-r 4`, `-r 2` (of full-res), `-r -1` (w1600) | `run_360.py` |
| stage lengths (indoor) | 3000, 3000, 3000, 30000 | `run_360.py` (`--iterations 3000`, final default 30000) |
| outdoor (bicycle, garden, stump) | single stage M=3, `-r -1`, 30000 it | `run_360.py` MV_largest=[1,1,1,3] |
| τ | 1 | supp. §A; code `diffs > 1` |
| window (h, w) | (20, 40) | code `train.py:155-156` |
| densify grad β | 0.0002 | 3DGS default |
| densify from / until / interval | 500 / 15000 / 100 (per stage, restarted) | 3DGS defaults, `first_iter=0` on restore |
| opacity reset | every 3000 it and at it=500 (white background), per stage | `train.py:178` |
| min opacity / percent_dense | 0.005 / 0.01 | 3DGS |
| LRs | 3DGS defaults; position LR schedule restarts each stage (max_steps 30000) | `train.py:79`, restore |
| background | white (`--white_background`) | `run_360.py` |
| view sampling | shuffled stack, M distinct pops per step | `train.py:91-94` |

## 4. Schedules
- Stage-local iteration counter restarts at 0 on every checkpoint restore (`first_iter = 0`):
  densification (500 < it < 15000, every 100), opacity resets (it%3000==0 and it==500), position
  LR (exp decay over 30000) all restart per stage. Parameters, Adam moments, active SH degree and
  densification accumulators are carried over (`capture`/`restore`).
- In 3000-iteration stages the reset at it=3000 happens at the last iteration, so the next stage
  starts from reset opacities; screen/world-size pruning never fires (needs it > 3000).
- SH degree +1 every 1000 iterations (stage 1 reaches degree 3 at it 3000; kept afterwards).
- Densification statistics: `add_densification_stats` is called once per iteration with the
  **last** rendered view's screen-space tensor/visibility (the M−1 earlier views' tensors are
  overwritten), so one view per step contributes to ḡ and the denominator.

## 5. Builds on
Inria 3DGS (train loop, rasterizer, densification, LRs) unchanged except the above.

## 6. Protocol
Mip-NeRF 360, 7 scenes (bicycle, bonsai, garden, stump, kitchen, counter, room; no flowers/treehill),
`--eval` (llffhold 8), images from full-res `images/` with `-r -1` → **width 1600** (our
`mipnerf360-1600`), white background, 30k iterations (+9k coarse for indoor). Metrics: 3DGS
`metrics.py` (LPIPS-VGG). T&T (truck, train) and DB (drjohnson, playroom) reported per scene, but
no official script/config for them exists in the repo.

## 7. Ambiguities / notes
- The paper's 3DGS rows (Table 1, supp. Table) are copied from the 3DGS paper, which used
  images_4/images_2 — a different resolution protocol than MVGS's own `-r -1` runs.
- Paper text says MVAD uses translation + rotation similarity; code uses translation only (used here).
- CRD box index quirk (above) — reproduced by default, switchable.
- Supp. Table "3DGS+Ours" outdoor scenes are *below* the 3DGS row (bicycle 25.08 vs 25.25).
- Multi-scale coarse stages are relative to the full-resolution images (`-r 8` of 3115 px = 389 px),
  i.e. ≈ 1/4, 1/2, 1/1.03 of the final 1600-px images.
