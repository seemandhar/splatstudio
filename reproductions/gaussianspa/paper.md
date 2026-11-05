# GaussianSpa — implementation spec

Zhang, Jia, Niu, Yin. *GaussianSpa: An "Optimizing-Sparsifying" Simplification Framework for
Compact and High-Quality 3D Gaussian Splatting*, CVPR 2025, arXiv 2411.06019.
Official code: https://github.com/noodle-lab/GaussianSpa (cloned to `source/code/`, Inria 3DGS
licence, research use). Sources: `source/latex/sec/{3_method,4_exp,X_suppl}.tex`,
`source/code/{train_imp_score.py,train_opacity.py,optimizing_spa.py,arguments/__init__.py,*.sh}`.

## 1. Method summary
- Simplification is posed as `min L s.t. ||a||_0 <= kappa` on the opacity vector `a` (Eq. 6).
- An auxiliary copy `z` of the opacities and a scaled dual `lambda` (code: `u`) turn it into an
  ADMM problem (Eq. 9–10): an **optimizing** step (normal training + quadratic penalty pulling
  `a` toward the sparse `z`) alternates with a **sparsifying** step (projection of `a + lambda`
  onto the l0 ball = keep top-kappa, zero the rest) and a multiplier update.
- After the ADMM phase the "zero" Gaussians are hard-pruned to the target count, then a light
  tuning follows.
- Built on **Mini-Splatting** (same starting checkpoint at 15k: blur split, depth reinit,
  importance-sampling simplification at 15k). GaussianSpa replaces Mini-Splatting's 20k 1% CDF
  prune with the ADMM phase 15.2k–25.2k and a hard prune at 25.2k; training runs to 40k.
- Sec. 5.2 "Generality": the sparsifying projection can rank by opacity (`train_opacity.py`)
  or by Mini-Splatting's importance score (`train_imp_score.py`); the README picks one per scene.

Contributions
- **C1 ADMM optimizing step** (Sec. 3.3, Eq. 11–14): penalty `delta/2 ||a - z + lambda||^2`
  added to the loss.
- **C2 Sparsifying step / prox projection** (Eq. 15–16): `z <- prox_h(a + lambda)` (top-kappa),
  opacity- or importance-ranked.
- **C3 Multiplier update** (Eq. 17): `lambda <- lambda + a - z`.
- **C4 Hard prune of "zero" Gaussians** at the ADMM stop iteration + light tuning (Sec. 3.4).
- **C5 Pipeline**: Mini-Splatting stages before 15k, modified 15k sampling (`prune_ratio1`,
  +1-smoothed importance over *all* Gaussians), 40k iterations (code).

## 2. Equations (verbatim from `sec/3_method.tex`)
- Eq. (obj-opacity-form): $\min_{\bm{a},\bm{\Theta}} \mathcal{L}(\bm{a},\bm{\Theta}),\ \text{s.t.}\ \|\bm{a}\|_0\leq\kappa$
- Eq. (indicator): $h(\bm{a})=0$ if $\|\bm{a}\|_0\le\kappa$, $+\infty$ otherwise.
- Eq. (obj-final): $\min_{\bm{a},\bm{z},\bm{\Theta}} \mathcal{L}(\bm{a},\bm{\Theta})+h(\bm{z}),\ \text{s.t.}\ \bm{a}=\bm{z}$
- Eq. (lagrangian): $L=\mathcal{L}(\bm{a},\bm{\Theta})+h(\bm{z})+\frac{\delta}{2}\|\bm{a}-\bm{z}+\bm{\lambda}\|^2+\frac{\delta}{2}\|\bm{\lambda}\|^2$
- Optimizing step: $\min_{\bm{a},\bm{\Theta}}\mathcal{L}(\bm{a},\bm{\Theta})+\frac{\delta}{2}\|\bm{a}-\bm{z}+\bm{\lambda}\|^2$,
  gradient $\partial L/\partial\bm a = \partial\mathcal L/\partial\bm a + \delta(\bm a-\bm z+\bm\lambda)$.
- Sparsifying step: $\bm{z}\leftarrow\mathrm{prox}_h(\bm{a}+\bm{\lambda})$ — keep top-$\kappa$ elements, zero the rest.
- Multiplier: $\bm{\lambda}\leftarrow\bm{\lambda}+\bm{a}-\bm{z}$.
- Base loss $\mathcal L = (1-\rho)\mathcal L_1 + \rho\mathcal L_{D\text{-}SSIM}$, $\rho=0.2$.

Variables: `a` = **activated** opacity `sigmoid(_opacity)` (code `get_opacity`), shape [N,1];
`z`, `u`(=lambda) same shape, no gradients; `delta` = `rho_lr` = 5e-4; `kappa = N - int(r2*N)`.

### Code numerics (`optimizing_spa.py`)
- init (`iteration == start`): `u = 0`, `z = prune(a + u)` without multiplier update.
- `append_spa_loss`: `loss += 0.5*rho*(torch.norm(a - z + u, p=2))**2` — a **sum** over all N.
- `update`: `z = prune(a + u)`; then `u += a - z` (a = current opacity, before the optimizer step).
- opacity projection `prune_z(v)`: `thr = sort(v)[int(r2*N)-1]`; `z = (v > thr) * v`.
- importance projection: `thr = sort(imp)[int(r2*N)-1]`; `z = v`, `z[imp < thr] = 0`;
  `imp` recomputed over all training views at every update (`update_imp_score`, metric from
  `--imp_metric`, scripts pass `indoor` = sum of blend weights; zero where never argmax).
- `adjust_rho` is defined but never called — delta stays 5e-4.

## 3. Hyper-parameters
| name | value | source |
|---|---|---|
| iterations | 40000 | `arguments/__init__.py:69`, both `.sh` (`--iterations 40000`) |
| delta (`rho_lr`) | 5e-4 | `arguments:94` |
| sparsify interval | 50 | `arguments:91` (`SPA_INTERVAL` unused in .sh) |
| ADMM start | 15200 | `arguments:92`; paper says 15K |
| ADMM stop / hard prune | 25200 | `arguments:93`; paper says 25K |
| prune_ratio1 (15k sampling) | per scene, default 0.5 | README table / .sh |
| prune_ratio2 (ADMM target, 1-kappa/N) | per scene, default 0.8 | README table / .sh |
| imp_metric | `indoor` in both scripts | `.sh` |
| simp_iteration1 | 15000 | `train_*.py:359` |
| num_depth / num_max | 3.5M / 4.5M | `train_*.py:360-361` |
| densify from/until/interval/grad | 500/15000/100/2e-4 | 3DGS defaults |
| LR | 3DGS defaults; means lr(it) <15k, lr(it-10000) after; max_steps 30000 | `train_*.py:99-103` |
| SH | degree 0 until 15k; +1 at every 1000 after | `train_*.py:105` |
| opacity reset | none (never called in the loop) | `train_*.py` |

README per-scene recommendation (criterion, r1, r2): drjohnson imp 0.75/0.50; playroom op
0.50/0.77; bicycle imp 0.50/0.72; bonsai imp 0.50/0.80; counter op 0.60/0.50; flowers op
0.50/0.70; garden imp 0.60/0.50; kitchen op 0.40/0.80; room op 0.50/0.80; stump imp 0.65/0.70;
treehill op 0.40/0.75; train op 0.50/0.80; truck op 0.62/0.80.

## 4. Schedules (1-based iteration `it`)
- `it < 15000`: Mini-Splatting densification (blur split, 3DGS clone/split/prune, depth reinit
  at 5000/10000, densify skipped at multiples of 5000 and when N >= 4.5M).
- `it == 15000`: importance (metric) over all train views, `imp[area_max==0]=0`; sample
  `int(N*(1-r1))` Gaussians without replacement with p ∝ imp+1; reinit from their centres
  (scale k-NN, opacity 0.1, SH reset), fresh Adam.
- `it == 15200`: create z (u=0). `15200 < it <= 25200` (opacity) / `< 25200` (imp) with
  `it % 50 == 0`: add penalty to this step's loss. `15200 < it <= 25200`, `it % 50 == 0`:
  z/u update after backward.
- `it == 25200`: hard prune. opacity: keep `topk(a, int((1-r2)*N))`; imp: recompute imp,
  remove `imp <= sort(imp)[int(r2*N)-1]`. Adam state of survivors kept (`prune_points`).
- `25200 < it <= 40000`: plain training ("light tuning").

## 5. Builds on
Mini-Splatting (`ms/train.py`) — everything except the 15k sampling formula and the 20k CDF
prune is identical (diffed against `reproductions/mini-splatting/source/code/ms/train.py`). Its
render_imp rasterizer provides `accum_weights`, `area_proj`, `area_max`.

## 6. Protocol
Mip-NeRF 360 (9 scenes), T&T (train, truck), Deep Blending (drjohnson, playroom). `--eval`
(llffhold 8). Scripts pass `-s <scene>` with default `--images images --resolution -1`: images
wider than 1600 px are PIL-resized to width 1600 → our `mipnerf360-1600` protocol (all 9 M360
scenes); T&T/DB are below 1600 and stay native. Black background. LPIPS VGG
(`lpipsPyTorch`), 3DGS SSIM. 3DGS rows are copied from EAGLES; Mini-Splatting rows retrained.
Per-scene numbers exist for every scene (supplementary tables).

## 7. Ambiguities
- Paper says ADMM 15K→25K; code uses 15200/25200 (code wins).
- Penalty applied only on every 50th step (code), not every step as Algorithm 1 suggests.
- `imp_metric` for the imp-score variant: scripts use `indoor` even for outdoor bicycle (followed).
- Per-scene criterion and ratios only in the README "recommended" table; `.sh` files use other
  defaults (op 0.5/0.8, imp 0.5/0.7). README table is followed for the benchmark.
- Opacity-variant loss window `<= stop`, imp variant `< stop` (each followed).
- The paper does not state delta or the interval; code values used.
- Repo CSVs differ from the paper table on several scenes (stump, T&T, DB); paper table is the target.
