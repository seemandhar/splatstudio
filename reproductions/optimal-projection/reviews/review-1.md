# Review 1 — optimal-gs (plan 123), arXiv 2402.00752

**Verdict: approved for benchmark (code: 0 critical, 0 major, 4 minor).** One protocol item
(P1, major) is a launch condition: the `optimal-gs-ewa` control has to run with the benchmark.
It is not optional.

Checks run: plugin pytest (7 passed); `ss-inspect` (no issues); two numerical checks (scripts in
the session scratchpad, GPU 6, no training):
- (a) the j12_x typo on real room/treehill views at SfM init;
- (b) the official closed-form d(u,v)/d(mean2D) against exact autograd.

## 1. Projection and per-pixel evaluation vs the official CUDA

| item | official | plugin | status |
|---|---|---|---|
| J = Q·J_p (Eq. 14/15/18), f' = max(512, f) | forward.cu `computeCov2D` | projection.py:12 `tangent_jacobian_exact` (algebraic e_u, e_v) | exact; probe checks it against a verbatim atan2 transcription and the paper's Q J_p |
| low-pass 0.3, det/conic | forward.cu | projection.py `_sym2`, conic | exact |
| radius: naive-EWA `mid` + optimal `det`, `ceil(3√λmax)` | forward.cu preprocess | projection.py:139-146 | exact; tile rect uses the gsplat rule (gap 3) |
| near 0.2 (only frustum test; lateral test commented out) | auxiliary.h:154 | projection.py:111 | exact |
| pixel/mean rays, tangent (u,v) (Eq. 16-17), hemisphere test `mu·t < 1e-7` | forward.cu renderCUDA | sphere_raster.cu:73-90 | exact. I re-derived e_u = (1,0,−a_x)/ρ and e_v = (−a_x a_y, ρ², −a_y)/(ρr) from θ, φ. Pixel convention: gsplat +0.5 with cx = W/2 is the same as Inria `cx = W/2−0.5` with integer pixels. Room/treehill have cx = W/2 exactly |
| α = min(0.99, oG); skip α < 1/255 or power > 0; stop when T(1−α) < 1e-4 (Gaussian excluded) | forward.cu | sphere_raster.cu:118-121, 173-176 | exact |
| gradient through the clamp (dL/do = G·dL/dα, ungated) | backward.cu:745-771 | sphere_raster.cu:288-305 | exact |
| background term in dα | backward.cu `-T_final/(1-α)·bg·dL` | autograd through `rgb + (1−alpha)` plus `v_alpha += T_final·ra·v_render_a` | equivalent |
| d(u,v)/d(mean2D) | backward.cu:700-764, closed form | dual numbers (exact) | **deviates from the official code, negligibly** (m2 below) |

The f' floor is a no-op on both scenes: room fx = 1586, treehill fx = 1054.

## 2. The j12_x typo: keeping it is fine, and the choice does not matter

Measured on room and treehill: all SfM points, 3-NN scales, random rotations, L1 against GT, 5
train views each. I compared the total `means.grad` with `official_cov_grad` true vs false:
- per-Gaussian relative difference: median 2e-7 to 3e-6; p99 2e-4 to 8.5e-3;
- fraction of Gaussians off by more than 10%: at most 0.17%;
- cosine of the whole gradient vector: 0.99943 to 1.000000 (room 0.99943 to 0.999999, treehill 0.99961 to 1.0).

The coefficient itself is wrong by a scene-scale-dependent factor (official/exact p5 to p95:
0.93 to 1.68, because x²z² has different units from x²+z²). It barely matters because the
covariance part of dL/dmean is tiny next to the mean2D part. The default (faithful to the
official code) is acceptable, and `official_cov_grad:false` is not worth a run.

## 3. Background: keep white for iteration 0

The official default is black (arguments/__init__.py:54 `_white_background = False`).
- The projection mechanism does not interact with transparency. ConeGS changed to black post hoc
  because its pruning/recycling depends on transparency. That reason does not apply here.
- Room is indoor (~3% partly transparent pixels, ConeGS diagnosis). Treehill's sky is shared
  identically by the method, the control and the gsplat baseline, so deltas stay valid.
- Switching to black would also need a black-bg gsplat baseline (none exists for these scenes)
  to keep the comparison honest.

Recommendation: keep white. Use black only as a labelled post-hoc variant together with a black
baseline.

## 4. Paper baseline row: confirmed stale; compare gains

`detailed_table.tex` (tab:360_scene_psnr): the 3D-GS row equals the published Inria per-scene
numbers digit for digit (room 30.63, treehill 22.49; `tools/reference_results.json` 30.632).
They were copied from the Inria paper, not retrained.

Our gsplat baseline (`baseline3dgs-mipnerf360-30000`):
- room 31.52 (offset +0.88) already matches the paper's *method* (31.58);
- treehill 22.28 (offset −0.21).

The paper's room gain (+0.95) is therefore mostly an artefact of its stale baseline. Expect our
room gain to be small (≈0 to +0.3). Report:
- (i) absolute PSNR vs paper;
- (ii) gain over the control and over gsplat, against the paper gain (room +0.95, treehill
  +0.28, M360 average +0.27);
- (iii) treehill, the cleaner gain test (offset −0.21 → expected ≈22.56).

Optional: add **counter**. It has the cleanest indoor signal (paper gain +0.35, baseline offset
only +0.15), whereas room and kitchen gains are inflated by their offsets.

## 5. Control `optimal-gs-ewa`: needed (P1, major protocol)

It is not an ablation. The paper's comparison is optimal projection vs EWA *on the Inria
rasterizer*: 0.99 clamp, ungated gradients, near 0.2. Our gsplat baseline differs from the
Inria rasterizer in exactly those semantics (0.999 gate, near 0.01), and also in the
rasterizer code path and tile rect. Only the control (identical kernel, `mode=1`, planar d,
EWA J with the 1.3·tan clamp) gives the paper's like-for-like denominator. The gsplat-baseline
delta mixes in the Inria-vs-gsplat semantics.

Cost: 2 extra runs. The smoke run shows why it matters: at 2k, optimal 23.058 vs control 23.058
vs gsplat 22.98, so the gain over gsplat came from semantics, not projection. Make it mandatory
in bench_request.json:3 ("Optional control …").

## 6. Densification

`gsplat DefaultStrategy` through the base Density, the same as the calibrated baseline:
- grow_grad2d 2e-4 on ‖dL/dpix‖·W/2, which equals Inria's ‖dL/dndc‖;
- refine 500 to 15k every 100;
- opacity reset every 3000 (runtime fix);
- world-size prune > 0.1·extent after the reset;
- screen-size prune disabled (`refine_scale2d_stop_iter = 0`), which matches the official no-op.

Our `means2d` is the [1,N,2] non-leaf (retain_grad works). `radii` [1,N,2] int is zeroed when no
tile is touched, matching forward.cu's `tiles_touched == 0 → radius 0`.

The requires_grad pitfall does not apply: the plugin never rebuilds Parameters (no custom
density), and `_OfficialJacobian` only replaces dJ/dt. Smoke: optimal grows 0.79M vs baseline
0.84M at 2k, plausible.

## Minor issues

- **m1** `sphere_raster.cu:23` and `plan.md` both claim the official d(u,v)/d(mean2D) is "a
  hand-derived closed form of the same derivative". It is not. backward.cu:595 overwrites `t`
  with the *unit* pixel ray, and backward.cu:724-725 (`tx = t.x, ty = t.y`) feed it into a
  formula derived for the z = 1 ray. I verified the formula matches autograd to 1e-15 with the
  z = 1 ray, but not with the unit ray. Effect at realistic offsets (≤15 px):
  - |dL/dmean2D| ratio: median 1.000 to 1.003; p95 1.009 (room) and 1.014 (treehill) at the
    image edges;
  - angle: median 0.15 to 0.2°; p95 0.8° at the edges.

  This is negligible (it slightly inflates edge densification stats). Fix the comment and add
  it as fidelity gap 6. Do not replicate it.
- **m2** The renderer hard-codes white (`renderer.py:95`) and has no `background` option for a
  possible post-hoc black variant. Vol3DGS exposes one.
- **m3** Expected-depth 4th channel: the paper/official code has no depth. It is harmless (no
  loss uses it) but costs ~25% raster bandwidth. Consider `with_depth` only when needed.
- **m4** The run is ~2.4× slower than gsplat (smoke: 592 s vs 248 s for 2k on a shared GPU).
  Budget ~2.5 to 4 h per 30k scene.

## Probe assessment

| probe | detects removal? | note |
|---|---|---|
| tangent-jacobian | yes | fails if J is replaced by EWA (off-axis diff > 0.1) |
| cov-gradient | yes | fails if the custom backward is unwired or a coefficient is wrong |
| raster-forward-vs-dense | yes | optimal vs EWA max diff ≥ 0.05 required |
| raster-backward-vs-dense | yes | float64 autograd reference |
| projection-error | yes | optimal < 0.5× the EWA error off-axis against the exact ray maximum |

## Launch

```
python3 tools/bench.py launch --exp optimal-gs-mipnerf360 --methods optimal-gs optimal-gs-ewa \
  --dataset mipnerf360 --scenes room treehill --steps 30000
```
(Optionally add `counter`. Baseline is the existing `baseline3dgs-mipnerf360-30000`. Put the
experiment on /bigdata per CLAUDE.md.)
