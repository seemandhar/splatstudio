# Review 1 — sort-free-gs (LC-WSR) / sort-free-gs-dir

**Verdict: approved for benchmarking with changes. 0 critical, 2 major, 6 minor.**
Neither major blocks a benchmark launch. Fix M1 before the 30k runs if possible (one line), and change the scene set (M2).

Sources checked: camera-ready LaTeX (`sec/3_method.tex`, `appendix/exp.tex`, `appendix/derivative.tex`, `rebuttal/rebuttal.tex`), the unofficial code and rasterizer (`source/code_unofficial*`), every plugin file, gsplat 1.5.3 `ProjectionEWA3DGSFused.cu`, and the runtime `state.py`/`runtime.py`/`backends/gsplat.py`.

## Equation coverage

| Eq. | status | code | notes |
|---|---|---|---|
| (7) WSR | exact | `wsr.py:compose`, `renderer.py:173`, `csrc/wsr_raster.cu` | `C = (c_B w_B + Σ c_i α_i w_i)/(w_B + Σ α_i w_i)`: the background sits in both numerator and denominator, as in the paper. It matches unofficial `forward.cu:482-487`, which replaces a zero denominator with 1e-4. The final image is clamped to [0,1] (rebuttal, and unofficial `render()`). |
| (2)+(11) α_i = u_i·G_i | exact | `wsr_raster.cu:32-42`, `wsr.py:view_dependent_opacity` | u = 0.1 + Y(dir; H) is then multiplied by the Gaussian kernel G inside the kernel and is **not** clamped. This matches the paper ("maximum opacity u_i is not clamped") and the unofficial code (SH-DC init 0.1/C0, so u starts at 0.1). SH uses the same active degree as colour (unofficial `gaussian_renderer:62`). |
| (10) LC | exact | `wsr.py:depth_weight` | `w = max(0, 1 − z/σ)·v_i`. v_i is a learnable per-Gaussian value, stored as `opacities` and passed through a sigmoid; init 0.1, lr 0.01, exactly as unofficial `_opacity`/`vi_lr`. d is camera-space z from gsplat (rebuttal: "same depth used for sorting"). The depth gradient flows into the means, as in unofficial `backward.cu:638`. The roles of u and v are swapped relative to the unofficial code, but only their product enters Eq. 7 (paper.md §7.1). |
| (8) DIR | deviates (documented) | `wsr.py:depth_weight` | `w = v_i` keeps v and the view-dependent opacity. This equals unofficial "w/o D". See m5. |
| (9) EXP | missing (documented) | — | Averages only; not a per-scene target. |
| Derivatives (appendix) | exact | `wsr_raster.cu:174-191` + autograd for the quotient | I checked the conic and means2d chain operator by operator (σ = ½ΔᵀconicΔ, gsplat convention). Probe `wsr-backward-vs-dense` compares the kernel against float64 autograd on unclamped u ≤ 2.5. |

The kernel footprint is consistent with the projection. gsplat 1.5.3 builds the per-axis box as `ceil(3.33·sqrt(Σxx))`, which contains the Mahalanobis ≤ 3.33 ellipse used in `wsr_alpha`. The output therefore does not depend on the tile layout.

## Issues

### Major

**M1 — the global-row mean trick is not equivalent to a scalar Adam parameter while N changes.**
- Where: `renderer.py:132-134` (`params[name].mean()`); the claim is made in `density.py:280-284` and `plan.md` ("Adam is scale-invariant ⇒ same update").
- Problem: each row receives the gradient g/N(t). Adam's step is m̂/√v̂. That ratio is scale-invariant only if N is constant over the β2 memory (~1000 steps). Under densification N grows from about 140k to about 5M. The second moment v keeps the older, larger g²/N² values, so the step shrinks.
- Evidence: I simulated Adam on a noisy gradient, scaled by 1/N(t) using the garden N(t) trajectory from `runs/baseline3dgs-mipnerf360-30000/gsplat/mipnerf360-garden/training.jsonl`. The effective step, relative to a scalar Adam, was 0.61 at step 1k, 0.26 at 2k, 0.23 at 3k, 0.36 at 5k, 0.75 at 8k, and about 1.0 only after 13k. So σ and w_B are learned with a 2–4× smaller learning rate during steps 1k–7k. This is exactly when σ moves (10 → 34.6 by step 2k in the smoke run). Pruning, which shrinks N, causes the opposite error.
- Fix: make the gradient independent of N. Use `p[0]` in the renderer (only row 0 gets the true scalar gradient g), and have `sync_globals` copy row 0's value and moments to all rows. After a sync every row is identical, so row 0 is always a valid source, even after pruning. Alternatively, keep `mean()` and scale it as `p.mean() * N - p.detach().mean() * (N - 1)`. The value is unchanged and each row's gradient becomes g. Adam on identical rows that all get g then equals a scalar Adam with eps 1e-15 (same as the unofficial code's Adam).
- Expected PSNR impact is probably small (≤0.1 dB). The fix is cheap and the docstring claim is currently false.

**M2 — garden + truck cannot tell a working WSR from the baseline.**
- Where: `bench_request.json`.
- Problem: the paper's per-scene deltas over 3DGS (appendix Table 6) are garden +0.37 and truck +0.10. Both sit inside the run-to-run noise of our runtime. A run that silently degraded to near-3DGS behaviour would still "match".
- Evidence: WSR's characteristic signature is large losses on the unbounded outdoor scenes: bicycle −1.05, stump −1.16, flowers −1.07. Gains on train (+0.84) and the indoor scenes are confounded, because our gsplat baseline already beats the old 3DGS row there (e.g. room 31.52 vs 30.63). The unofficial port (README) gets garden 27.07 and bicycle 23.89 at the 3DGS protocol.
- Fix: keep garden. Add **bicycle** (paper 24.20; baseline 24.92 vs 25.25, so a Δ of −1.05 is expected). Use **train** (21.93, Δ +0.84) instead of, or in addition to, truck. Report per scene against the unofficial numbers as a second reference.

### Minor

**m1 — background colour c_B.**
- Where: `renderer.py:202`.
- c_B is hard-coded to white. In WSR, c_B is a model parameter inside Eq. 7, not a compositing convention. mipnerf360 and tandt images have no alpha channel, so the runtime's "white" convention does not constrain it.
- The paper (3DGS eval) and the unofficial code (`train.py`: `white_background=False`) use black.
- Fix: expose `background` in the renderer config and default it to 0.0. The impact is small, since w_B → 0.0025 by step 2k, but the paper names "dark object in front of light background" as its failure mode (`appendix/exp.tex:38`).

**m2 — `cull_nonpositive`.**
- Where: `renderer.py:156-160`.
- Culling u ≤ 0 matches the unofficial code (`forward.cu:216-218`). That is the only implementation shown to reach roughly paper numbers, so keep it as the default. The paper says "unconstrained", but literal negative u is not tested.
- The `cull_nonpositive=False` path is unsafe. `compose` uses `clamp_min(den, 1e-4)`, which maps negative denominators to 1e-4 and blows up the output. If this path is ever run as an ablation, it needs a sign-preserving guard.
- Culled Gaussians (15% of u-DC ≤ 0 at step 2k) never get a gradient again and are never pruned. The unofficial code behaves the same way. The paper says pruning "needs to be extended" but does not specify how. Record as a fidelity gap; log the count at 30k.

**m3 — α skip threshold.**
- Where: `wsr_raster.cu:41`.
- The plugin skips a Gaussian when |u·G| < 1/255. The unofficial code skips when |u·v·(1−z/σ)·G| < 1/255 (`forward.cu:228,463`). At init (v = 0.1) the unofficial rule is about 10× stricter, and it also cuts all gradient to low-w Gaussians.
- The paper does not state a threshold. The plugin's choice is defensible and already documented (gap 3). Keep it, but note the difference when comparing Gaussian counts.

**m4 — densification.**
- Removing the opacity prune and reset matches the paper's wording ("removed the pruning based on the opacity threshold").
- Keeping the 3DGS world-size prune is the more literal reading of "maintained without any modification". The unofficial code removes it (`gaussian_model.py:592-597`). This is fine; record it.
- The `requires_grad` pitfall is handled: `opa_sh0`, `opa_shN`, `wsr_sigma` and `wsr_wbg` go through gsplat ops, and `density.py:336-339` guards leaf/requires_grad after every count change.

**m5 — the DIR ablation is unofficial "w/o D", not necessarily the paper's DIR-WSR row.**
- The unofficial README shows no LC-vs-D gap (27.21 vs 27.12 average), while the paper reports −1.2 dB (Table 4).
- Expect sort-free-gs-dir ≈ sort-free-gs. Report that this is consistent with the unofficial code, not as a failed ablation.

**m6 — w_B can be pushed negative.**
- w_B has lr 0.01, is unconstrained, and has already decayed to 0.0025 by step 2k. If it goes negative, pixels with a small S_w get a clamped denominator. The unofficial code has the same exposure.
- Log w_B, σ and the fraction of pixels with S_w + w_B < 1e-3 in the 30k runs. Don't add a constraint the paper doesn't have.

Other minor deviations, already documented: gsplat near plane 0.01 vs Inria 0.2; footprint 3.33σ vs unofficial 3σ (no effect on the sum beyond the 1/255 cut).

## Probe assessment

| probe | would it fail if the mechanism were removed? | why |
|---|---|---|
| wsr-forward-vs-dense | yes | Float64 dense reference, plus a permutation check, plus a check that the result differs from alpha compositing on a fixture that is verified to contain occlusion. It does not cover `compose`/w_B; `test_compose_eq7` covers that on CPU. |
| wsr-backward-vs-dense | yes | Compares every gradient output (means2d, conics, features, u, w) against autograd. |
| lc-weight | yes | Independent formula and the zero-beyond-σ check. It does not test M1 (gradient magnitude). |
| view-dependent-opacity | yes | Independent gsplat SH check, a view-change check, and u > 1 reachable. |
| recipe | partly | Checks row sync after a duplicate. It would not catch M1. Add a check that a row's grad equals the scalar grad (independent of N). |

## Launch commands

Run after the M1 fix, or label the runs "pre-M1".

```
python3 tools/bench.py launch --exp sort-free-gs-mipnerf360 --methods sort-free-gs sort-free-gs-dir --dataset mipnerf360 --scenes garden bicycle --steps 30000 --recipe 3dgs
python3 tools/bench.py launch --exp sort-free-gs-tandt --methods sort-free-gs sort-free-gs-dir --dataset tandt --scenes truck train --steps 30000 --recipe 3dgs
```

Compare against `baseline3dgs-mipnerf360-30000` (garden 26.89, bicycle 24.92) and `baseline3dgs-tandt-30000` (truck 25.21, train from that run). Paper Δ: garden +0.37, bicycle −1.05, truck +0.10, train +0.84. Unofficial absolute values: garden 27.07, bicycle 23.89.
