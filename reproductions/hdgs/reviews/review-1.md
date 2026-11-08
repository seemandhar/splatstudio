# HDGS review 1 (splat-reviewer)

**Verdict: APPROVED for benchmark.** 0 critical, 0 major, 6 minor.

The official code was cloned read-only to scratch: `main@b13a359` and `texture@01c6fd4`.

## What was checked against the official code

| item | result |
|---|---|
| Vendored CUDA (`csrc/stage1`, `csrc/texture`) | Byte-identical to the official `cuda_rasterizer/{forward,backward,rasterizer_impl}.cu`, `rasterize_points.cu`, `ext.cpp` and `tex.h`. The setup.py of both branches builds only these files, not the `*_origin` / `*_nocone` variants. The only edit is a `#ifndef SORT_WINDOW` guard in `auxiliary.h`, and `-DSORT_WINDOW=16` is passed (`raster.py:111`). `BILINEAR 1` and `TexRange 4.5` are unchanged. |
| Python wrapper (`raster.py:150-237`) | Arguments are passed in the same order as the official `diff_surfel_rasterization/__init__.py` of both branches. The projection matches getProjectionMatrix with znear 0.01 and zfar 100. The principal point enters P02/P12, and the `ndc2pix` arithmetic checks out (pixel i is at its centre). |
| Losses (`losses.py`) | dist 100 for steps > 3000 and normal 0.01 for steps > 7000, as in train.py and run_360.py. Stage 2 adds no regulariser (`total_loss = loss` in train_texture.py). |
| Densification (`density.py:105-164`) | grad threshold 2e-4, interval 150, from step 500 to 20000, cull 0.01, opacity reset every 3000 to min(o, 0.01). The 19000 / 8e6 skip, percent_dense 0.01 and the world prune (0.1·extent, step > 3000) all match. The split divides scales by 1.6 and samples in-plane (third scale exp(-30)), as in 2DGS `densify_and_split`. The screen-size prune is correctly a no-op (`max_radii2D` is zeroed at `gaussian_model.py:346`). |
| LRs | The base LRs (state.py) equal the 2DGS arguments: 1.6e-4·extent, f_dc 2.5e-3, f_rest 1.25e-4, opacity 0.05, scaling 5e-3, rotation 1e-3, eps 1e-15. The xyz exponential decay over 30k steps is re-applied in after_backward, which runs after the runtime's own means-LR override, so the plugin's value wins. |
| Stage switch (`density.py:120-125`) | The runtime calls opt.step() **before** after_backward. So step 40000 is a normal stage-1 update, followed by the prune, texture init and freeze. Stage-2 iteration t=1 is global step 40001. |
| Stage 2 | xyz, scales, quats, opacities and sh0 get lr 0. With Adam, lr 0 adds exactly 0 (`step_size = -lr/bc`), so the scale-derived texture key cannot drift. shN gets lr 1.25e-4 when t-1 > 1000, i.e. from global step 41002, which matches `run_iter > 1000`. Texture: Adam(8e-3, eps 1e-8) with lr at stage-2 iteration t = 8e-3·0.1^((t-1)//8000), same as StepLR stepped after each iteration. Total 60000 steps = run_360.py `--iteration 60000`. |
| Texture init (`texture.py:26-53`) | W = ceil(s_u·T) and H = ceil(s_v·T); when max(W, H) > 500, each side is scaled by 500 // max; every texel starts at features_dc. All as in `texture_manager.init_texture`. tex_res is 200 outdoor and 500 indoor, per the texture-branch run_360.py. |
| SH colour | Stage 1 evaluates SH at the surfel centre (preprocess). Stage 2 uses `computeTexFromSH`: SH_C0·tex + SH rest at the intersection direction, + 0.5. This comes from the vendored kernel, and the texture-vs-reference probe checks it. |
| requires_grad pitfall | No Parameter is rebuilt by hand. gsplat 1.5.3 `ops.duplicate/split/remove/reset_opa` keep `requires_grad=p.requires_grad` (ops.py:110,166,200). The texture is built with `nn.Parameter(t)`, which gives requires_grad=True even under no_grad. |
| Tests | `pytest tests` passes (6). The verify.json probes all pass. |

## ss-eval / bench.py texture reload: works with plain bench.py

- bench runs `ss-eval --run <abs run dir> --split test --device cuda:0` (bench.py:198), with an absolute run path (`root / method / ...`).
- `_run_from_argv` (`texture.py:68-75`) parses exactly that form.
- `cli.eval` → `runtime.evaluate` → `renderer.render` sees `hdgs_textured == 1` and calls `ensure_loaded`. That reads `checkpoint.pt["density_state"]["texture"]`, recomputes the offsets from the stored scales, and checks the key, the texel count and tex_res. If anything is missing it raises; it never silently falls back to the untextured render.
- `runtime.train` saves `density.state` after the last step. `state["texture"]` aliases the Adam-updated buffer, so the saved texture is the final one.
- The smoke-switch test confirms a fresh-process eval reproduces the in-process validation PSNR exactly (23.2609). That test went through `smoke_switch.py`, but with the same argv form.
- No custom evaluator is needed.

## Fisher score: adjudication

- **Literal reading.** The paper's Eq. U_i is ‖log diag F‖₁ = Σ_j |log F_jj|. That is the literal formula, and the builder's Σ_j log F_jj is a deviation from it.
- **The sign argument holds.** The two scores are equal only when every F_jj ≥ 1. Below 1, the literal score grows as sensitivity falls, so it is non-monotone in sensitivity. It also depends on units (scene scale, image size, number of views).
- **Measured on the smoke-1 checkpoint** (garden, 2k steps, 890k surfels, 41 views):
  - median F_jj is about 75–150;
  - 9–13 % of the entries are < 1, which is exactly the low-sensitivity tail that pruning acts on;
  - the 10 % sets chosen by the two scores overlap only **34 %**;
  - under the literal score, the least-sensitive surfels (F ≪ 1) get large scores and are kept.
- **Conclusion.** Σ log F_jj is the log-det of the diagonal, i.e. the diagonal form of PUP-3DGS's log|∇²L|. It is monotone and invariant to scale up to a constant, so it is the faithful reading of the intent. Keep it, and keep it labelled as a deviation in report.md (it already is). The paper's ablation without pruning is −0.06 dB, so the effect is small either way.

## Memory estimate (texture buffer + grad + 2 Adam moments = 48 B/texel)

Computed with the plugin's `texture_index` on scale distributions from proxy checkpoints (two largest scales):

| scene | tex_res | proxy checkpoint | texels | texture memory |
|---|---|---|---|---|
| garden | 200 | 3DGS baseline / PGSR | 107M / 64M | 5.1 / 3.1 GB |
| counter | 500 | 3DGS baseline / PGSR | 216M / 205M | 10.4 / 9.8 GB |

- Counter is dominated by the large surfels capped at 500²: the top 1 % of surfels hold 30–47 % of the texels.
- Adding Adam temporaries (about 1 extra copy) and stage-2 rendering puts the counter peak at about 15 GB. `min_free_gb 30` is sufficient.
- Counter's checkpoint is about 3–4 GB, and ss-eval loads it twice (runtime + `ensure_loaded`). That is fine.
- Put the runs on /bigdata.

## Issues

**minor-1 `fisher.py:40` (sensitivity)**: Σ log F instead of the literal L1. Justified above and documented. No change needed.

**minor-2 `density.py:38` (`fisher_probes 1`)**: one Rademacher probe per view gives about 20 % relative noise per diagonal entry, so membership at the 10 % boundary is somewhat random. The Fisher pass is about 1 render+backward per view, roughly 1 minute in total. Consider `fisher_probes 4`, which costs nothing material. Optional.

**minor-3 `density.py:193-195` (stage-2 SH-rest delay)**: in the official code, train_texture.py never calls zero_grad on the gaussian optimiser during stage-2 iterations 0–1000. The first f_rest step at iteration 41002 therefore applies the summed gradient of 1001 iterations to the restored Adam moments. The plugin instead uses lr 0 and keeps updating the moments. This is documented (gap 4). The effect is negligible: a single step of about 3·lr.

**minor-4 `texture.py:38-39`**: W and H are clamped to ≥ 1. In the official code, surfels with an aspect ratio above 500:1 can get a zero side. The deviation is harmless and documented in the docstring, but it is not in the report's gap list. Add it there.

**minor-5 SH degree / step-40000 step**: the SH degree increases at step 1001 instead of the official 1000. Also, the official code skips the optimiser step at iteration 40000 (`if iteration < opt.iterations`), while the plugin does that step. Both are one-step differences; the first is documented, the second is not material.

**minor-6 bench_request runtime estimate**: smoke-1 ran 2000 steps in 727 s (0.36 s/step, including the JIT build), and smoke-switch ran at about 0.1 s/step with ≤ 0.9M surfels. At 4–5M surfels with the k-buffer kernel, expect about 8–12 h per scene, not 3–6 h. Plan bench `wait` polling for that.

## Equation coverage

| equation | status | where |
|---|---|---|
| 2DGS ray-splat intersection / blending | exact | vendored kernel |
| Ld, Ln | exact | `losses.py` |
| Eq. avg (5-sample frustum) | exact | kernel; probe checks it |
| per-ray k-sort | exact | kernel, k=16 |
| Eq. Approx / U_i | deviates | Hutchinson estimate, Σ log instead of L1; code unreleased |
| texture bilinear lookup + Ĉ = C[u,v] + SH(d) | exact | kernel; probe checks it |
| two-stage schedule | exact | apart from minor-3 and minor-5 |

## Probes

- **raster-vs-reference**: fails if the sort or the frustum average is removed. It asserts sort_effect and frustum_effect > 1e-2 against an independent dense reference, and that k=4 differs from k=16.
- **texture-vs-reference**: fails if the texture is unused (texture_effect assert) or if the lookup or SH direction is wrong (forward and gradient comparison).
- **fisher-estimator**: compares against exact per-pixel Jacobians; would detect a wrong estimator. Does not test the sign choice (by design).
- **stage-schedule**: tests the LR and freeze logic in isolation. The end-to-end switch and the reload are covered by the smoke-switch evidence instead.

## Launch (two experiments: a second launch into a running exp is refused)

```
cd /home/sejain/splatify
for E in hdgs-mipnerf360-60000 hdgs-indoor-mipnerf360-60000; do mkdir -p /bigdata/sejain/runs/$E && ln -sfn /bigdata/sejain/runs/$E runs/$E; done
python3 tools/bench.py launch --exp hdgs-mipnerf360-60000 --methods hdgs --dataset mipnerf360 --scenes garden --steps 60000 --recipe 3dgs --min-free-gb 30
python3 tools/bench.py launch --exp hdgs-indoor-mipnerf360-60000 --methods hdgs-indoor --dataset mipnerf360 --scenes counter --steps 60000 --recipe 3dgs --min-free-gb 30
```

Compare the results with `baseline3dgs-mipnerf360-30000` (paper: garden 26.98, counter 28.94).
