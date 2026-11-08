# Pixel-GS review 1 (plugin `pixel-gs`, official zhengzhang01/Pixel-GS @05c9d55)

**Verdict: approved** — 0 critical, 0 major, 4 minor (one is a launch-budget recommendation).

## Checks against the official code

| Item | Official | Plugin | Status |
|---|---|---|---|
| Accumulation | `gaussian_model.py:405-407`: `accum[vis] += ‖grad[vis,:2]‖ · pixels[vis]`; `denom[vis] += pixels[vis]` | `density.py:52-66` + `core.py:15-27`: `num = ‖g‖·f·m`, `den = m`, index_add on `radii>0` | exact |
| Visibility | `radii > 0` (`gaussian_renderer/__init__.py:100`) | `(radii[0] > 0).all(-1)` (`density.py:55`) | exact (gsplat zeroes both axes when culled) |
| Gradient | signed (non-abs) screen grad of the dummy `screenspace_points`, NDC units | `means2d.grad · (W/2, H/2)` (`density.py:42-44`), `absgrad=False` enforced | exact |
| Depth factor | `diff_gaussian_rasterization/__init__.py:144`: `min(1, (depth/thr)²)`, `depth = t.z` view-space (`backward.cu:168`), `thr = 0.37 · scene.cameras_extent` (`train.py:91`) | `core.py:6-11` clamp((z/(0.37·R))², 0, 1), `z = info["depths"]` (camera z), R = `state["scene_scale"]` = runtime `camera_extent` = getNerfppNorm 1.1·max dist over **train** cameras (`runtime.py:92-97,145`) | exact; R is cameras_extent |
| Where f applies | Only `grad_means2D` returned to the dummy tensor; means3D grads are computed in CUDA before the Python scaling → **statistic only**; denominator unscaled | statistic only, graph untouched | exact; f is not applied anywhere else |
| Normalisation / NaN | `accum/denom`, NaN→0 | `grad2d / count.clamp_min(1)`; count=0 ⇒ grad2d=0 | exact (counts are integers) |
| Threshold | `>= 2e-4` (clone `:366`, split `:354`) | gsplat `> 2e-4` | minor (measure-zero) |
| Clone/split | Inria (split N=2, scale/(0.8N), clone ≤ 0.01·extent) | gsplat DefaultStrategy, same rules, clones not split | exact |
| Prune | opacity < 0.005; world-size > 0.1·extent only after 3000; screen-size no-op | `prune_opa 0.005`, `prune_scale3d 0.1` gated `step > reset_every`, `refine_scale2d_stop_iter 0` (guarded, `density.py:52-53`) | exact |
| Densify window | stats for `iter < 15000`; densify `iter > 500 && iter % 100 == 0` | `step >= 15000` return; `step > 500 && step % 100 == 0` | exact (600..14900) |
| Opacity reset | every 3000 below 15000, to min(o, 0.01) | runtime reset fix `density.py:83-84`, value 2·0.005 | exact |
| Other changes vs vanilla 3DGS | none: `arguments/__init__.py` defaults are vanilla + `depth_threshold 0.37`; script only sets `-r 4/2` | none | exact |
| Pixel count | `forward.cu:364` `atomicAdd(pixels[id],1)` after `power>0` skip, α=min(0.99,·) ≥ 1/255, stop if `T(1−α) < 1e-4` (exclusive) | `adc_stats.cu:83-94` same rule with gsplat numerics (α≤0.999, `≤ 1e-4`) on gsplat tile lists | equivalent up to renderer numerics (minor) |
| requires_grad pitfall | – | population ops via `gsplat.strategy.ops`; no Parameter rebuilt in plugin | ok |

## Issues

1. **minor — launch budget** `bench_request.json` `--max-gaussians 12000000`. Paper Table `tab:efficiency` (tau_table_all.tex): 1.2 GB vs 0.72 GB ⇒ ~1.67× Gaussians. Our pilr bicycle baseline ends at 6.25M ⇒ ~10.4M expected, and outdoor scenes with sparse SfM are where Pixel-GS grows most. `runtime.py:212` aborts the run if the budget is exceeded. The 2000-step bicycle check below can't rule this out: growth only diverges later. Recommend `--max-gaussians 20000000` (the A6000's 48 GB is enough at roughly 250 B/Gaussian plus activations; needs about 25 GB free).
2. **minor — background**: the runtime uses white, official uses black (and on white the official code would also reset opacity at iteration 500, `train.py:118`; the plugin correctly does not, because the paper ran black). The baseline has the same difference.
3. **minor — counts numerics**: gsplat α clamp 0.999 vs 0.99, `≤` vs `<` stop, and per-axis 3σ boxes (gsplat 1.5.3) vs Inria's circle at 3σ of the max eigenvalue. Inria can therefore count a few more edge pixels for elongated, high-opacity Gaussians. Recorded as fidelity gap 6. Small.
4. **minor — `>` vs `>=` threshold** (gsplat `_grow_gs`). Negligible.

## Probes
- `pixel-count-kernel`: exact equality with a brute-force per-pixel loop (911 pairs, max 67) → fails if the count rule changes.
- `pixel-aware-accumulation`: two real views and four flag combinations against an independent formula; it asserts that the pixel-only, depth-only and full statistics each differ from the 3DGS mean (max rel. diff 0.45) → detects removal of C1 or C2.
- `depth-scaling`: analytic (z/γR)² values plus a densify flip → detects a wrong power or a missing clip.
All pass (re-run here on GPU 5); pytest 4/4.

## Short check (2000 steps, pilr bicycle, GPU 5)
Runs cleanly. Gaussians at 1000/1500/2000: 129.7k / 350.1k / 701.3k, against the baseline's 123.5k / – / 698.4k; Pixel-GS only diverges once densification matures. The garden smoke run is already at 1.29× by 2k.

## Launch
```
mkdir -p /bigdata/sejain/runs/pixel-gs-pilr-30000 && ln -s /bigdata/sejain/runs/pixel-gs-pilr-30000 runs/pixel-gs-pilr-30000
python3 tools/bench.py launch --exp pixel-gs-pilr-30000 --methods pixel-gs --dataset mipnerf360-pilr \
  --scenes bicycle bonsai --steps 30000 --recipe 3dgs --min-free-gb 25 --extra "--max-gaussians 20000000"
```
(or keep the bench_request as is, but change its `--max-gaussians` to 20000000). Compare the gain over `baseline3dgs-mipnerf360-pilr-30000` (25.45 / 32.18) with the paper's gain over 3DGS* (+0.105 / +0.513).
