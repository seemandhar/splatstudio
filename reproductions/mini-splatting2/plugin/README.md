# splatstudio-method-mini-splatting2

Native SplatStudio port of **Mini-Splatting2: Building 360 Scenes within Minutes via Aggressive
Gaussian Densification** (Fang & Wang, [arXiv 2411.12788v1](https://arxiv.org/abs/2411.12788v1),
pinned to the preprint; v2 is a later journal version), written against the official code
<https://github.com/fatPeter/mini-splatting2> @ `5ec9202` (`msv2/train.py`, `msv2_d/train.py`,
`config/fast`, `scene/gaussian_model.py`, `submodules/diff-gaussian-rasterization_ms`).
Paper fidelity unverified. **Train with `--steps 18000`.**

| method | official script | importance metric |
|---|---|---|
| `mini-splatting2` | `msv2/train.py --imp_metric outdoor` | M360 outdoor scenes, T&T |
| `mini-splatting2-indoor` | `msv2/train.py --imp_metric indoor` | M360 indoor scenes, Deep Blending |
| `mini-splatting2-d` / `-d-indoor` | `msv2_d/train.py` | MSv2-D: importance pruning instead of simplification |

## Reused from the Mini-Splatting port (plan #28)
Copied, not imported (the two plugins stay independent; `ss_mini_splatting` is untouched):
`csrc/ms_raster.cu` (per-Gaussian blending-weight sum / pixel counts / max-weight area, per-pixel
argmax and mid-point depth points on gsplat 1.5.3 tile lists; adapted by that port from the
official Mini-Splatting `forward.cu`), `csrc/simple_knn.cu` (Inria simple-knn), `csrc/ext.cpp`,
`csrc/LICENSE-3DGS.md` (Inria licence: non-commercial research use), the kernel wrappers in
`kernels.py` and the CDF / sampling / importance helpers in `core.py`. The JIT extension is
named `ss_mini_splatting2_cuda` (built on first use through `tools/ssenv`, ~1.5 min, cached).
The MSv2 rasterizer computes the same statistics as Mini-Splatting's.

## Mechanism → code
- **C1 critical Gaussian identification** `density._culling_with_clone`: over all training views,
  `imp = Σ w`, zero for Gaussians that are never the max-weight Gaussian of a pixel, keep the
  0.99 CDF (`init_cdf_mask`).
- **C2 aggressive clone** `density._clone` / `core.clone_values`: 3DGS-MCMC with 2 copies,
  `α' = 1−√(1−α)`, `s' = α/(2α'−α'²/√2)·s`, opacity clamped to ≥ 0.0051; source overwritten, one copy
  appended with zero Adam moments.
- **C3 pipeline** (`core.events`, `config/fast`): densify (3DGS clone/split ∪ blur split, prune
  α<0.005) every 100 steps 600–2900 except 2000; `culling_with_clone` every 250 steps 500–2750
  except 2000 (also prunes `count_vis ≤ 1` and the 0.1 % tail of Σw); at 2000 intersection
  preserving (0.99) then depth reinit with `num_depth = N` (mid-point depth points ∝ 1−T,
  GT colours, simple-knn scales, opacity 0.1, fresh Adam); 3000: visibility + importance
  sampling (0.6) / MSv2-D importance pruning, SH enabled, fresh Adam; 8000: visibility +
  intersection preserving / importance pruning; 18000 steps.
- **C4 visibility culling** `renderer.MSv2Renderer` + `link.LinkTable`: per view the Gaussians
  outside the 0.99 CDF of that view's weights are not rasterized (subset render, no gradient, not
  updated by the sparse optimizer), from the first clone pass until 13000 (cleared at the 2000
  reinit); gradient statistics on culled views are scaled by `count_vis/(count_rad+0.1)`.
- **Accelerated training** (§6.1, 3DGS-accel): `optim.SparseGaussianAdam` (Taming numerics on
  gsplat's fused selective-Adam kernel, bias correction folded into lr), training views at
  int(W/2)×int(H/2) while step ≤ 3000 (`loss_target` from bicubic-antialiased 8-bit targets), SH
  degree 0 until 4000 then +1/1000 (inactive coefficients and moments zeroed), opacity LR 0.025,
  means LR lr(it) on a 30000-step horizon (official `position_lr_max_steps` not overridden), no
  opacity reset.

All hyper-parameters are in `ss_mini_splatting2.density.DEFAULTS` (official values).

## Fidelity gaps
Optimizer step before density control (official skips Adam on densification steps); white
background vs black; random view sampling with replacement (official shuffled stack, which also
keeps half-res views until the stack empties after 3000); gsplat rasterizer (alpha clamp 0.999,
3σ tiles) vs Taming rasterizer (0.99, opacity-aware extent); runtime SSIM (valid conv) vs
fused-ssim; torch vs PIL bicubic half-res targets; a training view never drawn before step 2000
gets rendered instead of GT colours for its depth points (`density_state.depth_views_fallback`);
times are measured on a shared 8×A6000 host with gsplat (paper: dedicated RTX 3090 + Taming).

## Commands
```bash
pip install -e .            # inside tools/ssenv
ss-inspect mini-splatting2 && python -m pytest -q tests && ruff check .
ss-verify mini-splatting2 --device cuda
ss-train mini-splatting2 --data ~/datasets/mipnerf360/native/garden --output runs/x \
  --steps 18000 --recipe 3dgs --num-gaussians 10000000 --max-gaussians 12000000
python smoke_schedule.py mini-splatting2 <data> <out>   # every event within 2000 steps
```
