# Review 1 — aaa-gaussians (ss_aaa_gaussians)

**Verdict: APPROVED for benchmark** — 0 critical, 0 major, 6 minor.

Sources checked: official `DerThomy/AAA-Gaussians` @ dc5b3f2 (`source/code`: train.py,
arguments/__init__.py, scene/gaussian_model.py, gaussian_renderer/__init__.py,
utils/mip_filter.py, full_eval.py, configs/aaa.json), rasterizer @ d03ce2d
(`source/code/submodules/diff-gaussian-rasterization`), and `ubc-vision/3dgs-mcmc` HEAD (shallow
clone to scratch) for the MCMC comparison. Runtime: splatstudio runtime.py/state.py/losses.py,
gsplat 1.5.3 `strategy/mcmc.py`, `strategy/ops.py`.

## Checks

| item | official | plugin | status |
|---|---|---|---|
| Vendored CUDA | d03ce2d | `diff -rq` of cuda_rasterizer/, rasterize_points.cu, ext.cpp: identical | exact |
| Python wrapper arg order (fwd/bwd), filter3D not differentiated | diff_gaussian_rasterization/__init__.py:73-97,126-150 | raster.py:112-147 | exact |
| Splatting settings | configs/aaa.json (HIER=3, PTD_MAX=3, 64/8/4, all 4 culling flags, load_balancing, proper_ewa_scaling, eval_3D, new_aabb) | raster.py:30-39 | exact |
| Background | `white_background=False` → black (train.py:47) | __init__.py:36 black | exact |
| Camera | FoV model, centred pp, znear 0.01/zfar 100 | raster.py:78-109 (asserts centred; native garden cx=648.5=W/2) | exact |
| Resolution | full_eval.py: `-i images_4` outdoor / `-i images_2` indoor, `-r -1` (≤1600 w → no resize) | native garden 1297×840 (images_4); room images_2 1557 w | exact |
| Init | opacity 0.5, scale log(sqrt(dist2)·0.1) (gaussian_model.py:137-142) | density.py:199-202 + runtime knn_log_scales (mean of 3-NN d², clamp 1e-7, sqrt, log) | exact |
| LRs | xyz 1.6e-4·extent→1.6e-6, f_dc 2.5e-3, f_rest /20, opacity 0.05, scale 5e-3, rot 1e-3, eps 1e-15 | state.py:35-50, runtime.py:153,209 | exact |
| Loss | 0.8 L1 + 0.2 (1−fused_ssim) + 0.01·|σ(o)|.mean + 0.01·|exp s|.mean (train.py:101-108) | base photometric + losses.py | exact (SSIM valid conv = runtime gap) |
| Relocate/add | 500<it<25000, %100, dead σ(o)≤0.005, multinomial w/ replacement, N=bincount+1, clamp [0.005,1−eps], target min(cap,1.05N), zeroed moments of sources | gsplat MCMCStrategy (mcmc.py step_post_backward; ops.relocate/sample_add) | exact |
| Caps | full_eval.py:21-24 | density.py:127-132 (garden 5,834,784; room 1,593,376), resolved from `--data` basename (bench passes native/<scene>) | exact |
| Noise | `randn·op_sigmoid(1−o; k=100,x0=0.995)·noise_lr·xyz_lr`, then `L Lᵀ @ noise` (train.py:126-133) | gsplat inject_noise_to_position (same op_sigmoid, covar = R S² Rᵀ) | exact |
| 3D filter | `compute_3D_filter(trainCameras)` only inside `if iteration in saving_iterations`; full_eval passes `--save_iterations -1` → only at 30000. Never recomputed during training (filter_3D=None until then; unlike Mip-Splatting's per-densify recompute) | density.py:232-233 at step == --steps; zeros during training (kernel `filter_3D ? filter_3D[idx] : 0` + fmaxf → zeros ≡ None) | exact; smoke checkpoint: filter_3d >0 for 99.999 %, median 2.2e-3 |
| SH view-dir gradient quirk | inside kernel backward | vendored unmodified → kept | exact |
| requires_grad pitfall | — | gsplat ops only; in-place `fill_/add_` under no_grad on leaves; runtime check density.py:228-231; smoke checkpoint `opacities.requires_grad=True` | ok |

### Is AAA training "plain MCMC"? Yes.
Diffing the official AAA `training()` against ubc-vision/3dgs-mcmc `train.py`: the only differences
are the renderer call (`splat_args`), `fused_ssim` instead of `ssim` (numerically the same
same-padded 11×11 Gaussian SSIM), and the `compute_3D_filter` at save. `arguments/__init__.py`
(all LRs, noise_lr 5e5, regs 0.01, densify 500/25000/100) and `create_from_pcd` (o=0.5, s·0.1)
are the same in both repos. So the paper's AAA < MCMC gap (garden −0.37, room −0.80) is
caused by the **renderer** (3D evaluation + always-on k=0.3 per-view dilation with
perpendicular scaling, and the train-view filter floor at test time), not by a different training
recipe. Nothing is missing from the plugin's training side. Suggestion: also run
`gsplat-mcmc`-style control (same caps) on garden/room so the report can separate renderer vs
runtime effects (mh-3dgs has one).

## Minor issues

1. **SH degree one step late** — runtime.py:197 `(step-1)//1000` vs train.py:84
   (`iteration % 1000 == 0` before rendering → degree 1 at it 1000). Runtime-level, listed as gap 2. Negligible.
2. **Densify-step optimizer order** — runtime runs Adam then density (runtime.py:204-211);
   official relocates/adds first and the rebuilt Parameters have no `.grad`, so Adam is
   effectively skipped on the 245 densify iterations (gaussian_model.py:282,333). Gap 1. Negligible.
3. **Final step** — official skips optimizer step and noise at it 30000 (train.py:124) and
   computes the filter on it-29999 means; plugin applies step+noise then filter
   (density.py:226-233). Negligible (lr 1.6e-6·extent).
4. **Noise lr off by one step** — after_backward sees the lr already set for step s+1
   (runtime.py:209) vs official `xyz_lr` of iteration s. Factor 0.01^(1/30000). Negligible.
5. **Runtime protocol gaps** (already listed): views sampled with replacement vs official
   shuffled stack; SSIM valid conv vs fused_ssim same padding; LPIPS package. Affect absolute PSNR
   both for method and baseline — compare deltas.
6. **Dropped base `renderer-gradient` probe** (__init__.py:83-86) — acceptable: the 3D-eval
   kernel's means2D grad is unused by MCMC, and `aaa-backward-reference` checks the kernel gradients
   (means, log-scales, quats, opacity, sh0) against autograd of an independent float64 reference
   and fails for the volume-scaled ablation (probes.py:108-131). Recommend recording this in report.md.

## Probe assessment
- forward/backward reference: fail on Eq. 8 / no-scaling / 2D-EWA variants (probes.py:101-103,131) — detects removal.
- perp_scaling, adaptive_filter (z² law + floor), frustum_bounds (old bounding mass, cull count), filter_train (independent loop, unseen→0, only at final step), mcmc_schedule (init, cadence, cap, reg value): each fails if its mechanism is removed.
- pytest: 5 passed. Smoke (garden 2k): 22.81 vs gsplat 23.02, same-count gsplat-MCMC 22.83 — consistent.

## Operational risks (not fidelity)
- Speed: smoke took 308 s / 2k steps at ≤288k Gaussians on a busy GPU; garden ends at 5.8 M
  Gaussians with hierarchical sort → expect a long run (many hours). Use a generous wait.
- Memory: request ≥24 GB free.

## Launch
```
python3 tools/bench.py launch --exp aaa-gaussians-30k --methods aaa-gaussians \
  --dataset mipnerf360 --scenes garden room --steps 30000 --recipe 3dgs --min-free-gb 24
python3 tools/bench.py summarize --exp aaa-gaussians-30k \
  --paper reproductions/aaa-gaussians/paper_results.json \
  --baseline baseline3dgs-mipnerf360-30000
```
(Per CLAUDE.md, create the run dir on /bigdata and symlink into runs/ first.)
