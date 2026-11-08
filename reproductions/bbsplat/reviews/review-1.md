# BBSplat review 1

**Verdict: approved for benchmark.** 0 critical, 0 major, 6 minor issues.

Sources checked: plugin `ss_bbsplat/*`, official `BBSplat@cff8d6a` (`train.py`, `scene/gaussian_model.py`,
`scene/dataset_readers.py`, `utils/camera_utils.py`, `gaussian_renderer/__init__.py`, `arguments/__init__.py`,
`scripts/*.sh`), `diff-bbsplat-rasterization@2dbc336`, LaTeX `arxiv_ICCV.tex` and `rebuttal.tex`, runtime
`splatstudio/runtime.py`, `data.py` and `state.py`, and `tools/bench.py`.

Checks run:
- Every vendored CUDA/C++ file was compared byte for byte with the upstream submodule: all identical.
- A CPU check ran `initialize`, then 3 Adam steps, then a forced relocation of 30 dead billboards, then `add_new`, then one more step. Afterwards every field was still a trainable leaf and was still the tensor registered in its optimizer, and the population went 220 → 231. No training was run.

## 1. Budget (cap = Tab. 1 "Points"; SfM = cap − 10K)

- No script or argument file for the Tab. 1 runs exists in the code or the LaTeX. `scripts/train_all.sh` only has the
  large-budget runs (cap 300K / 160K, `max_read = cap − 10K`, `--add_sky_box`, default `sphere_point` 10000).
  The cap = max_read + sky convention is therefore the published one. The `arguments/__init__.py` default
  `max_read_points = cap_max − 20_000` is not used by any script.
- Truck 30K matches Tab. 1 (T&T "Points 30K", 19 MB). That is the same storage per primitive as M360's 60K / 38 MB,
  so it is consistent.
- **minor (budget):** the LaTeX comment after the BBSplat row of Tab. 1 reads `%7 19 41(62K)`. This suggests the measured
  M360 storage was 41 MB at about 62K primitives, so the real population may have been a few % above 60K
  (for example an initial `max_read + sky` above the cap, since `add_new` never shrinks). The effect is negligible; mention it in the report.
- **minor (truck sky count, ambiguous):** Supp. §Initialization says "for scenarios when we use 20K points and less,
  we reduce the number of sphere points to 2K, and completely disable it when we use less than 10K points".
  - If "points" means the budget (cap 30K > 20K), then 10K sky is right, as in the plugin (`__init__.py:104-106`).
  - If it means the SfM points read (the plugin's 20K), the paper would use 2K sky.
  - Keep the current config as the primary run. If truck falls more than 0.5 dB short, try a labelled variant with
    cap 30K, `max_read_points` 28K and `sky_points` 2000 (cap − sky convention).

## 2. Init and schedule

- FPS (`density.py:66-86`) matches pytorch3d with `random_start_point=False` on the COLMAP-order point list (`points.npz`
  is written in `points3D.bin` order by `colmap_to_transforms.py:154`, like the 3DGS ply).
- The sky sphere matches `gaussian_model.py:176-186`:
  - radius = max|coord| of the FPS-subsampled points (the code uses this; the paper text says "distance from scene centre");
  - points prepended, colour 1.0, SH dc − 0.1;
  - kNN scales on the combined set.
- Textures are logit(0.5 G) / logit(0.1). The ±inf logits become ±1e4, which gives an identical sigmoid output.
- The texture window [500, 30000) and the "freeze all but f_rest after 30000" rule match `train.py:156-163` exactly,
  including the boundary steps. The lr set at the end of step t applies to step t+1.
- **minor:** `density.py:177` sets the `means` lr to 0 for every step > 30000. In the official code, `update_learning_rate(iteration)`
  (`gaussian_model.py:247-253`) overwrites the xyz lr at the start of **every** iteration. `deactivate_gaussians_training`
  zeroes it only once (it returns early after that). So positions keep training at `lr_final·extent` (1.6e-6·extent)
  for steps 30002–31999; only step 30001 has xyz lr 0. Fix: `frozen` should apply to `means` only at step 30001.
  The PSNR effect is tiny, but it is a code-fidelity deviation.
- **σ = 100 (code) vs 500 (paper):** keep 100. It is the default at the published commit, and Supp. Tab. "threshold"
  shows 0.07 dB PSNR spread across σ ∈ [10, 1500] on Truck. Minor; already recorded.
- **Depth-normal regularisation:** keep it off (`losses.py:131`). Evidence:
  - §3.3 (l. 269) says "we use only photometric losses", and a commented-out §3 paragraph says normal/distortion hurt rendering.
  - `train_all.sh` enables it only for DTU (`--lambda_normal 0.05 --lambda_dist 100`).
  - l. 509, the sentence claiming Tabs. 1/2 used it, sits in the DTU mesh paragraph and contradicts the scripts for M360/T&T.
  - An optional `lambda_normal 0.05` ablation is cheap if desired. Gap 5 is recorded correctly.

## 3. MCMC density and regularisers

- The relocate and add steps match `gaussian_model.py:523-604` line for line:
  - dead test `mean(sigmoid α) ≤ 0.005`;
  - sampling `multinomial(mean α)` with replacement and bincount ratio;
  - per-texel 1−(1−a)^(1/(n+1)) written to both dead and source;
  - scales unchanged;
  - Adam moments zeroed only at the sources, while dead slots keep their old moments (official behaviour);
  - growth 1.05·N up to the cap;
  - densify on 500 < t < 25000 with t % 100 == 0.
- MCMC noise is exactly 0: `exp(99.5)` overflows float32, so `op_sigmoid(0)` = 0. Confirmed.
- The texture regulariser matches `train.py:101-104`. `visibility_filter` is `impact > 0` in the official renderer
  (`gaussian_renderer/__init__.py`), the same as `losses.py:144`. The weights `1e-4/1e-4` and `opacity_reg 0.01·mean(α)` match too.
- Adam uses eps 1e-15 for all fields (`state.py:49`), as in the official code.
- **minor (gaps 1–2, already recorded):** the official code skips the Adam step on densify iterations (new Parameters have
  `grad=None`); the plugin steps first and then densifies.

## 4. Evaluation

8-bit quantisation (`texture.py` `quantize`) reproduces `compress_texture`/`decompress_texture`: alpha stored as
(a − G + 1)/2, clip after restore, colour round(c·255). It is applied whenever grad is disabled, and runtime eval is
`@torch.no_grad`. **minor:** the rendered images are not 8-bit PNG-quantised before metrics (runtime-wide, gap 8).

## 5. Resolution

- **M360:** the paper (l. 488) says outdoor ×4 (~1200 px), which is `images_4` 1237×822 here. That is the right choice.
  The official loader run as in the scripts (`-r -1`, `images/`) would give 1600 px wide, which conflicts with the paper text.
  `baseline3dgs-mipnerf360-pilr-30000` is available if a PIL-resize cross-check is wanted.
- **T&T:** the paper says ×2 (~900 px) from its own `Truck_COLMAP_big`. Ours is the 3DGS 979×546 version, close
  enough (minor; it is a different COLMAP/crop).
- cx = W/2 on both scenes, so the kernel's centred principal point is exact.

## 6. Kernel and requires_grad

- The vendored sources are identical to upstream. The Python wrapper's argument order matches
  `diff_bbsplat_rasterization/__init__.py`. Verify reports forward and gradient errors ≤ 3e-6 against the float64 reference.
- requires_grad: `relocate` writes through `.data` in place, and `add_new`/`initialize` rebuild each Parameter from
  `detach()` with `requires_grad=p.requires_grad` through gsplat ops. The CPU check confirmed all 8 fields stay trainable leaves bound to their optimizers.
  The `mcmc-relocation` probe also asserts this.

## 7. Benchmark launch

`bench.py launch` takes one dataset and a method list applied to every scene, and has no per-scene method map. Run two launches:

```
mkdir -p /bigdata/sejain/runs/bbsplat-m360 /bigdata/sejain/runs/bbsplat-tnt
ln -s /bigdata/sejain/runs/bbsplat-m360 runs/bbsplat-m360; ln -s /bigdata/sejain/runs/bbsplat-tnt runs/bbsplat-tnt
python3 tools/bench.py launch --exp bbsplat-m360 --methods bbsplat     --dataset mipnerf360 --scenes bicycle --steps 32000 --min-free-gb 12
python3 tools/bench.py launch --exp bbsplat-tnt  --methods bbsplat-tnt --dataset tandt      --scenes truck   --steps 32000 --min-free-gb 12
```

Compare against the paper's numbers (23.28 / 24.19) and the budget-matched 3DGS-MCMC numbers (23.87 / 23.91). Do not
compare against the unconstrained gsplat baseline.

## Issues

| sev | file:line | problem | official ref | fix |
|---|---|---|---|---|
| minor | density.py:177 | means lr 0 for steps > 30000 | gaussian_model.py:247-253 resets the xyz lr every iter; train.py:162 zeroes it once | means lr = expon(t) except t = 30001 |
| minor | __init__.py:104-106 | truck sky 10K vs Supp. "≤20K points → 2K sphere" (ambiguous) | arxiv_ICCV.tex l. 736 | keep; variant with 28K + 2K sky if truck short |
| minor | __init__.py:100 | M360 population may have been ~62K (`%7 19 41(62K)`) | arxiv_ICCV.tex l. 409 comment | note in report |
| minor | losses.py:28 | σ = 100 vs paper 500 | arguments/__init__.py `max_impact_threshold=100` | keep; recorded |
| minor | plan gap 1–2 | Adam step not skipped on densify iterations | train.py:166-176 | recorded |
| minor | runtime | renders not PNG-quantised before metrics; T&T COLMAP differs | metrics.py | report |
