# Review 1 — erank-gs (arXiv 2406.11672, official junhahyung/erank_gs @99fd128)

**Verdict: approved** (0 critical, 0 major, 6 minor). Ready to benchmark.

## Red flag resolved: the 2k deficit comes from the evaluation timing, not from a port bug
The 2k smoke (erank-gs 25.19 vs erank-gs-3dgs 26.68) and all `smoke-diag-*` runs end on step 2000.
Step 2000 is a densify step (`step > 500 and step % 100 == 0`). So the checkpoint that gets evaluated
comes straight out of `densify_and_prune`, with no optimisation steps after it. That step adds
resampled clones, splits with s/1.6, and removes Gaussians with opacity < 0.05. The fixed ADC
densifies more Gaussians (abs-quantile candidates) and moves its clones, so this final step hurts it more.
The training loss shows the same thing. Logged photometric loss is lower for the fixed ADC (mean over
steps >1000: 0.0320 vs 0.0339), yet its train-view PSNR on the evaluated checkpoint is 1.9 dB lower.

Checks run by the reviewer (scan24, same protocol and seed; scripts and runs are in the scratchpad):

| steps | ends on densify step? | erank-gs-adc test / train | erank-gs-3dgs test / train | Δ test |
|---|---|---|---|---|
| 2000 (existing smokes) | yes | 25.19 / 25.86 | 26.68 / 27.80 | −1.49 |
| 2900 | yes | 25.10 / 26.27 | 27.83 / 29.59 | −2.73 |
| **2950** | **no (50 steps after)** | **28.85 / 30.86** | **28.71 / 30.78** | **+0.14** |

With 50 recovery steps the fixed ADC is level with 3DGS or slightly better: +0.14 test PSNR, SSIM
0.914 vs 0.909, LPIPS 0.151 vs 0.169. The fixed ADC has about 25% fewer Gaussians, close to the
paper's "~10% fewer". The red flag does not affect the 30k benchmark, because densification stops
at 15000 and step 30000 is not a reset or densify step.
The `smoke-diag-*` attribution (no-abs / no-resample / min-opacity 0.005 each "recover" part of the
gap) has the same flaw. Each variant only changes how much the final densify step disturbs the
model. Those diagnostics should not be cited as evidence about the method.

## Densification vs official code (line by line): exact
`plugin/ss_erank_gs/density.py` against `source/code/scene/gaussian_model.py:424-509` (`densify_and_*_v2`, `add_densification_stats`) and `train.py:405-417`:
- Stats (density.py:76-88): signed `||g_ndc||₂`. The abs stat is `|grad[:,2]|`, and that channel is written as `Σ_pix |gx|+|gy|` (official
  `backward.cu:548-549`, `dL_dmean2D` is float3, so `grad[:,2:]` is that single channel). gsplat's per-pixel absgrad scaled by
  (W/2, H/2) and taken as an L1 norm gives the same value. The absgrad-units CUDA probe checks this. Visibility is radii > 0. ✔
- `ratio = mean(grads ≥ thr)`, `Q = quantile(grads_abs, 1−ratio)` over all N, Gaussians with denom = 0 included (density.py:120-122 vs :482-483). ✔
- Clone uses the union criterion with max(s) ≤ 0.01·extent. The new mean is `R·N(0, s)` + μ and the original is kept (density.py:46-66 vs :452-474). ✔
- Split runs on the post-clone set with clones padded with 0. It draws 2 samples with s/(0.8·2) = s/1.6 and removes the parent (:127-133 vs :424-450). ✔
- Prune: opacity < 0.05. The world-size prune (0.1·extent) only applies when step > 3000. The screen-size prune is correctly omitted, because `densification_postfix` zeroes max_radii2D (:134-137 vs :492-497). ✔
- Ordering and schedule: densify on 500 < step < 15000 at %100, then reset on %3000 to min(o, 0.01) with moments zeroed (:96-103 vs train.py:405-417). The white-bg reset at densify_from does not apply (bg is black). ✔
- Baseline variant: `INRIA_ADC` = `densify_and_prune` (no abs criterion, exact-copy clones, 0.005), the same as official `no_new_densification`. ✔
- Identical to `reproductions/gof/plugin/ss_gof/density.py:densify_and_prune` minus the 3D filter. erank's render uses `get_scaling`, the filter is never computed, so dropping it is correct. That GOF port matched GOF on M360 at 30k and also showed a 2k smoke deficit (−0.74).
- requires_grad pitfall: `clone_gaussians` builds from `p.detach()` and copies `requires_grad`. gsplat ops are used for split, remove and reset. The density-recipe probe checks that every field is still a trainable leaf after each op. ✔

## Erank loss (Eq. 1–4): exact to code
`losses.py:22-34` matches `utils/get_stats.py:4-14` (q = s²/Σs², temp = 1) and `train.py:353-357`:
λ·mean(clamp(−log(erank−1+1e-7), 0)) + 1·mean(smallest scale). It is active for 7000 ≤ step ≤ 30000.
λ = 0.01 and thin = 1 come from train.sh. Scales are activated (exp) and the erank is computed before rendering.
The paper's Σ and ε = 1e-5 are not used, because the code wins (documented). ✔

## DTU protocol / data
- Split: the official loader is `readColmapSceneInfo(..., eval, llffhold=8)` (dataset_readers.py:183-205): every 8th image of the sorted names is test, with no list of specific test ids. But `train.sh` has **no `--eval`**, so that run had no test cameras. Its wandb report (train.py:446ff) then only evaluates *training* views. The paper says only "split into training and test sets". So llffhold8 is the only split the code supports. → minor 1.
- Resolution: the 2DGS release images are RGBA 1554×1162 (verified), so `-r 2` gives `round(W/2)` = 777×581. The paper's "800×600" is nominal. ✔
- Masking: the RGBA alpha is the mask, and Camera multiplies RGB by it (cameras.py:45-51). Black background, `white_background=False`. The prep reproduces PIL `resize` (same call and filter as `PILtoTorch`) and multiplies rgb by alpha. ✔
- Cameras: PINHOLE fx/fy are scaled by w/W, and the principal point is centred (official uses FoV only, and the COLMAP cx = 777 = W/2 anyway). Reprojecting `points3D.txt` with the native K and viewmat against the `images.txt` observations gives a median error of 0.15–0.28 px on 777×581 (5 views checked). ✔ The scale normalisation is the one already in the release's sparse/0 (the same input the official code reads). `cameras.npz` is not used by either. ✔
- Init: native `points.npz` matches the 31,205 points of `sparse/0/points3D.ply`, which the official `fetchPly` prefers. The txt and ply means are identical. ✔ Extent: runtime `camera_extent` = getNerfppNorm over the train cameras (runtime.py:93-97). ✔

## Minor issues
1. **Split / metric ambiguity** (plan.md gap 1). The paper's DTU PSNR may be training-view PSNR, given train.sh has no `--eval`. Fix: report test PSNR (llffhold8) *and* train-view PSNR (`runtime.evaluate(m, run, split="train")`; `ss-eval` has no train split) for all three methods. Then say which protocol the paper's 3DGS row (30.45 / 38.07) resembles. Judge the result on Δ vs erank-gs-3dgs.
2. **Smoke/diagnostic evidence ends on a densify step** (`tools/smoke_variant.py:19`, `state.json.smoke`). Fix: end smokes off a densify step (e.g. 2050 or 2950). Correct state.json / plan / report: the "2k deficit" is an evaluation artifact. Use the table above.
3. `prepare_dtu_2dgs.py:81-83` re-quantises rgb·alpha to uint8. Official keeps float values after the multiply. The difference is ≤ 0.5/255 per pixel. Note it in the report.
4. Runtime order: the Adam step runs before densify. Official skips the step on densify iterations, because the new Parameters have grad = None. Already documented. Small.
5. gsplat backward drops gradients where α ≥ 0.999 (TODO-runtime). The Inria rasterizer does not. This affects both arms equally. List it as a gap.
6. `config.json` protocol says `"background": "white"` (data.py:121, a static label). Rendering and targets are actually black. The label is misleading in the evidence. Note it in the report (runtime fix).

## Launch
```
python3 tools/bench.py launch --exp erank-gs-dtu --methods erank-gs erank-gs-adc erank-gs-3dgs \
  --dataset dtu-2dgs --scenes scan24 scan122 --steps 30000 --recipe 3dgs
```
(Per CLAUDE.md, first run `mkdir -p /bigdata/sejain/runs/erank-gs-dtu && ln -s /bigdata/sejain/runs/erank-gs-dtu runs/erank-gs-dtu`.) Then compute train-view PSNR for each run (minor 1).
Tests: `pytest plugin/tests` 4 passed; verify JSONs show all probes passed.
