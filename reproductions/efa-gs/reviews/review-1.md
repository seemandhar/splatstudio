# EFA-GS review 1 (splat-reviewer)

Target: `reproductions/efa-gs/plugin` (`efa-gs`, `efa-gs-base`), checked against official
jcwang-gh/EFA-GS @57f330f `Mip-Splatting/` (train.py, scene/gaussian_model.py,
scripts/run_mipnerf360.py, submodules/diff-gaussian-rasterization/cuda_rasterizer/backward.cu).
Checks run: `pytest` (14 passed); code read line by line against the official code. I ran no GPU
job: the builder's 2k smoke (evidence/test.json) already exercises both methods.

**Verdict: approved = false** (0 critical, 1 major, 4 minor). The single major issue is a one-line
renderer config change. Once it is fixed the plugin can launch without another review round.

## Base = later official Mip-Splatting with GOF abs-grad densification: confirmed
- `densify_and_prune(_lff)` (gaussian_model.py:651, 850): `ratio = mean(‖g‖ ≥ τ)`,
  `Q = quantile(g_abs, 1−ratio)`, clone/split when `g ≥ τ ∨ g_abs ≥ Q`. The rasterizer writes
  `Σ_pix |∂L/∂x_ndc| + |∂L/∂y_ndc|` into `grad.z` (backward.cu:594, ddelx_dx = 0.5W). This is the
  later autonomousvision/mip-splatting GOF densification, not paper-era 1ecf7d5. The claim holds.
- Port: `density.py:139-145` takes `|absgrad_x|·W/2 + |absgrad_y|·H/2`, which equals the official
  scalar. `lfcf.py:113 quantile_linear` matches torch.quantile linear interpolation. Clone uses
  max s ≤ 0.01·extent. Split uses N=2, s/1.6, σ(o) unchanged, and padded zero grads for the clones
  (`density.py:217-235`). All exact.
- 3D filter: recomputed at init (`set_training_cameras`), after every densify (500<it<15000,
  %100), and every 100 steps in (15000, 29900). No update at 15000 (`mip_filters.py:104`,
  train.py:186-193). Coefficient √0.2·d/f, near 0.2, margin 0.15, max focal_x. Exact.
- 2D Mip filter: gsplat antialiased mode with eps2d 0.1 = `--kernel_size 0.1`. Filter-aware opacity
  reset at 3000/6000/9000/12000, run after densify and filter recompute (as train.py:188). Prune
  rules: σ(o) < 0.005 unfiltered; world size > 0.1·extent only when it > 3000; NaN. The screen-size
  prune is a no-op in the official code too (max_radii2D is zeroed by the postfix). Exact.
- `efa-gs-base` faithfully reproduces the official code path with interval_times = ∞ (every densify
  takes the `islff=False` branch).

## LFCF vs `densify_lff` (gaussian_model.py:758-848): exact
- Selection: `sel = g≥τ ∨ g_abs≥Q`, computed on the **LFCF** accumulators and the Q of those
  accumulators. Enlarge = `sel ∧ ¬P`, plus `sel ∧ P ∧ ¬decent`. Shrink = `sel ∧ P ∧ decent`, where
  `decent = g ≤ |g_prev−tol| ∨ g_abs ≤ |g_abs_prev−tol|` (`lfcf.py:125-145`). This matches
  official lines 767-781.
- Depth factor: `c = θ·c_min + (1−θ)·c_max` with θ = log-minmax of (min visible z / max focal).
  Nearest gets 1.5, deepest gets 1.0. The paper prints the labels swapped; the code is followed
  here, which is correct. The enlarge then uses `log c·frac`.
- Annealing: `frac = 1−(it−500)/14500` (percent_lb = ln1/ln1.5 = 0), so the factor decays to 1 by
  15000. The code's decay direction is followed. Correct.
- Scale strategy, enlarge: per-axis coefficients (1, −1/3, −2/3) on axes sorted ascending.
- Scale strategy, shrink: −½(log c + ln 2) times (1, ½, 0) on axes sorted descending. It applies to
  every decent Gaussian, before the random draw, as official lines 806-819 do.
- Edits are in place on `scales.data`, and the Adam moments are kept (= `set_attributes`).
- Split probability: `θ(ub−lb)+lb` with lb = 1−(it−500)/14500, drawn over the intersection.
  Exactly 1 child per split. The child uses the shrunk scale and the parent is kept (N=1, no prune).
- Bookkeeping:
  - prev grads are set to the current LFCF means; children get 0.
  - P is set to the selection; children get True.
  - lff accumulators and the 100-step accumulators are reset.
  - Non-LFCF steps keep the lff accumulators and give new rows zeros and P = True
    (`_child_stats`, matching postfix_lff `islff=False`).
- Schedule: every 2nd densify, so LFCF at 600, 800, …, 14800 and GOF at 700, 900, …
- Hyper-parameters match `run_mipnerf360.py` (c_max 1.5, c_min 1, interval 2, diffscale True) and
  the train.py defaults.
- requires_grad pitfall: handled. `_append`/`_remove`/`reset_opacity` build the new Parameter from
  `p.detach()` with `requires_grad=p.requires_grad`, and `_check_trainable` runs after every
  population op (`density.py:173-207, 330-342`).

## Issues
1. **MAJOR — background white, official black** (`renderer.py:58`
   `render[0,...,:3] + (1 - alpha[0])`).
   - The official code uses black: arguments `_white_background=False`, and train.py:62 builds
     `bg_color=[0,0,0]`. Mip-NeRF 360 GT has no alpha, so the background only shows wherever
     accumulated opacity is below 1. That is exactly the low-opacity floater and transparency regime
     that LFCF targets.
   - Fix: add a renderer `background` config, default `[0,0,0]`, and pass
     `backgrounds=torch.cat([bg, 0])[None]` to `rasterization`. Precedent:
     `reproductions/bags/plugin/ss_bags/renderer.py:58-63`. Use it for both methods.
   - Then re-run the probes and smoke. Note in the report that the gsplat baseline uses white.
2. minor — the `renderer-absgrad` probe (`probes.py:269-294`) re-derives the statistic from
   gsplat's own `absgrad`. It is not the independent per-pixel autograd that plan.md claims.
   - It checks the W/2, H/2 scaling and the L1 combination, not gsplat's |·| accumulation.
   - Either fix the plan wording or add a per-pixel autograd reference on a 1-Gaussian case.
3. minor — plan.md fidelity gap 1 lists "means LR without 3DGS delay". The official
   `get_expon_lr_func` has `lr_delay_steps=0` (utils/general_utils.py:30), so delay_mult 0.01 is
   inert and this is not a gap. Remove it from the plan.
4. minor — `mip_filters.py:41-42` projects with K's cx, cy, while the official code uses W/2, H/2
   (gaussian_model.py:227). This only affects the in-frustum test of the 3D filter and the LFCF θ.
   The effect is negligible for the pilr COLMAP PINHOLE cameras. Keep it documented.
5. minor — the runtime densifies after the optimizer step. Official train.py:196 densifies before
   it, and the new Parameters have no grad, so that iteration's Adam update is skipped. This is
   documented and shared with the Mip plugin; no action needed.

## Is the `efa-gs-base` control needed? Yes, run it
- The claim on Mip-NeRF 360 is "no loss / slight gain" over the Mip row: garden −0.02,
  bonsai +0.22, average +0.02. On its own, absolute PSNR cannot be interpreted.
- Our runtime offset for a Mip base is −0.33 dB on garden (`runs/mip-splatting-i1-pilr`: 27.43 vs
  the Mip paper's 27.76), and the paper's own Mip row runs on a different (GOF) base. An efa-gs
  garden result of about 27.7 would therefore be equally consistent with "LFCF works" and with
  "LFCF hurts by 0.3".
- The smoke shows LFCF strongly changes the optimisation: at 2k steps, 669k Gaussians vs 1.21M for
  the base. The base is the only reference that separates LFCF from the runtime offset.
- Cost is 2 extra runs. The 0.22 dB bonsai gap is resolvable at single seed (±0.05). The garden
  gap is not, and the report should say so.
- This is an interpretive control, not an ablation, so it meets the "needed to interpret" rule.
- Also record #Gaussians and training time per run. Paper Tab. speed (bonsai) reports EFA faster
  than Mip (22'55" vs 25'05").

## Resolution / protocol
The official `-r 4/2` path uses PIL resize of full-res images (utils/camera_utils.py:23,41). That
matches the `mipnerf360-pilr` dataset, the baseline `baseline3dgs-mipnerf360-pilr-30000`, and llffhold8.

## Launch (after fix 1)
```
mkdir -p /bigdata/sejain/runs/efa-gs-pilr && ln -s /bigdata/sejain/runs/efa-gs-pilr runs/efa-gs-pilr
python3 tools/bench.py launch --exp efa-gs-pilr --methods efa-gs efa-gs-base \
  --dataset mipnerf360-pilr --scenes garden bonsai --steps 30000 --recipe 3dgs
```
