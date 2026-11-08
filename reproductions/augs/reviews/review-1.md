# AuGS review 1 (splat-reviewer)

**Verdict: approved.** 0 critical, 0 major, 5 minor. Fix minor #1 before launching. Changing plugin
source after ss-train starts makes ss-eval refuse the run (`source_manifest` check).

Compared against the official code at `source/code` @ 57cca00: refine.py, train_2d.py, benchmark.py,
gaussian_renderer/__init__.py, gaussian_2d_renderer/{__init__,projector}.py,
diff-gaussian-2d-rasterization/cuda_rasterizer/{forward,adam}.cu,
submodules/gsplat/{rendering.py, rasterize_to_pixels_fwd.cu}, and wpca_fork.
Checks I ran: pytest (11 passed), a cdist precision test on GPU 7, and counts of the pilr train
splits (bicycle 169, room 272).

## Issues

| # | sev | where | problem / evidence | fix |
|---|---|---|---|---|
| 1 | minor | inverse.py:110 `adjacency`, :75 `sparse_edges` | SLINK edges come from float32 `torch.cdist` on raw world coordinates. For more than 25 points cdist uses the matrix-multiply form, which loses precision through cancellation. Bicycle coordinates have \|p\| of about 3-5 and the threshold is about 5·d/fx (≈0.01). Against float64 (official: sklearn on float64 points), 556-1126 of ~28k edges differ at depth 2, and 5026 differ at \|p\|=20. Centring the points first brings this to 0. Fidelity gap #7 ("float32") understates this. Edges near the threshold decide whether clusters split or merge. | Subtract the pivot (or mean) of each kernel's points before cdist, or pass `compute_mode="donot_use_mm_for_euclid_dist"`, or compute in float64. |
| 2 | minor | plan.md "Variant", bench_request.json notes | "Implied base matches our pilr baseline" holds for room only: 31.67 implied vs 31.69 ours. Bicycle is 25.74 vs 25.45 (−0.29 dB), and SSIM is 0.790 vs 0.772. | Report the gain over the in-run `base_test.json` (official Δ: +0.11 / +0.72), not absolute PSNR. Label bicycle's absolute target as base-limited. |
| 3 | minor | optim.py:515 | `is_used` is approximated as `grad != 0`. gsplat 1.5.3 drops opacity gradients where alpha > 0.999 (TODO-runtime). The official SparseGaussianAdam (adam.cu) still applies a momentum step to used rows whose gradient is zero. Already gap #6, so this is documentation only. | None needed. Optionally take `is_used` from the contrib_stats counts. |
| 4 | minor | density.py:386 | The 2D stage fits kernels over the white-background render; official uses black (train_2d.py:320). This changes the per-view budget and the loss map where alpha < 1. It follows the runtime-wide gap #1. | Keep it listed in the report. |
| 5 | minor | bench_request.json `extra` | Repeats `--recipe/--num-gaussians/--max-gaussians`, which bench.py `job_command` already passes (harmless). bench.py takes one `--steps` per experiment, so each scene needs its own experiment. | Drop `extra`; launch as below. |

## Equation / mechanism coverage (all match official unless noted)
- **Eq. 1 lobe** (renderer.py:100-123 vs gaussian_renderer:19-39): matches exactly. Covers acos of clip ±0.9999, T clamp(max=π), the 1e-5 pow guard, a normalised d, and the view direction from the learnable aug means. Gradients reach opacity, d, T, β and means.
- **MD depth** (renderer.py:168-212 + contrib_stats.cu vs official gsplat fwd.cu:173 and rendering.py:583-627): matches. The `cur==-1 || T>=0.7` test is on T before blending, with exclusive 1e-4 stop, alpha clamp 0.999, max-response tau in double, FAR 1e10 / NEAR 0.01, and eps2d 0.3 classic.
- **2D stage** (fit2d.py vs train_2d.py / Gaussian2DModel / forward.cu): matches.
  - Budget is ceil(l_i/Σ·0.1N). Sampling is loss_map² without replacement.
  - Adds are 20 % per round, with 200 coarse and 1000 finetune steps.
  - Prune is σ(o) ≤ .005 or pix_covered ≤ 25, using counts from the last forward pass. pix_covered uses Inria semantics (0.99, inclusive).
  - AA uses cov + 0.3 with sqrt(det ratio) compensation.
  - Pixel-centre mapping (+0.5): exact.
  - lrs: xy 0.001·(H+W)/2, colour .01, opacity .02, scale .001, rotation .02.
  - Init: opacity 0.02, scale 0.005·min(W,H), colour clamp [1/255, 244/255].
  - Scale clamp: log min(50, 0.01(H+W)).
  - Bilinear zero-padded depth resample after every step. Adam with bias correction, keeping the step count and moments across add/prune.
  - Loss: L1 + 0.2(1 − fused-ssim, same padding).
- **Inverse splatting** (inverse.py vs projector.py): matches, except the SLINK distance precision in #1.
  - Rect, weights (cov + 0.3, raw opacity, ≤ 0.99) and validity (≥ 1/255, depth < 100).
  - Pivot taken before and after the validity filter. Threshold 5·max(0.2, d)/fx, ×1.5 retry; largest-cluster fallback.
  - New depth from pivot or weighted mean. WPCA: w-mean, (wX)ᵀ(wX)/wᵀw, descending, var ≥ 1e-7.
  - Normal flipped against the initial posWorld; det fix applied.
  - Depth < 0.2 → 100.
  - Eq. 6: k = ⟨Ĉ,q⟩/⟨q,q⟩ with raw Ĉ and J clamped at 1.3·tan(fov/2).
  - θ̄ from the closest other visible camera; T = clip(7θ̄/π, .01, 1); β = 0; d points to the camera; SH0 = RGB2SH(colour); shN = 0.
- **Joint phase** (density.py:464-472, optim.py vs refine.py / benchmark.py:141-164): matches.
  - Aug rows of every base field always step with no bias correction. Original rows update opacity only, and only where used. Lobe params use torch Adam with eps 1e-15.
  - lrs: xyz 1.6e-5·extent (constant), f_dc 1e-3, f_rest 5e-5, opacity .02, scale 2e-3, rotation 5e-4, d 1e-3, T 2e-4, β 2e-3.
  - No densification; sh = 3.
  - Joint steps = 30·|train|, matching the official `range(30000, iters+1)` with step only when `iteration < iters`.
- **requires_grad**: OK. density.py:462 builds `nn.Parameter(cat(old.detach(), extra), requires_grad=True)`, and the guard at :473 checks every field is a trainable leaf. All optimizers are rebuilt. The runtime re-reads the `optimizers` dict every step, and check_state passes after augmentation (smoke).
- **Base phase**: the same as the built-in `gsplat` method.
  - Same rasterization call, and an aug_flag of 0 skips the lobe.
  - Same Density config (absgrad off, 2e-4, reset on).
  - The means LR schedule is re-based on 30000 so it matches a 30k gsplat run.
  - The extra fields get no gradients and do not consume RNG.

## Variant / steps / evaluation
- **Variant**: the "Ours 3DGS sh=3" row is correct. per_scene_m360 gives bicycle 25.85 and room 32.39; diff_ps_mipnerf gives Δ +0.110 and +0.720.
- **Protocol**: `--resolution 4/2` corresponds to our pilr protocol.
- **Steps**: correct. pilr train counts are 169 and 272, matching official `scene_train_cameras`, so bicycle needs 35070 and room 38160. The density raises on any other `--steps` (density.py:335-342; ss-train only).
- **Evaluation**: `ss-eval` (runtime.evaluate) rebuilds AugRenderer from the method and renders `checkpoint["params"]`, which include lobe_dir/t/beta and aug_flag. sh_degree is 3. So the test metrics come from the final post-augmentation model. The smoke run confirms this: test 21.47 vs the in-run base 20.91.

## Launch (after fixing #1, then re-run pytest and verify)
```
mkdir -p /bigdata/sejain/runs/augs-bicycle /bigdata/sejain/runs/augs-room
ln -s /bigdata/sejain/runs/augs-bicycle runs/augs-bicycle; ln -s /bigdata/sejain/runs/augs-room runs/augs-room
python3 tools/bench.py launch --exp augs-bicycle --methods augs --dataset mipnerf360-pilr --scenes bicycle --steps 35070 --recipe 3dgs --min-free-gb 24
python3 tools/bench.py launch --exp augs-room    --methods augs --dataset mipnerf360-pilr --scenes room    --steps 38160 --recipe 3dgs --min-free-gb 24
```
Compare test PSNR against `baseline3dgs-mipnerf360-pilr-30000`, and the gain against `<run>/base_test.json`.
