# Pixel-GS (Zhang, Hu, Lao, He, Zhao; ECCV 2024) — implementation spec

arXiv 2403.15530. Official code: https://github.com/zhengzhang01/Pixel-GS (commit 05c9d55, cloned
read-only to `source/code/`; Inria Gaussian-Splatting licence, non-commercial research use).
LaTeX in `source/latex/`.

## 1. Method summary
- 3DGS with an unchanged training loop; only the densification statistic changes.
- C1 **Pixel-aware gradient** (Sec. 3.2, Eq. 7–8): each view's 2D (NDC) positional-gradient norm is
  weighted by m_k^i, the number of pixels the Gaussian takes part in under that view, and the
  accumulator is normalised by Σ m_k^i instead of the view count M^i.
- C2 **Scaled gradient field** (Sec. 3.3, Eq. 9–11): each view's NDC gradient is multiplied by
  f(i,k) = clip((z_cam / (γ_depth · radius))², 0, 1) before accumulation, suppressing growth of
  floaters close to the camera. Combined criterion Eq. 11.
- Everything else (clone/split/prune, thresholds, schedules, LRs, loss) is 3DGS default.

## 2. Equations (verbatim from sections/method.tex)
Eq. 4 (3DGS criterion):
  \frac{\sum_{k=1}^{M^i}\sqrt{(\partial L_k/\partial\mu^{i,k}_{ndc,x})^2+(\partial L_k/\partial\mu^{i,k}_{ndc,y})^2}}{M^i}>\tau_{pos}
Eq. 7 (pixel-aware):
  \frac{\sum_{k=1}^{M^i} m_k^i \times \sqrt{(\cdot)^2+(\cdot)^2}}{\sum_{k=1}^{M^i} m_k^i}>\tau_{pos}
Eq. 8 (pixel participation): ||pix − μ_p|| < R_k^i,  Π_{j≤i}(1−α_j) ≥ 1e−4,  α ≥ 1/255.
Eq. 9: radius = 1.1 · max_i ||C_i − mean_j C_j||  (3DGS getNerfppNorm, = `cameras_extent`)
Eq. 10: f(i,k) = clip( (μ^{i,k}_{c,z} / (γ_depth × radius))^2, 0, 1 )
Eq. 11: \frac{\sum_k m_k^i f(i,k) \sqrt{\cdot}}{\sum_k m_k^i} > \tau_{pos}
Variables: M^i = views with radius>0 since the last densification (100 iters), m_k^i = pixel count,
μ_{c,z} = camera-space z.

Official code (authoritative numerics):
- `submodules/diff-gaussian-rasterization/cuda_rasterizer/forward.cu:364` — `atomicAdd(&pixels[id],1)`
  for every (pixel, Gaussian) pair that is actually blended (after the alpha≥1/255 skip and the
  `test_T < 1e-4` early stop, which stops *before* blending; alpha clamp 0.99).
- `diff_gaussian_rasterization/__init__.py:144` — backward: `scaling_factor = min(1,
  (depth / depth_threshold)**2)` multiplies **only grad_means2D** (the screen-space dummy whose grad
  feeds densification); means3D gradients are unchanged. `depth` = view-space z (`backward.cu:168`).
- `train.py:91` — `depth_threshold = opt.depth_threshold * scene.cameras_extent`, `depth_threshold=0.37`.
- `scene/gaussian_model.py:405-407` — `accum[vis] += ||grad[vis,:2]|| * pixels[vis]`,
  `denom[vis] += pixels[vis]`, `vis = radii > 0`; `grads = accum/denom`, NaN→0; `>= threshold`.

## 3. Hyper-parameters
| name | value | source |
|---|---|---|
| γ_depth (`depth_threshold`) | 0.37 | paper §4.1; code arguments/__init__.py:99 |
| τ_pos | 2e-4 | paper §4.1; code arguments:98 |
| densify from / until / interval | 500 / 15000 / 100 | code arguments:95-97 (3DGS) |
| opacity reset | every 3000 | code arguments:94 |
| percent_dense | 0.01 | code arguments:92 |
| min opacity prune | 0.005 | train.py:111 |
| screen-size prune | 20 px after 3000 — no-op (max_radii2D zeroed in densification_postfix, Inria quirk) | gaussian_model.py:347,398 |
| world-size prune | scale > 0.1·extent after 3000 | gaussian_model.py:399 |
| LRs, λ_dssim 0.2, 30k iters, SH every 1000 | 3DGS defaults | arguments |
| pixel alpha clamp / stop | 0.99 / T<1e-4 exclusive | forward.cu |

## 4. Schedules
Statistics accumulated every step < 15000; densify at steps 600..14900 (iteration > 500, %100);
opacity reset at multiples of 3000 < 15000. No new schedules.

## 5. Builds on
Inria 3DGS (official code is a fork). Loss, optimiser, rendering, SH schedule unchanged.

## 6. Protocol
Mip-NeRF 360 all 9 scenes, `-r 4` outdoor / `-r 2` indoor (script/mipnerf360.py) = PIL resize of
full-res images → our `mipnerf360-pilr`. llffhold8 (`--eval`), 30k iters, black background,
3DGS metrics.py (LPIPS-VGG). Tanks&Temples: all 21 scenes with the authors' own processing (not our
`tandt` split) — not comparable. Per-scene numbers for all 9 Mip-NeRF 360 scenes in supp. Table 7
(`360scene_compare.tex`), with retrained 3DGS* row.

## 7. Ambiguities
- Pixel-participation rule: paper Eq. 8 states T ≥ 1e-4; code stops before blending when the
  next T < 1e-4 (exclusive) and uses the Inria alpha clamp 0.99. Our renderer is gsplat
  (alpha clamp 0.999, exclusive stop next_T ≤ 1e-4); we count with gsplat numerics so counts
  describe the image actually rendered.
- Eq. 8's radius test: Inria has no per-pixel radius test, only tile membership (bbox of 3σ);
  gsplat uses per-axis 3σ boxes — counts can differ by a few edge pixels with α≥1/255.
- Denominator of Eq. 11 is Σ m (without f) — matches code (only the gradient is scaled).
