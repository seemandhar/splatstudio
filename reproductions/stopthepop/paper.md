# StopThePop: Sorted Gaussian Splatting for View-Consistent Real-time Rendering

Radl, Steiner, Parger, Weinrauch, Kerbl, Steinberger. ACM TOG 43(4) (SIGGRAPH 2024).
arXiv 2402.00525 (LaTeX v3, Oct 2024). Project https://r4dl.github.io/StopThePop/.

Sources: `source/latex/` (method.tex, results.tex, supplementary.tex);
`source/code/` = github.com/r4dl/StopThePop @ 859b11b (train/eval scripts);
`source/rasterizer/` = github.com/r4dl/StopThePop-Rasterization @ d8cad09 (the CUDA renderer,
submodule `submodules/diff-gaussian-rasterization`); `source/popping/` = r4dl/PoppingDetection.

## 1. Method summary
- 3DGS sorts Gaussians globally by the view-space z of their means; the blend order changes
  under camera *rotation* (popping). StopThePop blends each ray in the order of the per-ray
  depth of maximum contribution `t_opt` (Eq. 1).
- A full per-ray sort is >100x slower, so it uses a hierarchy: global sort by a per-tile depth
  (`t_opt` of the ray through the tile's max-contribution point), then in-kernel cooperative
  resorting through a 4x4-tile queue (64), a 2x2-tile queue (8) and a per-pixel head queue (4),
  culling non-contributing Gaussians at each level (effective sort window 24-72).
- Tile-based culling (exact max of the 2D Gaussian inside a tile vs 1/255), load balancing of
  per-tile work; only the blend order is changed, the EWA 2D splat approximation is kept.
- Backward pass in front-to-back order using C(r) - C_i and T_N / T_i, with the identical sort
  order as the forward pass. Models are **trained** with the sorted renderer (train.py passes
  the splatting config to every render) — not just rendered.
- Optimisation is otherwise unchanged 3DGS (same losses, ADC, schedules).

Contributions:
- **C1** per-ray depth `t_opt` along the view ray (§3.2, Eq. 1, App. A), S^-1 bounded at 1e3.
- **C2** hierarchical per-pixel resorting rasterizer (per-tile depth global sort + 4x4/2x2/pixel
  queues; §3.3–3.4) giving per-pixel sorted blending that removes popping.
- **C3** tile-based / hierarchical culling + load balancing (§3.4) — speed only.
- **C4** front-to-back backward pass with the same sort order (§3.5) — training through C2.
- **C5** FLIP-based popping metric `FLIP_t` over warped video frames (RAFT flow; §4.1).
- (option) Opacity Decay 0.9995 every 50 iterations instead of opacity reset (§4).

## 2. Equations
Eq. (1) (method.tex `eq:dopt`, derivation supp. App. A):
$$t_{opt}=\frac{\vec{d}^T\Sigma^{-1}(\vec{\mu}-\vec{o})}{\vec{d}^T\Sigma^{-1}\vec{d}},\qquad
\Sigma^{-1}=R\,\mathrm{diag}(s_x^{-2},s_y^{-2},s_z^{-2})R^T$$
ray $\vec r(t)=\vec o+t\vec d$, $\vec d$ normalised. Code (`stopthepop_common.cuh:13,45`):
`S^-1_ii = 1/max(1e-3, s_i)`, denominator clamped `max(1e-5, d^T Σ^-1 d)`; ray direction
`normalize(pix2world(pix) - campos)` with `pix_ndc = pix * 2/W - 1` (integer pixel coords).

Tile culling (method §3.4): $\hat x=\arg\max_{x\in X}G_2(x)=\arg\min_{x\in X}(x-\mu_2)^T\Sigma_2^{-1}(x-\mu_2)$; cull if $o\,G_2(\hat x)<\epsilon_O=1/255$. Per-tile depth = $t_{opt}$ of the ray through $\hat x$ (sort order PTD_MAX).

Backward (§3.5): $\sum_{j>i}c_j\alpha_j\prod_{k<j}(1-\alpha_k)=C(\vec r)-C_i$, $\prod_{k\ge i}(1-\alpha_k)=T_{N}/T_i$.

Blending (unchanged 3DGS): $\alpha=\min(0.99,\,o\,e^{-\frac12\Delta^T\Sigma_2^{-1}\Delta})$, skip if
power>0 or $\alpha<1/255$, stop when $T(1-\alpha)<10^{-4}$; 2D dilation +0.3; bounding extent
3.33σ (code; 3DGS uses 3σ), `tight_opacity_bounding` → $\min(3.33,\sqrt{2\ln(o/\epsilon_O)})$.

Popping metric (results.tex): $E_{i+t}=\mathrm{FLIP}(\hat F_{i+t},F_{i+t})$,
$\mathrm{FLIP}_t=\frac{1}{N-t}\sum_{i=0}^{N-t}E_{i+t}$, $\hat F$ = $F_i$ warped by RAFT flow,
occlusion masking (Ruder et al.), 20-px border dropped, per-pixel min subtracted; t∈{1,7}.

Sort error δ (Tab. 1): sum over consecutive out-of-order pairs of their `t_opt` difference.

Opacity decay: every 50 its after densify_from, $o \leftarrow 0.9995\,o$ (`train.py`), replaces reset.

## 3. Hyper-parameters
| name | value | source |
|---|---|---|
| sort_mode | 3 (HIER) | configs/hierarchical.json |
| sort_order | 3 (PTD_MAX: per-tile depth at max-contribution pos) | configs/hierarchical.json |
| queue sizes tile_4x4 / tile_2x2 / per_pixel | 64 / 8 / 4 | json; paper §3.4 |
| hierarchical_4x4_culling | true | json |
| rect_bounding | true | json |
| tight_opacity_bounding | true | json |
| tile_based_culling | **false** in released json (paper (A) describes it on) | json vs §3.4 |
| load_balancing | true (threshold 32 tiles) | json; §3.4 |
| proper_ewa_scaling | false | json |
| S^-1 bound | 1/max(1e-3, s) | §3.2; stopthepop_common.cuh:19 |
| iterations | 30000 | arguments/__init__.py |
| lrs (pos 1.6e-4→1.6e-6, feat 2.5e-3, opac 0.05, scale 5e-3, rot 1e-3) | 3DGS | arguments |
| densify 500–15000 every 100, grad 2e-4, percent_dense 0.01 | 3DGS | arguments |
| opacity reset every 3000 (<15000) to 0.01 | 3DGS | train.py |
| prune opacity < 0.005; world size > 0.1·extent after 3000 | 3DGS | train.py |
| screen-size prune (20 px) | inert (max_radii2D zeroed in densification_postfix) | gaussian_model.py:352 |
| λ_dssim | 0.2 | arguments |
| background | black | train.py (white_background False) |
| opacity_decay | 0 (paper's main rows); 0.9995 for "Opacity Decay" rows | full_eval.py |

## 4. Schedules
Identical to 3DGS: SH degree +1 every 1000; position LR exp decay over 30k; densify 500–15k;
opacity reset 3000·k < 15000. No StopThePop-specific schedule.

## 5. What it builds on
Inria 3DGS code base (train.py, gaussian_model.py unchanged except opacity decay), rasterizer
forked from diff-gaussian-rasterization. Everything except the blend order/culling is 3DGS.

## 6. Protocol
`full_eval.py`: 13 scenes; Mip-NeRF 360 outdoor `-i images_4`, indoor `-i images_2`; T&T and DB
at the provided resolution; `--eval` (llffhold 8); 30k iterations; black background; metrics.py
(SSIM 11x11 gaussian same-conv, LPIPS VGG, FLIP). The "3DGS" row is retrained by the authors
(per-scene values differ from the 3DGS paper, e.g. counter 28.98 vs 28.70). Per-scene numbers
for all 13 scenes are in supp. Tables (per_scene_tatdb_psnrssim, per_scene_360_psnr).
Timings: RTX 4090, Full HD, interpolated COLMAP path.

## 7. Ambiguities
- Released `hierarchical.json` has `tile_based_culling: false`, the paper's configuration (A)
  includes tile-based culling. Culling is exact (contribution < 1/255 is skipped anyway), so
  images are essentially unaffected; we follow the released config and expose the switch.
- Bounding extent 3.33σ instead of 3σ even in GLOBAL mode (code).
- Ray direction for `t_opt` uses `pix*2/W-1` (no +0.5) — half-pixel offset vs splat centres; code followed.
- Popping metric needs RAFT weights + FLIP + custom camera paths; not in the image-metric tables.
