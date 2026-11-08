# Review 1 — ss_lpm (lpm, lpm-indoor) vs GS-LPM @7c06026

**Verdict: approved — 0 critical, 0 major, 4 minor.** This is a faithful port. I compared it line by line with `source/code/lpm/*.py` and `train.py`, and also ran a numerical check against a reimplementation of the official ray path.

## Numerical check (scratch script, GPU 6, smoke-1 2k checkpoint, bicycle)
- I rebuilt the official rays: the 3DGS projection matrix (znear 0.01, zfar 100), NDC z=1 unprojection, `world_view_transform.T.inverse()`, and the 6-D `[o, d]` rays. I also copied `gather_rays`, the pad-with-ray[0] step and `get_rays_intersection`. Then I fed real SuperPoint+LightGlue matches and real error maps from a 2k render to both versions, over 6 view pairs (56 zones).
  The plugin zones equal the official ones within max |Δ| = 1.7e-3 world units, which is just the principal-point offset.
- **Large zones are what the official code does, not a units bug.** Official zone max-extent: median 2.5, p90 58, max 215 world units (scene_scale ≈ 5).
  The cause is `region2zone_3d`. It pairs the *i-th perimeter ray* of r0 with the *i-th perimeter ray* of r1. When the two boxes have different widths and heights, the top/bottom/left/right segments drift out of alignment, so the paired rays do not correspond. Rays with a small baseline that are nearly parallel then give closest points far from the scene.
  Camera convention (OpenCV w2c, y down), pixel centres (pix2ndc = (2v+1)/S−1 ↔ +0.5), and the depth/NDC handling are all correct. Closest points do not depend on the length of the ray direction, so the official far-plane direction length (~100) does not matter.
- Neighbour sets: the official `find_angles` takes the angle between the 6-D vectors `[camera_centre, dir·100]`; the plugin uses the direction only. They disagree on 0.1% of view pairs (bicycle and room, 45° and 90°). Negligible.

## Component check (all match official)
- **C1 error map**: matches. Normalisation `/(mean+0.01)`, L1 summed over channels, 0.4 quantile, 16×16 patches, >85% fill (core.py:15-29 ↔ utils.py:get_errormap).
- **C2 regions**: matches. cv2 8-connected bounding boxes, inclusive keypoint test, int bounding boxes of the matches (core.py:33-69 ↔ region_matching.py). SuperPoint(max_kp=None) + LightGlue(depth 0.9, width 0.95) (matching.py:274-277 ↔ lpm.py:initialize_*). 3 random neighbours, or the view itself when it has none (density.py:116-129 ↔ get_paired_views).
- **C4 addition**: matches.
  - Clone = (g≥τ & small) | (zone & small & g≥τ/2); split is the same with large Gaussians.
  - n_extra = in-zone points with τ/2 ≤ g < τ. Prune the lowest-logit points (`≤` the n_extra-th value, ties included) plus σ<0.005, plus world scale >0.1·extent after step 3000 (core.py:143-167 ↔ lpm_densify_and_clone/split/prune).
  - **No zone → no densification or prune, and statistics keep accumulating** (density.py:178-186). This matches the official code: `points_addition` only calls `lpm_densify_and_prune` when `len(zones3d) > 0`, and `densification_postfix` is the only place statistics are reset.
- **C5 localized reset**: matches.
  - Inside = sphere at the zone centre with radius = min half-extent. The threshold is the k-th largest logit among inside points, k = int(0.7·n).
  - The reset hits points in the camera→zone cone within height−radius, sets them to logit(0.01) in place, and leaves Adam moments untouched (core.py:171-207, density.py:227-240 ↔ reset_localized_points / get_points_in_single_cone).
  - Schedule: 500 < s < 15000, every reset_interval (100 or 200).
- **Global 3DGS reset is kept** in the official code (train.py:133, every 3000 while < densify_until; it also fires at step 500 only for white-background datasets). The plugin keeps it (density.py:187). The official `reset_opacity` zeroes the Adam moments, and so does gsplat `reset_opa`.
- **Order** within a step is the same: addition → global reset → localized reset. Both use the same sampled pair and the same render. The official code recomputes the zones, but the result is deterministic and identical.
- **requires_grad / optimizer state**:
  - All population changes go through `gsplat.strategy.ops`.
  - The localized reset writes `.data`, which keeps the Parameter as a leaf.
  - `check_trainable` runs after every change.
  - New or split Gaussians get zero moments, as in `cat_tensors_to_optimizer`.
  - No pitfall found.

## Lazy pairing (fidelity gap 1)
Unlikely to change results.
- The official code also picks 3 random neighbours once. The plugin draws the same distribution, restricted to views already seen, and tops up later. With random-with-replacement sampling, about 85–95% of views have been seen by step 500.
- The only behavioural difference: an LPM step on a view whose neighbours have not been seen yet does nothing. The official code would densify there. This is rare and only happens early.
- Images are cached as uint8, which is lossless for 8-bit JPG/PNG sources.

## Issues
1. **minor**, core.py:158-167 (prune_mask). The official code also prunes `max_radii2D > 20` px when step > 3000 (lpm.py `lpm_densify_and_prune`, `if max_screen_size`). In the official code, max_radii2D only resets when zones are found. The shared gsplat baseline also omits this prune, so the plugin is consistent with the baseline. Record it as a fidelity gap, or add it behind a flag.
2. **minor**, core.py:211-227. Neighbour angle uses direction only; the official code uses the 6-D `[o, d]` vector. Measured disagreement is 0.1%. Document it.
3. **minor**, renderer/runtime. The error map is computed on a white-background render; the official code uses black for Mip-NeRF 360. This is a runtime gap with no effect on 360 scenes, which have no empty background.
4. **minor**, matching.py. LightGlue @5a650c4, whereas the official code uses its vendored `LightGlue/` copy. The match sets may differ slightly. Record the version used.

## Risk to watch (not a port bug)
- Because official zones are often tens of units wide, the localized reset every 100 steps can hit many Gaussians.
- The 2k smoke right after a reset (19.2 vs 21.1 dB) and at step 1950 (20.97 vs 21.34) both reflect this.
- The paper's gain shows up at 30k after resets stop at 15k, so judge at 30k. If the gap persists, log `lpm_reset` / N per call before suspecting the port.

## Probes
- Error map, zone projection (against least squares) and densify/reset probes all detect removal of their mechanism.
- No probe covers neighbour selection or the "no zone → no densify" branch. The density-cycle probe partly covers the second.
