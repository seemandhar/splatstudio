# Review 1 — nerf-init-gs (arXiv 2404.12547)

**Approved: yes.** 0 critical, 0 major, 5 minor. No implementation bug found that would explain the
2k smoke gap. bench_request.json needs two small fixes before launch (minor 1, 2).

## Smoke-gap investigation (garden, 2k steps, test split)

Diagnostics re-rendered the three 2k checkpoints (scripts in the reviewer scratchpad, GPU 6):

| run | PSNR | PSNR after mean-colour match | mean bias (R,G,B) | top-¼-image PSNR | N | scale q99 / max | α mean |
|---|---|---|---|---|---|---|---|
| gsplat baseline (SfM, `--recipe 3dgs`) | 23.02 | 23.04 | ≈0 | 21.34 | 845k | 0.21 / 37 | 0.976 |
| nerf-init-gs | 22.37 | 22.47 | +0.003 | 20.31 | 1.33M | 0.75 / 135 | 0.9997 |
| nerf-init-gs-nodepth | 21.10 | 21.57 | +0.013/+0.012/+0.008 | 19.33 | 925k | 0.45 / 245 | 0.994 |

- **The colour/brightness hypothesis is mostly ruled out.** Matching the mean colour recovers only
  0.46 dB (nodepth) and 0.09 dB (depth), so most of the deficit is something else. The SH DC path is
  correct: `state.py:31` gives `sh0 = (rgb-0.5)/C0`, the NeRF colours are sigmoid radiance in linear
  [0,1] clamped at `preprocess.py:103`, and the mean init colour is 0.466/0.446/0.299 against an
  image mean of 0.440/0.429/0.286. The NeRF itself has no colour bias: its test-view bias is ≤0.004.
- **No far-plane or opaque-background points.** `far_fraction_t_gt_half_far = 0.0` in
  nerfinit.json. The 95th-percentile distance from the centroid is 14 world units, and the 3-NN
  log-scale p99 is 0.34 (≈1.4 units).
- **Units and alignment between the NeRF and the 3DGS world are correct.** I projected the SfM
  points into training views 0/40/80/120 and compared their camera-z with `depth_z_f16`. The
  median ratio is 1.03–1.09, and the ratio for the NeRF points is 1.02–1.04. A ratio slightly above
  1 is expected because occluded points are included. The `names.json` order equals the train
  split. The ray-distance to camera-z conversion `t/scale/|K⁻¹p|` (`preprocess.py:126`) is correct,
  because the nerf-frame rotation is orthonormal.
- **Where the deficit is:** the top quarter of the image (background trees and sky) is 1–2 dB
  worse, and the checkpoints have more large Gaussians.
  - The 5k iNGP reaches only **22.1 / 24.0 / 24.8 dB on test views** DSC07956/08020/08084 (ray PSNR
    24.1 on training rays).
  - 500k noisy samples at opacity 0.1 converge more slowly in the background than SfM points do.
  - The paper reports only final 30k numbers. Its NerfAcc@5k result is −0.08 dB against COLMAP on
    garden.
- **Verdict:** a 2k smoke cannot decide this, and nothing is broken. It is a real risk that the
  30k number lands below baseline because our fp32 iNGP is weaker than tcnn's. The paper says 5k
  iterations take about 30 s on an A6000; ours took 171 s (fidelity gap 2).

## Checks passed
- **Eq. 4.** One inverse-CDF sample per uniformly drawn training ray, in the proposal sampler's
  lindisp s-space (`ingp.py:436-460`). The colour is queried at the sample in the ray direction.
- **Eq. 6.** Σ wₖ tₖ with an opaque last interval (`ingp.py:478-490`). The probe checks it.
- **Eq. 5.** `depth_accum = ED·α = Σ wᵢ zᵢ` is not normalised (`renderer.py:41`). Using the
  camera-z of each centre instead of the ray-conditional mean (Supp.) is a declared gap.
- **Eq. 7 / 13.** `λ = 0.9·0.9^(step/100)` on the 1-based step (`losses.py:16`), times the mean L1
  over all pixels (`losses.py:31`). The depth term is not masked. That is valid here: every
  NeRF depth is finite (opaque far plane), and the paper does not mask either.
- **Init loading.** Exact 3-NN scales are SHA-checked (`density.py:32-40`), opacity is 0.1, and
  quats are identity. The camera extent of 4.92 replaces the splatstudio scale of 11.39 for both
  `state["scene_scale"]` and the means LR. `set_training_cameras` runs before the runtime captures
  `initial_lr` (`runtime.py:149-165`), so the decay uses the corrected LR. This is equivalent to
  `--recipe 3dgs`.
- **requires_grad.** Only in-place `copy_`/`fill_` happen under no_grad, and densification goes
  through gsplat ops. There is no Parameter rebuild, so the pitfall does not apply.
- **Train/eval separation.** The sidecar is looked up only when means and rgb require grad
  (`renderer.py:42`), so evaluation renders never touch the NeRF depth.
- **Protocol.** The garden copy has protocol_id `53deee9d…`, identical to
  `baseline3dgs-mipnerf360-30000/gsplat/mipnerf360-garden`. The copy has the same dataset_sha256,
  and the test list is the llffhold8 list.
- **Opacity reset and densification.** The 3DGS opacity reset is inherited from the runtime
  `Density`, and the densification parameters are unchanged.

## Issues

1. **minor — bench_request.json: dataset key.** The key `mipnerf360-nerfinit` is not in
   `tools/datasets.json`, so `bench.py launch` will KeyError. Add the
   `proposed_datasets_json_entries` block, following the `mipnerf360-conegs` precedent.
2. **minor — bench_request.json: `extra` field.** `"extra": "--recipe splatstudio"` duplicates
   bench.py's own `--recipe` flag. bench.py:195 emits `--recipe 3dgs` first. argparse takes the
   last value so the run works, but the job record says 3dgs. Use `--recipe splatstudio` and set
   `extra` empty. `--num-gaussians 10000000 --max-gaussians 12000000` are already hard-coded in
   bench.py:196, so they are not needed in extra.
3. **minor — the unit of D_NeRF in Eq. 7 is unstated (paper Sec. 3.3).** nerfacc's normalised frame
   is 0.2485× world on garden. If the authors supervised in NeRF units, their effective λ was about
   4× smaller than ours. With world units, the depth term starts at 5.68 against a photometric
   term of about 0.2, and the run has more Gaussians at 2k (1.33M against 925k). Sec. 3.5 warns
   about excessive Gaussians. Record this in paper.md ambiguity 5. If the 30k result is low, a
   labelled "tuned" ablation in NeRF units is justified.
4. **minor — λ schedule shape (Sec. 3.5).** "Decay rate of 0.9 every 100 iterations" may mean a
   staircase, `0.9^floor(i/100)`. The continuous form (Supp. Eq. 13 as written) is fine. The
   difference is at most a 10% λ ripple.
5. **minor — NeRF quality and time.** The 5k iNGP reaches about 23.6 dB on test views and takes
   171 s, against the paper's ~30 s. This is plan gap 2. Report it, and add the 836 s of
   preprocessing to the training time. Room trains the NeRF at images_2, where nerfacc uses
   factor 4 (gap 3).

## Probes
The probes would catch removal of each mechanism:
- the surface sampler, via the 0.3/0.7 two-surface split;
- λ and the L1 gradient, via analytic checks;
- the scales, opacity, extent and LR, plus a SHA-mismatch check that must raise;
- ED·α against gsplat's "D" render mode, plus a check that eval renders do not touch the sidecar.

No probe checks the world-unit conversion of `depth_maps`. The SfM-projection check above covers
it empirically.

## Launch (after adding the datasets.json entry and the room copy finishing)
```
python3 tools/bench.py launch --exp nerf-init-gs-30000 --methods nerf-init-gs nerf-init-gs-nodepth \
  --dataset mipnerf360-nerfinit --scenes garden room --steps 30000 --recipe splatstudio
```
Compare the result to `baseline3dgs-mipnerf360-30000` for garden and room.
