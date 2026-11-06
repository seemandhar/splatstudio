# Hierarchical 3DGS (plan id 74) — built, ready for benchmark

**Feasibility.** The main tables (Tab. 1/3/5) use one *unnamed* chunk of the authors' own
SmallCity / Campus / BigCity / Wayve captures. Only SmallCity is released (3.0 GB, plus 2.5 GB
pretrained), and getting those numbers needs the full large-scene pipeline: COLMAP chunk bundle
adjustment, scaffold and skybox, DPT/Depth-Anything depth, masks, exposure, and 4 chunks plus
consolidation. That is out of scope. Mill-19 (Tab. 2) is also out of scope (aerial data,
Mega-NeRF protocol).

**Target.** §6.2 reports the **Mip-NeRF 360 "average over the entire dataset"** for the
single-chunk optimization:

| | PSNR (dB) |
|---|---|
| 3DGS (their run) | 29.11 |
| leaves | 28.96 |
| τ = 3 px, basic / opt | 28.86 / 28.87 |
| τ = 6 px, basic / opt | 28.05 / 28.52 |
| τ = 15 px, basic / opt | 24.82 / 27.25 |

There are no per-scene numbers and no SSIM/LPIPS. Their 3DGS figure (29.11) is far from the
3DGS paper's 27.21 and cannot be a 9-scene llffhold8 mean at any resolution we have (our gsplat
9-scene ≈ 27.2-27.3). It fits the 7 original public scenes (no flowers/treehill; our 7-scene
28.81 at images_4/2, 3DGS paper 28.69) plus the official per-channel PSNR
(render_hierarchy.py:111 → image_utils.psnr on [3,H,W], ≥ full-image PSNR). We therefore run all
9 scenes and report 9- and 7-scene scene means, each with full-image and per-channel PSNR
(`summary.py`). The primary comparisons are the τ-row deltas vs leaves (independent of the scene
set and the 3DGS run) and leaves − 3DGS on the 7-scene mean (paper −0.15).

**Protocol.** `mipnerf360-1600`: the code default (`-r -1`, `images` folder, width capped at
1600 px). The README eval commands pass no `-r/-i`, and τ is in pixels at the render resolution
(render_hierarchy.py:57). Baselines: `baseline3dgs-mipnerf360-1600-30000` [garden, bicycle,
counter] + `-1600b-30000` [other 6]. The config follows the official single-chunk eval command
(README.md:550): `--skip_scale_big_gauss` (big-Gaussian shrink **off**; still available with
`skip_scale_big_gauss: False`) and `--exposure_lr_init 0.0`. The scene extent is
1.1 × the 0.9-quantile of training-camera distances (dataset_readers.py:58), computed in the
density from the training cameras. The runtime's 1.1 × max is 5-10 % larger. The extent scales
the position lr and the clone/split boundary.

**Build.** Native plugin `hierarchical-3dgs` (plugin/), which follows the official
`full_train.py`: train_single (30k) → GaussianHierarchyCreator → train_post (15k) →
render_hierarchy.
- The hierarchy rasterizer and the gaussian-hierarchy C++/CUDA are vendored **unmodified**
  (Inria non-commercial licence), with glm and Eigen 3.4. All three are JIT-built through
  tools/ssenv.
- The creator's `main()` is wrapped as a function (csrc/creator_ext.cpp).
- The tree is packed into a per-Gaussian field `h3d` so that ss-eval can render cuts.
- `eval_taus` evaluates τ ∈ {leaves, 0, 3, 6, 15}. τ = 0 is the official finest row, a 1 px
  threshold; "leaves" is threshold 0. Every view also records per-channel PSNR. Its `--basic`
  option rebuilds the unoptimized hierarchy from the frozen leaves, after checking that every
  leaf is an anchor. One 45k run per scene therefore gives every row. The rebuilt hierarchy comes
  from half-precision leaves and is not bit-identical to the 30k hierarchy (smoke: 251140 vs
  251189 Gaussians). On the smoke run, the leaves PSNR is identical for both.
  `eval_taus --channel-only` re-renders any run, such as the baseline, for per-channel PSNR.

**Evidence.**
- ss-verify: `declared_checks_passed`. 5 probes cover the Eq. merge vs leaf descendants, cut
  validity and t/K, α′ (K children at t=0 = parent), the post-optimization freezes, and the
  train_single rules. The leaf probe also checks that the shrink is off by default and on when
  enabled, that after_backward overrides the runtime's means-lr decay, that the post phase
  restarts the schedule, and that extent = 1.1 × q0.9.
- 5 unit tests pass; ruff is clean.
- The smoke runs below predate the review fixes: shrink on, max extent, images_4.
- 2k-step smoke on garden: 22.00 dB vs gsplat 22.98 dB. The gap is expected from the official
  0.01 initial opacity, the 8× lower position lr, and the conservative densification (0.40M vs
  0.84M Gaussians).
- Quick pipeline (1.5k + build + 0.5k): at τ = 15 the optimized hierarchy gives 20.59 dB vs
  20.37 dB basic, so post-optimization already helps at coarse cuts.

**Fidelity gaps.**
- **No depth regularization, although the official single-chunk eval command uses it**
  (`-d depths`, README.md:550; the inverse-depth L1 loss has weight 1 → 0.01 exponential,
  train_single.py:111-117). The paper does not say whether the Mip-NeRF 360 runs passed `-d`.
  Their ablation shows depth costs −0.12 dB on average, so the expected effect is small.
  Adding it is feasible but not done:
  - the vendored rasterizer already returns inverse depth with a backward;
  - Depth-Anything-V2 (the official `generate_depth.py` default) can be downloaded;
  - the COLMAP `sparse/0` with tracks needed by `make_depth_scale.py` (per-image scale/offset)
    is on disk;
  - the depth maps could be looked up by viewmat, as the DNGaussian plugin does.
  It would cost a preprocessing script, a loss component, a few GB of depth maps on /bigdata
  and about 1-2 GPU-h. This is recorded as a gap.
- Exposure optimization is off (the paper disables it for single-chunk comparisons).
- No scaffold/skybox/chunking/consolidation/streaming.
- Runtime view sampling is with replacement; densification runs after the optimizer step
  rather than before; the SH degree steps up one iteration later.
- The appearance filter uses train cameras only.
- Black eval background.
- Leaf-phase ordering: the official loop densifies/resets before the optimizer step (that
  step is skipped) and does not step the final iteration. We step every iteration
  (negligible).

**Disk.** Checkpoints are kept, under /bigdata/sejain/runs/<exp> with a symlink in runs/. Each
holds params only after step 45000 (Adam state cleared): 312 B per hierarchy Gaussian, with
about 1.3× as many hierarchy Gaussians as leaves. At 1600 px we expect 1.2-2.5 GB per scene and
10-20 GB for 9 scenes.

Benchmark: see bench_request.json — mipnerf360-1600, 9 scenes × 45k steps, about 30-40 GPU-h.
