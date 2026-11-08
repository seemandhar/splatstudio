# GaussianPOP — native SplatStudio plan

Method **`gaussianpop`** (post-training, P=0.8, C=8, 5k fine-tune; one 35k-step run).
Module `ss_gaussianpop`. Variants `gaussianpop-p07`, `gaussianpop-p09` (Tab. 8 other rows).

## Mode chosen: post-training (Supp. Tab. 8)
- Tab. 8 is complete per scene (13 scenes × P∈{0.7,0.8,0.9}); its base is a vanilla 30k 3DGS
  model, which our `gsplat` recipe reproduces (the `baseline3dgs-mipnerf360-30000` runs are the
  pre-prune reference).
- On-training mode needs GaussianSpa's config with Mini-Splatting densification (40k for the
  per-scene table) — a different base we do not have; rejected.
- The runtime cannot resume from a checkpoint, so (as in `reproductions/pup-3dgs`) one run does
  3DGS 1–30000 (identical recipe/LR to the 30k baseline), prune at the end of step 30000,
  fine-tune 30001–35000.

## Benchmark
mipnerf360 `bicycle` (outdoor) + `counter` (indoor), `--steps 35000`. Targets P=0.8:
25.12 / 29.00 (paper 3DGS row: 25.13 / 29.11; ours 24.92 / 28.85). Δ-vs-unpruned is the key
comparison (paper −0.01 / −0.11 dB). Post-prune-only (Tab. 3: bicycle 20.71 dB at 80%) is
available by running `--steps 30000` (prune at step 30000, no fine-tune).

## Components
| component | kind | factory | requires/provides | config |
|---|---|---|---|---|
| gsplat | renderer | `splatstudio.backends.gsplat:create_renderer` | unchanged | — |
| pop-density | density | `ss_gaussianpop.density:create_density` | requires `gradient.screen.signed` | `base` (baseline DefaultStrategy config), `base_steps` 30000, `prune_step` 30000, `prune_ratio` 0.8, `cycles` 8, `split` "compound", `n_max` 64, `eps` 1e-9, `background` [1,1,1], `lr_final_ratio` 0.01, `reset_optimizer` true |
| photometric | loss | unchanged | — | — |

No new fields. Training cameras via `set_training_cameras`.

## CUDA (C1+C2)
`csrc/pop_error.cu`, adapted from gsplat 1.5.3 `RasterizeToPixels3DGSFwd.cu` (Apache-2.0) via
`reproductions/augs/.../contrib_stats.cu`: same tile traversal, 1/255 threshold, 0.999 alpha
clamp, exclusive 1e-4 termination. Per pixel two traversals of the tile list: pass 1 computes
C_render = Σ T α c + T_final·bg; pass 2 recomputes P_k, T_k, T_{k+1}, and for the first n_max
contributors adds (T_k α_k)² ||c_k − (C − P_k)/(T_{k+1}+ε)||² to Δ_SE[g] (float atomics).
Mathematically identical to Alg. 1's cached arrays. Python: gsplat `fully_fused_projection`
(eps2d 0.3, near 0.01, same as `rasterization`), SH colours with gsplat `spherical_harmonics`
(+0.5, clamp_min 0, degree 3), `isect_tiles`/`isect_offset_encode`, JIT `load`.

## Pruning (C3)
At the end of step 30000: for c=1..C: Δ_SE over all training views; keep_c =
round(N0·(1−P)^{c/C}) (compound; `split: linear` gives round(N0(1−P·c/C))); remove the
N_cur−keep_c lowest (ties by stable sort) via `gsplat.strategy.ops.remove`. Then clear Adam
state. LR: steps ≤30000 means LR = lr0·0.01^(s/30000) (overrides the runtime's 35k decay);
fine-tune means LR = lr0·0.01 (3DGS clamp), others unchanged.

## Probes
| probe | contribution | fails when |
|---|---|---|
| `delta-se-oracle` (CUDA) | C1+C2 | kernel Δ_SE vs brute force: remove each Gaussian of a random fixture, re-render with gsplat `rasterization` (white bg), Σ_pixels ||C−C'||². Also kernel C_render == gsplat render. Fails for T_k/T_{k+1} swaps, missing background, wrong colours; also asserts the blending-weight score ranks differently |
| `prune-cycles` (CUDA) | C3 | density on a fixture: C quantification calls, per-cycle counts = compound schedule, removed set = lowest scores recomputed independently each cycle, final count = round((1−P)N0), Adam cleared, LR pre/post |

Unit tests (CPU): keep-count schedule (compound/linear), LR functions, lowest-k mask.

## Files
`plugin/pyproject.toml`, `ss_gaussianpop/{__init__,density,kernels,schedule,probes}.py`,
`ss_gaussianpop/csrc/{pop_error.cu,ext.cpp}`, `tests/test_gaussianpop.py`, `README.md`.

## Fidelity gaps
1. Base model is our gsplat 3DGS (white bg, random views, valid SSIM), 0.2–0.26 dB below the
   paper's 3DGS row on bicycle/counter. Compare Δ vs unpruned. Not closable.
2. Per-cycle split (compound vs linear) and N_max semantics unstated; chosen as above.
3. Fine-tune LR/optimizer unstated: chosen 3DGS continuation with fresh Adam.
4. Background in C_render: white (our training bg); paper likely black.
5. Pre-prune model not evaluated in the same run; baseline run is the reference.
6. On-training mode not reproduced (needs Mini-Splatting/GaussianSpa base).
