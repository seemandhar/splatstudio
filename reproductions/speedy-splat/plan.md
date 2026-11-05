# Speedy-Splat — native SplatStudio plan

Method `speedy-splat`, module `ss_speedysplat`, package `splatstudio-method-speedy-splat`.
Extra methods: `speedy-splat-accutile` (paper row "+AccuTile": culling rasterizer, no pruning),
`speedy-splat-stock` (paper "Baseline": stock Inria rasterizer 59f5f77, same density, no
pruning — control for quality and for the unculled render path).

## Components
| kind | name | factory | requires / provides | config (paper value) |
|---|---|---|---|---|
| renderer | speedy-renderer | `ss_speedysplat.renderer:create_renderer` | req state.gaussian3d, camera.pinhole; prov render.rgb, gradient.screen.signed | rasterizer `speedy` (`stock` for the control), background [0,0,0] |
| density | speedy-density | `ss_speedysplat.density:create_density` | req gradient.screen.signed | Inria ADC (500/15000/100, 2e-4, 0.01, 0.005, big_world 0.1 after 3000, reset 3000→0.01) + soft_ratio 0.8, hard_ratio 0.3, prune_from 6000, prune_until 30000, prune_every 3000, prune true |
| loss | photometric | base | — | L1 + 0.2 D-SSIM (unchanged) |

No new FieldSpecs. The pruning score is a transient per-Gaussian buffer computed at prune steps.

## Contributions → code → probes
| C | code | probe (fails if removed) |
|---|---|---|
| C1+C2 SnugBox/AccuTile | vendored `csrc/speedy` (official, unmodified) | `speedy-culling-exact`: (a) image == independent float64 dense compositor that evaluates every Gaussian at every pixel (no tiles) to 1e-4; (b) `num_rendered` == independent exact ellipse∩tile count (quadratic minimised over each tile rectangle, Python) and < 3σ-square count of the stock rasterizer (also run) |
| C3 score | `backward.cu` dL_dG2, `raster.speedy_scores` | `speedy-score-kernel`: kernel score == Σ_p (σ·∂(ΣC)/∂α)² from float64 autograd of the dense compositor (rel 2e-3); multi-view density score == sum of per-view references |
| C4/C5 schedule | `density.py` | `speedy-prune-schedule` (CPU, stub scores): prune only at 6000/9000/12000 (0.8, after densify, before opacity reset) and 15000…27000 (0.3); percentile rule `score ≤ sort[int(r(N-1))]`; optimizer state aligned |
| ADC base | `density.py` | `speedy-adc-schedule` (CPU): densify only s>500, s%100==0, s<15000; no screen-size prune |

## Files
`plugin/pyproject.toml`, `ss_speedysplat/{__init__,raster,renderer,density,probes,fps}.py`,
`ss_speedysplat/csrc/speedy/` (rasterizer b38abb6 verbatim + LICENSE.md),
`ss_speedysplat/csrc/stock/` (graphdeco 59f5f77 verbatim, extracted with `git show`),
`ss_speedysplat/csrc/third_party/glm` (upstream glm headers, copied from the steepgs port),
`tests/test_speedysplat.py`, `README.md`.

## Implementation notes
- raster.py: Inria camera conventions copied from the steepgs port (FoV camera, znear 0.01,
  zfar 100, centred principal point check). Speedy Function returns (color, radii) and
  maps the 9th backward output (dL_dG2) to the `scores` input; `num_rendered` stored in info.
- Density hooks: `set_training_cameras` stores the cameras (score needs no images: L = ΣI).
  `after_backward` (runtime: after optimizer step). Order at s: ADC (s<15000) → soft prune
  (s∈{6000,9000,12000}) → opacity reset (s%3000==0, s<15000) → hard prune (s≥15000).
  Official order has the optimizer step between reset and hard prune; here the step already
  happened (same for both prune types: equivalent up to one Adam step on the pruned set).
- Score: under `torch.enable_grad()`, detached params, zero `scores` leaf per view,
  `render(...).sum().backward()`, accumulate `scores.grad` in float32 (as official).
- SH degree for the score render: the degree used at that step ((s-1)//1000 capped 3, the
  runtime's); official uses the active degree (s//1000) — one-step offset only at 6000/9000/12000.
- fps.py: CLI timing of forward rasterization (CUDA events) speedy vs stock on a trained
  checkpoint's test views; secondary, GPU-specific numbers.

## Fidelity gaps
1. Runtime optimizer step happens before density control; official densifies/soft-prunes before
   the step (and its densification discards that step's gradients). Small effect, not closable
   without runtime change.
2. View sampling with replacement (official: shuffled stack); SSIM valid conv; SH degree one
   step later; LPIPS from `lpips` package (official lpipsPyTorch, same VGG weights).
3. Resolution: mipnerf360-1600 protocol matches the official `-r -1`; our gsplat baseline
   `baseline3dgs-mipnerf360-1600-30000` is the comparison (paper baseline is Inria code).
4. Paper results are a mean of 3 runs; we run one seed.
5. Speed numbers: shared A6000, not the paper's A5000; only ratios vs our stock path reported.
