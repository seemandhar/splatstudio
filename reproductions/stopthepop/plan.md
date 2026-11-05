# StopThePop — native SplatStudio plan

Method `stopthepop`, module `ss_stopthepop`; control `stopthepop-global` (same vendored code,
`configs/vanilla.json`: 3DGS global view-z sort) to separate the renderer from the runtime.

## Strategy
Wrap the author CUDA unmodified (csrc/ = StopThePop-Rasterization @ d8cad09, JIT via
`torch.utils.cpp_extension.load`, all template variants — no FASTBUILD). The paper trains *with*
the sorted renderer, and everything else is 3DGS, so the method = vendored renderer (training
and eval) + official Inria ADC density (as in reproductions/steepgs `strategy=adc`) + base
photometric loss (L1 + 0.2 D-SSIM).

## Components
| kind | factory | requires / provides | config (paper values) |
|---|---|---|---|
| renderer | `ss_stopthepop.renderer:create_renderer` | req `state.gaussian3d, camera.pinhole`; prov `render.rgb, gradient.screen.signed` | `splatting`: hierarchical.json (HIER, PTD_MAX, 64/8/4, 4x4 culling, rect + tight bounding, load balancing, tile_based_culling false); `background` [0,0,0] |
| density | `ss_stopthepop.density:create_density` | req `gradient.screen.signed` | 3DGS: from 500, until 15000, every 100, grad 2e-4, percent_dense 0.01, min_opacity 0.005, world prune 0.1·extent after 3000, reset 3000 → 0.01; `opacity_decay` 0 (0.9995 / 50 its option) |
| loss | base `photometric` | — | λ 0.2 |

No new FieldSpecs.

## Probes (all CUDA except schedule)
| probe | contribution | measures / fails when |
|---|---|---|
| `stopthepop-topt` | C1 | single anisotropic rotated Gaussian, `render_depth`: Σtαt/(1−T) per pixel = independent float64 Eq.(1) t_opt along the kernel's pixel ray (rel < 1e-3); must differ from view-z of the mean (fails if depth were the global key) |
| `stopthepop-perpixel-order` | C2 | crossing elongated splats where global z order ≠ per-pixel order: HIER and PPX_FULL images = independent float64 per-pixel t_opt-sorted compositor (max abs < 2e-3); GLOBAL differs (> 0.05) — fails if sort removed |
| `stopthepop-backward` | C4 | same fixture, random weighted loss: kernel grads of opacity logits, sh0, screen means = autograd of the float64 per-pixel-sorted reference (rel < 5e-3); global-order reference must be far |
| `stopthepop-popping` | C2/C5 proxy | yaw sweep 0→6° (0.25° steps) over two overlapping splats whose mean z-order flips; exact homography warp of F_i to F_{i+1}; L1 warp error per frame. HIER max/median small; GLOBAL max ≥ 5× HIER max (the pop) |
| `stopthepop-culling` | C3 | random scene: culling/bounding on vs off give identical images (< 1e-4) and fewer sort entries; reports HIER vs GLOBAL forward time |
| `stopthepop-schedule` | 3DGS density | densify at s>500, s%100==0, s<15000; reset at 3000; world prune only after 3000; no screen-size prune; opacity decay option multiplies by 0.9995 at s%50 |

## Files
`plugin/pyproject.toml`, `ss_stopthepop/{__init__,raster,renderer,density,reference,probes}.py`,
`csrc/` (vendored), `tests/test_stopthepop.py` (CPU: reference t_opt formula, settings builder,
density schedule), `README.md`.

## Benchmark
Per-scene numbers exist (supp.). Mip-NeRF 360 garden (outdoor, paper 27.16 vs its 3DGS 27.24)
+ counter (indoor, 28.59 vs 28.98), images_4/images_2 as full_eval.py, 30k, vs
`baseline3dgs-mipnerf360-30000`. Compare Δ(method − our gsplat) with paper Δ (−0.08, −0.39).

## Fidelity gaps
1. Runtime takes the optimizer step before density control (official: densify, then step) — shared with steepgs; negligible.
2. Runtime differences: sampling with replacement (official: shuffled stack), SSIM valid conv, LPIPS package, SH step timing.
3. Popping metric FLIP_t (RAFT flow + FLIP + 3 camera paths/scene) not reproduced at scene scale; replaced by an analytic warp-consistency probe with exact homography flow. Closable later (torchvision RAFT + FLIP port).
4. `tile_based_culling` false (released config) vs paper (A); speed-only.
5. Timings measured on A6000 not RTX 4090; not compared.
6. Opacity-decay rows not benchmarked (option implemented).
