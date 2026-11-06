# Faster-GS — native SplatStudio plan

Method `faster-gs`, module `ss_faster_gs`, package `splatstudio-method-faster-gs`.
Route: the authors' backend-integration route — vendor `FasterGSCudaBackend` (main @ 090e8ad,
Apache-2.0) `rasterization/` + `adam/` + `utils/` into `ss_faster_gs/csrc/`, JIT-built with
`torch.utils.cpp_extension.load` (official flags `-O3 -use_fast_math -std=c++17`). Verified: builds
with the env's CUDA 12.1 / gcc 11 for sm_86 in ~75 s.

Pinned configuration = paper headline row "Ours" / Tab. 2 "Full" = `main` defaults:
no MCMC, no Mip-Splatting filters, no Speedy-Splat pruning, no AbsGS, no PPISP, no random bg,
SfM init, black background, opacity lr 0.025, Morton every 5000 ≤ 15000, separate (non-fused)
backward and optimizer step with the own CUDA Adam. The fused backward+optimizer (`FasterGSFused`
branch, Tab. 3 optional row) is NOT used. Borrowed and baked into the kernel: warp-level sub-tile
culling (HPG 2026) — labelled, lossless, speed only.

## Components
| kind | name | factory | requires / provides | config (paper value) |
|---|---|---|---|---|
| renderer | fastergs-renderer | `ss_faster_gs.renderer:create_renderer` | req state.gaussian3d, camera.pinhole; prov render.rgb, densification.fastergs | background [0,0,0], near 0.2, far 10000 |
| density | fastergs-density | `ss_faster_gs.density:create_density` | req densification.fastergs | densify_from 600, densify_until 14900, densify_every 100, grad_threshold 2e-4, percent_dense 0.01, min_opacity 0.005, big_world 0.1, big_world_after 3000, reset_every 3000, reset_value 0.01, morton_every 5000, morton_until 15000, lr_opacities 0.025, fused_adam true, final_prune_opacity 1/255, total_steps 30000 |
| loss | photometric | base | — | 0.8 L1 + 0.2 D-SSIM (runtime SSIM, valid conv; fused-ssim not installed) |

No new FieldSpecs.

## Data flow
- Renderer: `_Rasterize` autograd Function = official `torch_bindings/rasterization.py`. Inputs are
  the raw parameters (fused activations): means, log-scales, raw quats, opacity logits `[N,1]`
  (view of `[N]`), sh0 `[N,1,3]`, shN `[N,15,3]` (separate SH buffers). Camera: w2c = `viewmat`,
  cam position = inv(viewmat)[:3,3], fx, fy, cx, cy from `K` (same pixel-centre convention as
  gsplat: pixel centre = index + 0.5). Output CHW → HWC.
- Densification info: renderer owns a persistent `[2,N]` float buffer (`FasterGSInfo` holder,
  official `Gaussians._densification_info`), (re)allocated with zeros when N changes; passed to the
  backward only while `holder.enabled`; the density reads it via `outputs["info"]["fastergs"]`,
  zeros it after ADC and disables it after the last ADC (official `update_densification_info`).
- Density = official `Model.adaptive_density_control` / `reset_opacities` / `apply_morton_ordering`
  / `training_cleanup` re-written against the SplatStudio param dict, with every population change
  routed through `gsplat.strategy.ops._update_param_with_optimizer` (append with zero moments,
  index/permutation incl. moments) or `gsplat.strategy.ops.remove`/`reset_opa`, so optimizer state
  stays aligned (`check_state` + a probe).
- Optimizer: `density.initialize` replaces each runtime `torch.optim.Adam` in place by
  `FusedAdam` (subclass of `torch.optim.Adam`, same state keys `step/exp_avg/exp_avg_sq`;
  `step()` launches the vendored `adam_step` kernel) — prior art: adamw-gs / taming-3dgs replace
  optimizers in `initialize`. Opacity lr set to 0.025 (code default; runtime default is 0.05).

## Step mapping (official 0-based callback iteration i runs before training step i; runtime
`after_backward(step=s)` runs after s steps ⇒ official i ≡ runtime s)
- s=0 (initialize): Morton sort.
- ADC at s ∈ [600, 14900], s % 100 == 0; prune-large only s > 3000; re-zero info if s < 14900,
  disable info after s ≥ 14900.
- Morton at s % 5000 == 0, s ≤ 15000 (after ADC, before reset — priorities 100/99/90).
- Opacity reset at s % 3000 == 0, 3000 ≤ s ≤ 14900.
- s == total_steps: training_cleanup prune (opacity < 1/255, degenerate quats). Changes only the
  count: such Gaussians are never rasterised (α < 1/255 culled in preprocess).
- SH degree (runtime: (s−1)//1000) = official.

## Probes (each fails if the mechanism is removed/wrong)
| probe | measures | fails if |
|---|---|---|
| `fastergs-render-reference` (cuda) | image and all 6 parameter gradients vs an independent float64 dense compositor (every Gaussian at every pixel, EWA + 0.3 dilation, α≥1/255 skip, front-to-back, T<1e-4 stop, black bg, raw-parameter activations) | rasterizer, fused activations, per-Gaussian backward or separate-SH path wrong |
| `fastergs-tile-culling` (cuda) | returned tile-instance count n vs independent counts: lower bound L = tiles with a pixel centre α≥1/255, rect AABB count R, 3σ-square count S (3DGS): require L ≤ n < R ≤… and n < S on an elongated rotated fixture | culling/opacity-aware bounds removed or over-culling |
| `fastergs-adam` (cuda) | FusedAdam vs torch.optim.Adam(eps=1e-15) over 5 steps with changing lr (rel 1e-5); kernel differs from plain SGD | fused Adam wrong |
| `fastergs-morton` (cuda) | after `morton_reorder` the means follow an independently computed (Python bit-interleave) Morton order; moments permuted identically; checks schedule {0,5000,10000,15000} | z-ordering removed |
| `fastergs-adc` (cuda) | ADC on a fixture: duplicate/split/prune sets equal an independent re-implementation of the official rule; moments of survivors preserved, new ones zero; schedule set (144 ADC steps, resets 3000..12000) | ADC rule / schedule / state alignment wrong |
| `fastergs-densification-info` (cuda) | backward accumulates count and NDC-scaled ‖∂L/∂μ2D‖ equal to the same quantities from the dense reference's μ2D gradients | info statistic wrong |
| base `state-transitions`, `photometric-gradient` | runtime contracts | — |

Base probe `renderer-gradient` is dropped (requires gsplat `means2d`, which this renderer does not
expose; the densification statistic is checked by `fastergs-densification-info`).

## Profiling (speed claim)
`python -m ss_faster_gs.profile --data <scene> --run <run with checkpoint> [--views 50]`:
on the same params, same GPU, same views: time forward, backward and optimizer update separately
(CUDA events, warm-up) for (a) Faster-GS kernels + FusedAdam and (b) the runtime gsplat renderer +
`torch.optim.Adam` (runtime default, non-fused); also (c) torch Adam(fused=True) as reference.
Writes `<run>/fastergs_profile.json`. Loss time reported separately (shared). GPUs are shared and
~100 % utilised by other users ⇒ numbers are ratios on the same GPU at the same time, noisy.

## Files
`plugin/pyproject.toml`, `ss_faster_gs/{__init__,kernels,raster,renderer,density,optim,morton,
probes,reference,profile}.py`, `ss_faster_gs/csrc/` (vendored + LICENSE), `tests/test_faster_gs.py`,
`README.md`.

## Fidelity gaps
1. Loss: runtime SSIM (valid conv, separable not fused) vs official fused-ssim (same padding).
   Shared with baseline; small quality effect, speed of the loss excluded from the claim.
2. View sampling with replacement (official: random permutation per epoch). Shared with baseline.
3. Means LR: runtime decays lr after each step (step s uses lr(s−1)); official uses lr(s). One-step
   offset, negligible.
4. Paper averages 5 runs; we run one seed. Scene-level noise up to ±0.5 dB (paper).
5. Baseline comparability: our gsplat baseline uses opacity lr 0.05, white background, gsplat
   kernels; the paper's 3DGS row uses opacity lr 0.025 and black bg. Compare Faster-GS PSNR to the
   paper's absolute numbers and the Δ vs our baseline to the paper's Δ vs 3DGS (≈0).
6. Kernels from `main` (16×12 tiles + warp-level culling, borrowed) rather than the paper's testbed;
   lossless speed features, checked by the reference probe.
7. Speed: A6000 (sm_86) shared with other jobs, CUDA 12.1 (authors: 12.8, RTX 4090). Training images
   are on the GPU in both paths. Runtime overheads (Python loop, check_state at log steps, per-step
   frame `on_device`) are common to both methods. No expandable-segments allocator setting.
8. LPIPS-VGG from `lpips` package (paper: unspecified).
