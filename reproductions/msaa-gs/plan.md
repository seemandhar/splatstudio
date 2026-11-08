# MSAA-GS native plan

Method `msaa-gs` (variant chosen: "Ours(3DGS)", the only one with per-scene numbers; the
Pixel-GS/AbsGS-based `msaa-gs-plus` rows are averages only and not implemented). Module `ss_msaa_gs`.

## Components
| kind | name | factory | requires → provides | config (paper value) |
|---|---|---|---|---|
| renderer | msaa-renderer | `ss_msaa_gs.renderer:create_renderer` | state.gaussian3d, camera.pinhole → render.rgb, render.depth.expected_z, gradient.screen.signed | `samples: 4`, `offsets: rotated-grid 4x`, `eps2d: 0.3`, `tile_size: 16` |
| density | adaptive-density | gsplat `DefaultStrategy` wrapper (built-in, unchanged; 3DGS reset) | gradient.screen.signed | 3DGS defaults |
| loss | aws-recon | `ss_msaa_gs.losses:create_recon` | render.rgb | `lambda_w 0.8, lambda_ssim 0.2, alpha 0.5, eps 1e-6` (replaces `photometric`) |
| loss | gdc | `ss_msaa_gs.losses:create_gdc` | render.rgb | `weight 0.1` |

C1 renderer: gsplat 1.5.3 classic pipeline (fully_fused_projection with opacity-aware radii,
eps2d 0.3, SH, isect_tiles) unchanged; `rasterize_to_pixels` replaced by a JIT CUDA kernel
`csrc/msaa_raster.cu` adapted from gsplat 1.5.3 RasterizeToPixels3DGS{Fwd,Bwd}.cu (via the
validated `reproductions/linear-splatting` port). Each thread (pixel) carries S=4 independent
blend states (T, accumulated colour, last index, done); every Gaussian of the shared-memory batch
is evaluated at the 4 sample positions; output = mean of the 4 colours (Eq. 5), alpha = mean of
the 4 alphas, depth = mean expected-depth accumulations / alpha. Backward: per sample, gsplat's
back-to-front recurrence with v_render/S, gradients summed over samples before one warp reduce +
atomic (so means2d.grad = sum over sample paths, the densification statistic of a fused MSAA
rasterizer). Tile lists are exact: sub-samples stay inside the pixel square, which lies inside
the tile gsplat's box test uses. Applies at train and eval.

C2/C3 losses: Eq. 6-9 and 10-13 with the reductions in paper.md §7 (w detached; L_w per-channel
mean; GDC mean|Δ∇x| + mean|Δ∇y| on (H-1)(W-1)).

## Probes
| probe | contribution | fails if |
|---|---|---|
| msaa-forward-vs-dense | C1 | CUDA render ≠ float64 dense reference averaging 4 single-sample composites at the offsets; or equals single-sample gsplat render (MSAA removed) |
| msaa-backward-vs-dense | C1 | CUDA gradients (means2d, conics, colours, opacities) ≠ autograd of dense reference |
| msaa-single-sample-equals-gsplat | C1 | with samples=1 at (0.5,0.5) the kernel ≠ gsplat rasterize_to_pixels (adaptation drift) |
| aws-weights | C2 | loss ≠ independent formula; or equals plain L1 (weights removed) |
| gdc-loss | C3 | loss ≠ independent formula on an analytic ramp; zero gradient on edges |
| loss-weights | C2/C3 | summed components ≠ 0.8 Lw + 0.2 DSSIM + 0.1 Lgrad |

## Files
plugin/pyproject.toml, ss_msaa_gs/{__init__,kernels,renderer,losses,probes,reference}.py,
csrc/{ext.cpp,msaa_raster.cu}, tests/test_msaa_gs.py, README.md.

## Fidelity gaps
1. No official code: offsets pattern, α, ε, loss reductions guessed (paper.md §7). Effect: small
   (α/reduction change loss balance by ≤3x on terms that are ~0.1 of the total).
2. λ3 schedule "can be gradually increased" — constant 0.1 used (paper's stated value).
3. Runtime protocol differences vs Inria 3DGS (white bg, sampling with replacement, valid SSIM)
   — compare deltas over our gsplat baseline to Table 2 deltas.
4. Table 2's 3DGS per-scene row is a retrained run (mean 27.49 ≠ 27.21): delta comparison uses it.
5. Training cost ≈ 4x rasterization work per step (projection/sort shared).
