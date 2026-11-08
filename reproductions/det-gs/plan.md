# DET-GS plan — native SplatStudio plugin `det-gs` (module `ss_det_gs`)

Base: `make_method("det-gs")` = gsplat renderer + 3DGS adaptive density (opacity-reset fix) +
photometric (0.8 L1 + 0.2 D-SSIM). All kept. Renderer is a thin subclass of the gsplat renderer that
additionally returns `outputs["camera"]` (viewmat/K/size) so losses can look up per-view priors.

## Components
| name | kind | factory | requires | config (paper value) |
|---|---|---|---|---|
| det-gs-renderer | renderer | ss_det_gs.renderer:create_renderer | state.gaussian3d, camera.pinhole | gsplat caps unchanged |
| adaptive-density | density | splatstudio gsplat (unchanged) | gradient.screen.signed | 3DGS |
| photometric | loss | splatstudio.losses (unchanged) | render.rgb | λ=0.2 (3DGS) |
| det-gs-depth (C1) | loss | ss_det_gs.losses:create_depth | state.gaussian3d, camera.pinhole | gamma 0.1, eta 1.0, every 5, patch 5..20, omega 0.95, error_tolerance 0.2 (DNG code default), space depth |
| det-gs-edge (C2) | loss | ss_det_gs.losses:create_edge | render.depth.expected_z | beta 0.1, eps 1e-8, cross 3x3 |
| det-gs-tv (C3) | loss | ss_det_gs.losses:create_tv | render.rgb | phi 0.8, tau_edge 1e-2, tau_smooth 1e-4 |

No new fields. Priors (DA-V2-L disparity, Canny 20/200 mask) precomputed offline by
`tools/prepare_detgs.py` into `<data>/detgs_priors.npz`, keyed by native viewmat
(`ss_det_gs.priors`, as DNGaussian/FatesGS plugins do). Dataset `mipnerf360-detgs` (hard-linked).

## Equations → code
- C1: hard depth = gsplat `render_mode="D"` with every opacity ω, scales/quats detached (gradient
  to means only, "depth supervision directly targets the mean positions"); target = −disparity;
  patchify F.unfold(p, stride p); Eq.5 `(x−μ_P)/(σ_P+δ)`, δ = 1e-2·std(all patches) (DNG); Eq.6
  `(x−μ_P)/σ_I` (+δ, σ_I detached, DNG); margin-L2 (DNG error tolerance); two independent patch sizes.
  Active on steps with step % 5 == 0; else exact 0 attached to graph.
- C2: D = ED·alpha (Eq.3 unnormalised), m = 1−canny; D̄ = conv(D·m, cross)/(conv(m, cross)+ε)
  with zero padding; L = Σ m (D−D̄)² / Σ m.
- C3: forward differences of pred and gt (HWC); M = |∇gt| < τ_edge per channel; hinge; mean over
  all H·(W−1)·3 and (H−1)·W·3 entries (each direction normalised by |Ω|·3 ≈ same count).

## Probes
| probe | contribution | fails if |
|---|---|---|
| depth-formula | C1 | normalisation/margin/weights differ from independent loop; loss not invariant to affine transform of disparity |
| depth-schedule | C1 | loss nonzero at step%5≠0 or zero at step%5==0 (CUDA, real hard-depth render; gradient only on means) |
| edge-formula | C2 | differs from independent per-pixel loop; edge pixels contribute |
| tv-formula | C3 | differs from independent loop; masked/sub-margin gradients contribute |

## Fidelity gaps
1. No official code. **Error tolerance = 0.2**: DNGaussian's code default. The paper does not state it; DNGaussian also uses
   0.01 (DTU/MVS) and 2.5e-4 (3-view LLFF, where it is effectively a no-op). It sets how strong the dominant η=1 term is,
   so it is the largest single uncertainty. δ, λ, ω also come from DNGaussian / 3DGS.
1b. Depth vs disparity supervision space: unspecified; default z vs −disp (`space` option exists).
1c. The depth loss shares the Adam step with the colour loss (DNGaussian takes a separate step).
1d. Logging: `--log-every` multiples of 5 always land on depth steps, so the logged det-gs-depth and total overstate
    the average objective. The average depth contribution is about the logged value / 5.
2. z-depth instead of ray distance (Eq.3/4). Small.
3. Runtime: random-with-replacement view sampling, white background, gsplat rasterizer (cf. calibration).
4. Paper's 3DGS row is above the 3DGS paper → compare gains: paper garden +0.36, counter +0.95 dB.
5. TV channel averaging and edge-loss depth choice (soft depth) are interpretations.

## Review-1 diagnostics (garden, no ablation)
Per-loss gradient norms relative to photometric (`tools/grad_diag.py`, 6 train views, weighted). Means: depth 3.6× / 5.1× (on depth steps),
edge 0.0025× / 0.011×, TV 0.04× / 0.07× at steps 500 / 2000. Opacities: edge 0.008×, TV 0.03–0.12×.
L_edge is therefore nearly inert in COLMAP z units (gap: the paper's +0.2 dB edge ablation is unlikely to reproduce).
2k smoke with tolerance 0.2: PSNR 22.85, vs 23.01 with 2.5e-4 and a gsplat smoke baseline of about 22.96–23.02.
