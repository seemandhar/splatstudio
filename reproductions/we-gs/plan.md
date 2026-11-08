# WE-GS native plan

Method `we-gs`, module `ss_wegs`, package `splatstudio-method-we-gs`. Ablation entry points:
`we-gs-noressh` is NOT provided (paper ablations not benchmarked).

Prior art reused (pattern, not code paths): `reproductions/swag/plugin/ss_swag` — run-wide
learned state in a module-global bank owned by the density, stepped by its own Adam in
`after_backward`, exported to `density_state["bank"]`, rebuilt by a custom evaluator
(`ss_swag.evaluate`). WE-GS differs: no per-image embedding table — the embedding is encoded
from the image, so the evaluator is feed-forward (paper protocol) instead of NeRF-W left-half
optimisation.

## Components
| component | kind | factory | requires → provides | config (paper value) |
|---|---|---|---|---|
| wegs-renderer | renderer | `ss_wegs.renderer:create_renderer` | state.gaussian3d, camera.pinhole → render.rgb, gradient.screen.signed | background 0 (code), sh_from_bank |
| wegs-density | density | `ss_wegs.density:create_density` | gradient.screen.signed | 3DGS strategy (refine 100, 500-15000, reset 3000, grad 2e-4); position_lr_max_steps 30000; bank_* (dim 16, widths, lrs) |
| wegs-masked-photometric | loss | `ss_wegs.losses:create_masked_photometric` | render.rgb | l1 0.8, ssim 0.2 (Eq. 19); DEVIATION: M detached in D-SSIM |
| wegs-mask-reg | loss | `ss_wegs.losses:create_mask_reg` | render.rgb | weight 1e-5, sum over M_k at W/4×H/4 (Eq. 20) |
| wegs-sh-reg | loss | `ss_wegs.losses:create_sh_reg` | render.rgb | weight 0.1, reduction mean-over-Gaussians of L2 (Eq. 13) |
| wegs-transient-opacity-reg | loss | `ss_wegs.losses:create_transient_reg` | render.rgb | weight 1e-5, eps 0.1 (Eq. 21; chosen, paper.md §8) |

The base `photometric` loss is **replaced** by Eq. 19 (paper replaces it).

## Bank (`ss_wegs.bank.WEGSBank`, ACTIVE global)
- `encoder: SpatialAttentionEncoder` — input = the training image (W' = W/4, paper; resized to ×16), Enc¹ (3 convs,
  code layer shapes, →512×64×64), CAM (avg/max pool → shared FC 512→32→512, sigmoid; Eq. 15-16),
  U-Net (conv s2 512→128, conv s2 128→128, convT 128→128 (+skip), convT 128→1, sigmoid → M
  64×64; Eq. 17), F'' = M·F' (Eq. 18), Enc² (2 convs s2 + avgpool + 1×1 conv → 16; Eq. 18 with
  residual F''+F).
- `transnet: TransNet` — input [sh (48, PE 0 freq), PE₄(μ) (27), l (16)] → 256 → 256 → 256 → 48,
  ReLU, tanh output (Eq. 9).
- Adam eps 1e-15: transnet 1e-4, Enc¹/CAM/Enc² 1e-4, U-Net 1e-5 (code lrs).
- export/load for the evaluator.

## Renderer
`camera["image"]` (present in training and runtime evaluation frames) → `bank.encode(image)` →
`(l, M)`; `Δsh = transnet(sh, μ, l)`; rasterise `sh + Δsh` with gsplat SH (degree from runtime).
Outputs: `rgb, alpha, info, static_mask (1/4 res), static_mask_full (H×W bilinear), sh_residual`.
Without a bank (plain ss-eval) it raises — no silent fallback; the evaluator is mandatory.

## Probes
| probe | contribution | fails if |
|---|---|---|
| sh-transfer (cuda) | C1 | rendered colours ≠ independent SH(sh+G(sh,γ(μ),l)); l has no effect; no grads to transnet/sh |
| attention-encoder | C2 | Eq. 15-18 independent recomputation differs; M not in [0,1] at 1/4 res; M has no effect on l |
| masked-loss | C3 | gradient reaches render pixels where M=0; regM ≠ λΣ(1-M)²; M gets no gradient |
| sh-reg | C1 | value ≠ λ·mean‖Δ‖ |
| transient-opacity-reg (cuda) | C4 | gradient on α of Gaussians projecting where M≥ε, or none where M<ε |
| bank-training (cuda) | all | an optimiser step does not move encoder, U-Net and transnet; lrs differ |
| feedforward-eval (cuda) | protocol | evaluator does not encode the test image itself |

## Fidelity gaps
0. DEVIATION: M stop-gradient inside the D-SSIM term of Eq. 19 (literal form collapses to M ≡ 0; paper.md §8). ε = 0.1 chosen. L_regSH reduction = mean over Gaussians (paper Σ_j). Black background (code) vs runtime white (GT has no alpha).
1. Unreleased training code: L_regM/L_regSH reductions, ε, U-Net/CAM widths are assumptions
   (paper.md §7). Effect: mask strength; could change PSNR by >1 dB. Closable only with code.
2. Paper vs released inference model differences (8-layer G_φ, dim 32, SH 5, cat-mask encoder,
   ULite) — paper followed.
3. Runtime: random-with-replacement view sampling; SSIM valid conv; LPIPS from `lpips` (Alex for
   headline). gsplat rasteriser instead of Inria.
4. Runtime's own final `validation.json` is produced with the live bank (feed-forward) and is
   the same protocol as the evaluator; ss-eval cannot be used.
5. Trevi Fountain not prepared → 2 of 3 scenes.
