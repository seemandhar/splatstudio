# splatstudio-method-turbo-gs

This is a native SplatStudio port of the Turbo-3DGS variant from **Turbo-GS: Accelerating 3D
Gaussian Fitting for High-Quality Radiance Fields** (Lu et al., CVPR 2026,
[arXiv 2412.13547](https://arxiv.org/abs/2412.13547)). Paper fidelity is unverified.

The official repository, https://github.com/inspirelt/Turbo-GS (commit aeddd12), contains only
a README. Everything here was written from the paper and its supplementary Algorithm 1.

## Mechanisms
| paper | code | values |
|---|---|---|
| Dilated rendering (Sec. 4.1) | `renderer.py`, `losses.py` | d = 2, random offset in {0,1}^2 per step |
| Convergence-aware budget (Sec. 4.2, Eq. 5-6, Alg. 1) | `budget.py` | λ = 0.5, EMA 0.1, warm-up 100, ±0.05 rate, ×1.1/×0.9, M_ad ∈ [0.5, 1.5]·M, α ∈ [0.1, 2] |
| Position + colour densification (Sec. 4.3) | `density.py` | τ_pos = 2e-4, τ_col = 0.01·τ_pos, colour branch p = 0.2 |

**Dilated rendering** is the gsplat 1.5.3 rasterizer with two changes:
- sub-pixel shifted intrinsics `fx/d, (cx − o − ½)/d + ½`;
- `eps2d = 0.3/d²`.

Each downsampled pixel then evaluates the full-resolution Gaussian, with the 0.3 px² high-res
filter, at its high-res pixel centre. Interleaving the 4 offsets reproduces the full render
(probe: 151 dB). Evaluation renders the full image.

**Budget M**: the paper only says M is "manually set". We use Taming-3DGS's `full_eval.py`
big_budgets (= the 3DGS counts). The paper's Turbo-3DGS peak #G on Deep Blending is 2.82 M; the
mean of those budgets for the DB scenes is 2.80 M. Scenes are keyed by initial SfM count.
Override with the density config `budget`.

## Interpretations (no official code)
- `t` in Eq. 6 is training progress, `t = clamp(100·step/total_steps, 1, 100)`, because Alg. 1 takes "Total steps" as input. Densification ends at t = 50, so the cap binds. `t_mode: "densify_window"` gives the alternative reading.
- The dilation offset is p = d·p_ds + o (chessboard sub-lattice); the colour branch is one draw per event.
- `deviation = α_base − rate`, which follows Alg. 1 with α_base read as a slope. A slower recent
  decay raises α and slows densification, as the prose says.
- The colour gradient is ‖∂L/∂sh0‖ averaged over the steps where the Gaussian is visible.
- When the candidates exceed B − n, the top max(g_pos/τ_pos, g_col/τ_col) are kept.
- The 3DGS densification schedule (500–15000 / 100, opacity reset 3000) is assumed.

## Fidelity gaps
- Batched training and "better initialization" (README TL;DR) are undocumented and not ported.
- The Taming-3DGS speed kernel is not ported; it changes speed only.
- Sensitivity pruning (the † rows) is not used.
- The density reads the loss value from `outputs["training_loss"]`, written by this plugin's
  `photometric` loss. It raises if that value is missing.

## Commands
```bash
pip install -e .
ss-inspect turbo-gs && ss-verify turbo-gs --device cuda
ss-train turbo-gs --data ~/datasets/tandt_db/native/drjohnson --output runs/x --recipe 3dgs \
  --num-gaussians 10000000 --max-gaussians 12000000
```
