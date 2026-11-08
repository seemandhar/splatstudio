# splatstudio-method-loss-aware-dc

A native SplatStudio port of **Unified Loss-Aware Density Control for 3D Gaussian Splatting**
(Kim, Chung, Heo, NeurIPS 2026, https://arxiv.org/abs/2611.02044).

No official code has been released: the author's repository contains only a README, at commit
267f5ef. Everything here is implemented from the paper text, the main equations and appendices
A.1–A.5. The method builds on SteepGS (https://arxiv.org/abs/2505.05587). This repository's
`reproductions/steepgs` port was used to understand SteepGS; no code is copied from it.
Paper fidelity is unverified.

## Mechanism (Alg. 1, priority Split → Clone → Decay)
Every 100 steps in the interval (500, 15000), the method does the following:
- **Split** (Eqs. 5–9):
  - A per-Gaussian splitting matrix in φ = (p, log s) is averaged over the views since the last
    densification.
  - It is normalised by J = blockdiag(R diag(s), I/√2), and its smallest eigenpair is computed.
  - Gaussians that are stationary (‖ḡ_p‖ ≤ 1e-4 and ‖ḡ_log s‖ ≤ 1e-4), uncertain (see the gaps)
    and have λ_min < −1e-8 are split.
  - Each split produces two children at p ± L v_p, log s ± v_s/√2, each with opacity o/2.
- **Clone** (Eqs. 10–13): η = −g_dup/h_dup. A duplicate with opacity η·o is appended when
  g_dup < 0 and 1e-6 < η ≤ 1e-2.
- **Decay** (Eqs. 14–17): η = g_dec/h_dec. Then o ← (1−η)·o when g_dec > 0 and 1e-4 < η ≤ 1e-2.
  This replaces 3DGS's global opacity reset.
- **Prune:** remove Gaussians with o < 0.005, plus the 3DGS world-size prune after step 3000.

## Implementation
- `csrc/stats.cu` is a custom CUDA kernel with two parts:
  - `surrogate_stats` replays gsplat 1.5.3's forward traversal over gsplat's own tile lists. Per
    Gaussian it accumulates g_z, H_z (the screen-space gradient and Hessian of α weighted by
    ∂L/∂α), g/h_dup and g/h_dec.
  - `joint_split_accumulate` pulls these back to (p, log s) with exact hyper-dual derivatives of
    gsplat's projection.
- `surrogates.py` holds the proxy curvature ω (App. A.5), a functorch reference of the
  splitting matrix, the normalisation, the eigen-split, and the selection rules.
- `density.py` contains Alg. 1. `renderer.py` is the runtime gsplat renderer plus the camera
  and SH degree. `losses.py` is the built-in L1 + 0.2 D-SSIM loss, which also passes the target
  to the density, as ω needs it.

## Config (density component; defaults are the paper's values)
| key | default | source |
|---|---|---|
| tau_pos, tau_scale | 1e-4 | App. A.1 |
| tau_split | −1e-8 | App. A.1 |
| tau_clone | 1e-6 | App. A.1 |
| tau_decay | 1e-4 | App. A.1 |
| eta_max | 1e-2 | Eqs. 13, 17 |
| charb_eps | 1e-3 | App. A.5 |
| gamma_ssim | 1e-4 | App. A.5 |
| ssim_weight | 0.2 | |
| eps_ssim | 1e-8 | unspecified in the paper |
| require_uncertain | true | SteepGS filter, mean ‖∇₂D‖ ≥ 2e-4; ambiguity |
| densify_from / every / until | 500 / 100 / 15000 | |
| prune_opa | 0.005 | |
| big_world | 0.1 after 3000 | |
| split / clone / decay / drop_mixed_term | true | ablation switches |
| exempt_new_clones_from_prune | false | diagnostic switch, not the paper |

## Fidelity gaps
1. **No official code.**
   - The "unresolved view-space variation" filter is an assumption: the text mentions it, but
     Eq. 8 omits it.
   - ε_ssim is unspecified.
   - The expectation is normalised as a sum over pixels, averaged over visible views (SteepGS).
2. **Clones are almost always pruned.** A clone's opacity is η·o ≤ 0.01·o, so with Alg. 1
   read literally it is removed in the same interval by the o < 0.005 prune.
3. **Background.** White, as in the runtime and the shared baseline; the paper uses black (the
   3DGS default).
4. **Shared runtime differences.**
   - Density control runs after the optimizer step; the statistics use pre-step snapshots.
   - Views are sampled with replacement.
   - SSIM uses valid convolution.
   - Rasterizer constants are gsplat's: α clamp 0.999 and T stop 1e-4.

## Commands
```bash
tools/ssenv pip install -e reproductions/loss-aware-dc/plugin
tools/ssenv ss-verify loss-aware-dc --device cuda
tools/ssenv ss-train loss-aware-dc --data ~/datasets/mipnerf360/native/garden --output runs/x \
  --steps 30000 --recipe 3dgs --num-gaussians 10000000
```
