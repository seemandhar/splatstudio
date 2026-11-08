# Review 1 — denoising-gs (arXiv 2605.14880)

**Verdict: not approved. 1 critical, 1 major, 5 minor.** The MCMC backbone, C3, C4, the cap and
protocol safety are sound. C2 has the wrong sign compared with the paper's Fig. 2 and its text. C1
is numerically a no-op under the literal reading, while the paper's own α ablation shows it has an
effect.

## Measured magnitudes (short check, no training)
I loaded `evidence/smoke-mcmc/checkpoint.pt` (garden, 2k steps, 288k Gaussians, scene_scale 4.92)
and computed per-view gradients of L1+0.2·D-SSIM+reg on 5 train views. Script:
`/tmp/claude-1013/-home-sejain/6ad4135e-056e-4bd4-9844-6b8764bea041/scratchpad/mag.py`.
- |∇M|: p50 7.5e-6, p99 3.3e-4. |∇log s|max: p50 9.9e-8, p99 4.7e-6.
- C1 offset α·λ_30k·|m| at p99 ≈ **2.5e-10 world units per step**. The Adam step at the same LR is
  ≈1.5e-5, so C1 is about 5 orders of magnitude smaller.
- C2 offset β2·|∇S_j| at p99 ≈ 2.4e-6 per step (median Gaussian scale 0.04). 35 % of Gaussians are
  "aligned", and **in 100 % of cases Δμ·∇M > 0**, i.e. Δμ always points up the gradient (ascent).
- Smoke runs: quick vs mcmc control are 22.8705 vs 22.8694 dB, no measurable effect.

## Critical
1. **C2 sign is inverted** (`core.py:denoise_term`, line `mag = |gS_j| * sign(g_local_j)`; applied at
   `density.py:259`, `means.add_(beta2*world)`).
   - Paper Fig. 2 (`Figures/denoise1111.pdf`): the target lies to the left. The mean moves left
     (`M + ∇M + ΔD → M̃`), and Δμ moves it **further left**, in the direction of the update. The
     figure's notation uses "∇M" for the applied update.
   - §4.3 says the sign "ensur[es] that the denoising displacement occurs in the **same directional
     sense as the original gradient update**". The original update is −λ∇M.
   - The code adds +|∇S_j|·sign(∇M^G_j). That is ascent, opposite to every mean update (measured 100 %).
   - Fix: make descent the default (`Δμ_j = −|∇S_j| sign(∇M^G_j)`, equivalently the sign of the
     update) and keep the literal sign as an option. Record this as resolved ambiguity #1 in
     paper.md §7. Update the probe reference to match.

## Major
2. **C1 is a numerical no-op** (`core.py:momentum_drift`, `density.py:253-256`). It uses the raw
   gradient EMA times λ_t. This is the literal MSGLD form of Kim et al. 2020, but there the gradient
   step is also raw. In 3DGS the gradient step is Adam-normalised, so −αλm is about 1e-5 of the step.
   - The paper's Table 3(a) contradicts a no-op. On DB, α = 0.05 / 0.1 / 0.5 gives 29.87 / 29.40 /
     29.53 dB, and β1 = 0.1 gives 29.49. An offset of 1e-10 cannot move PSNR by 0.4 dB.
   - Where m comes from: "Adam's first moment" with β1 = 0.9 is the same raw EMA, so it does not
     change the scale. Only an Adam-normalised drift does.
   - Fix: add `momentum_source: "adam"`, drift = −α·λ_t·m̂/(√v̂+ε), with m and v taken from
     `optimizers["means"].state` (exp_avg / exp_avg_sq with bias correction) or kept per Gaussian.
     Make it the default, keep `raw` as the literal variant, and list it as a fidelity gap. Without
     this, C1 cannot contribute, and the report must say so.

## Minor
3. **Δμ magnitude reading** (`core.py:denoise_term`). Fig. 2 writes `s + ∇s_j`, which uses ∇ for the
   applied update. So |ΔS_j| (the Adam scale step in world units, roughly exp(s)·5e-3·|ŝ|) is an
   equally plausible magnitude, and it is much larger than the raw |∇S_j| (p99 4.7e-6). Keep the
   literal reading, but offer the option and list it in the gaps.
4. **Schedule tied to config, not to `--steps`** (`density.py:140-142`, `DEFAULTS.total_steps=35000`).
   Launching with `--steps 30000` puts the prune on the last step and never runs SCR. Read `--steps`
   from argv the way `--data` is read, or assert they agree.
5. **Schedule hints in the main text** (no Appendix). The guess (prune at 30k, SCR 30.5k–32.5k) is
   consistent with them:
   - Table 2 counts are exactly 0.9× the MCMC row plus a tiny remainder (5.90 → 5.32, i.e. 5.31 + 5
     SCR splits). So the prune comes after MCMC growth stops (≥ 25k) with no regrowth.
   - §6 says SCR adds points "under the 10 % pruning setting from the previous denoising stage", so
     SCR comes after the prune.
   - §6 gives 13.9K added points for 0.1 % = 0.1 % × 2.789M × 5. The 5.6K figure for 0.05 % does not
     fit 5 applications (that would be 7.0K) and is probably a paper typo.
   - Nothing pins 30k versus 25k. Record this as a guess (already in gap 1).
6. **Comparability**. The paper's MCMC row (26.15 / 29.51) is the original 3DGS-MCMC paper's numbers
   at 30k steps (its official caps: bicycle 5.9M, counter 1.2M). Denoising-GS runs 35k. Our control
   uses the same cap at 35k, which is the right comparison, but say in the report that part of the
   paper's +0.02 / +0.06 dB may come from the 5k extra steps. Other small differences: runtime LR
   decay over 35k versus official MCMC's 30k (`position_lr_max_steps`), and noise uses the post-decay λ_t.
7. **Fisher pass at SH degree 3 fixed** (`density.py:291`). This is correct at 30k, but the quick
   variant prunes at 1500 when the runtime is at degree 1, so the smoke run is slightly off. Pass
   the runtime degree, e.g. `min((step-1)//1000, 3)`.

## Checked and OK
- **MCMC backbone equivalence** with gsplat `MCMCStrategy` and official 3dgs-mcmc:
  - noise_lr 5e5 × λ_t via `inject_noise_to_position`; relocation of opacity ≤ 0.005.
  - +5 % growth every 100 steps for 500 < step < 25000, up to the cap.
  - λ_o = λ_s = 0.01 L1 on activated opacity and scale (`losses.py`); init opacity 0.5.
  - Momentum state stays aligned through relocate (sources zeroed by gsplat, dead slots zeroed at
    `density.py:278`), add, remove and split.
  - The control is closer to official MCMC than the built-in `gsplat-mcmc`, which has no regulariser,
    init opacity 0.1 and a 1M cap.
- **Cap**:
  - Source is the paper's own Table 2 MCMC count, which equals the official 3dgs-mcmc caps; it is not
    "0.9×". The 0.9× counts come from the 10 % prune.
  - Lookup uses `Path(--data).name` and raises an error for unknown scenes, so it is honest. The cap
    source is logged in `history`.
  - bench passes `--max-gaussians 12e6`, so 5.9M fits.
- **C3**:
  - u = Σ_v‖∇_{M, log S} L_v‖² = trace F = Σσ (F is PSD). L_v is the squared-L2 sum, as in Eq. f2.
  - The *lowest* u is removed. That is correct: §4.4 says "primitives with smaller u values are
    considered to have higher uncertainty … pruned", and u is defined as an inverse measure.
    "Remove the top 10 % most uncertain" means lowest u.
- **Fisher protocol**:
  - Loads `splits["train"]` only and checks each reloaded view's K, viewmat and size against the
    runtime cameras.
  - Renders on detached copies, so the model is not modified. White background matches the runtime
    renderer.
- **Hooks**: `register_hook` on leaf Parameters receives the full summed gradient (photometric + reg)
  before `zero_grad`, and is re-registered each step on the current Parameters. Fine.
- **requires_grad pitfall**: avoided. `split_selected` builds `Parameter(cat, requires_grad=p.requires_grad)`,
  and all other operations use gsplat ops.
- **C4**: d is the mean squared 3-NN distance excluding self, with the top 0.05 % largest selected.
  Splits use N = 2 (`ratios = 2`, topk indices unique). Copy moments are zero, the same as
  `sample_add`. Correct.

## Launch (after fixing 1–2)
```
python3 tools/bench.py launch --exp denoising-gs-35k --methods denoising-gs denoising-gs-mcmc \
  --dataset mipnerf360 --scenes bicycle counter --steps 35000 --recipe 3dgs
```
Put the run directory on the overflow disk first:
`mkdir -p /bigdata/sejain/runs/denoising-gs-35k && ln -s /bigdata/sejain/runs/denoising-gs-35k runs/`.
Compare the gain over the control with the paper's +0.02 / +0.06 dB, and also against
`baseline3dgs-mipnerf360-30000`.
