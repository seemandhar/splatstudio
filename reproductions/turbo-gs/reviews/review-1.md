# Review 1 — turbo-gs (Turbo-3DGS)

**Verdict: not approved — 0 critical, 1 major, 5 minor.** The major issue is an interpretation
choice, not a coding bug. Fixing it takes one line (or it can be accepted as a labelled gap).
Everything else checks out against the paper. Unit tests: 5 passed.

## Builder's claim about the supp. per-scene table: confirmed
The supp. Table "PSNR comparison" has identical rows for `Turbo-Scaffold-GS` and `Turbo-3DGS`.
The single "Ours" rows in the other per-scene tables also average to the Turbo-Scaffold-GS
column of main Table 2, not to Turbo-3DGS:

| table | DB mean of per-scene row | Turbo-Scaffold-GS (Tab. 2) | Turbo-3DGS (Tab. 2) |
|---|---|---|---|
| PSNR | (29.85+31.12)/2 = 30.49 | 30.48 | 29.21 |
| SSIM | (0.907+0.915)/2 = 0.911 | 0.911 | 0.905 |
| LPIPS | (0.241+0.224)/2 = 0.233 | 0.233 | 0.240 |
| time | (4.04+3.98)/2 = 4.0 m | 4.0 m | 12.13 m |

M360 PSNR 9-scene mean = 27.43, which equals Turbo-Scaffold-GS. Conclusion: Turbo-3DGS has
averages only, and the DB-average comparison (29.21) is the right target.

## Major

**M1. The convergence-aware budget is effectively inert. M_ad saturating at 1.5 M follows the
literal Alg. 1. The likely misreading is the t-normalization.** (`budget.py:50-53, 94-97`,
`density.py:158`)
- Alg. 1 really does say `rate>0.05 → M_ad=min(1.1 M_ad, 1.5 M)`. While the loss decays as a
  power law, rate ≈ the decay exponent (0.15–0.6 in smoke-1), so saturation at 1.5 M after about
  5 events is the literal algorithm. The ×1.1 growth is not a misreading.
- With t mapped over the densify window [500, 15000] → [1, 100], the cap reaches B(100) = 1.5 M:
  4.91 M on drjohnson and 3.49 M on playroom.
- Our gsplat baseline peaks are 3.61 M and 2.26 M. With α ≈ 0.1–0.3 the curve is front-loaded
  (B ≈ 0.5·M_ad by step 3000), so the cap never binds. smoke-1 shows `selected == candidates`
  at all 15 events. CA reduces to "no cap except on colour bursts".
- The paper's peak #G is about M, not 1.5 M: DB 2.82 M vs mean M 2.80 M (1.01×), and M360
  3.64 M vs 3.31 M (1.10×). Ablation Tab. 3 also says CA cuts time (14→12 m) and slightly lowers
  3DGS quality, so in the paper the budget binds.
- Alg. 1 lists `steps: Total steps` as an input but never uses it in the body. That points to
  **t = 100·step/steps** (steps = 30k). Densification then stops at t = 50, and
  B(50) = (50^α−1)/(100^α−1)·M_ad ≈ 0.75–0.82 × 1.5 M ≈ 1.1–1.2 M. That matches the paper's peak
  ratios.
- Sign question: Eq. 5 and Alg. 1 **agree** if the α are fitted log-log *slopes* (negative):
  deviation = −rate − α_base = slope_recent − slope_hist = ε. They disagree only under a
  positive-exponent reading. The builder's `deviation = α_base − rate` (positive α_base) is the
  same quantity and matches the prose ("slower convergence → slow densification → larger α").
  No bug there; see m2.
- Fix: use `t = 100·step/steps` (pass the total steps to the density, clamp to ≥ 1). If you
  keep the current reading, label it a gap. In either case, log peak #G and compare
  peak(method)/peak(baseline) with the paper's 2.82/2.46 = 1.15 (DB).
- PSNR consequence: with an inert cap, the run is ≈ 3DGS + DR + CG, which is likely ≥ baseline.
  The paper's Δ is −0.54 dB. Our DB baseline average is 29.28, so the relative target is ≈ 28.74.

## Minor
- **m1** `renderer.py:35-36`: paper Eq. 3 is `p = d·p_ds + ½(d−1)`, the block centre with no
  offset term. The code uses `p = d·p_ds + o`, the chessboard sub-lattice. This is the only
  reading consistent with the 4-offset merge validation and Fig. 3. Correct choice; record it as
  an interpretation.
- **m2** `budget.py:90-100`: under the literal Alg. 1 (α_base = negative slope), α would clip to
  0.1 almost always. The builder uses a positive exponent (α ≈ 0.1–0.3). Both are front-loaded.
  The effect is small while M1 stands, but document it.
- **m3** `density.py:153`: the 20 % branch is one Bernoulli draw per event (paper: "activate it
  with a probability of 20 %", per event or per Gaussian unstated). With `Random(0)` the first
  firing is event 26 (step 3100) and 25/145 events fire. τ_color = 0.01 τ_pos and p = 0.2 are
  verbatim §4.3. The colcheck-p1 run shows the branch is selective (+14 k on 90 k).
- **m4** `density.py:123`: the colour gradient is taken w.r.t. SH-space `sh0`. A gradient w.r.t.
  RGB colour would be 1/C0 ≈ 3.5× smaller, which changes what τ_col = 2e-6 selects. The paper
  does not name the attribute. Keep it, but list it as an interpretation (already in paper.md §7).
- **m5** probes: `turbo-color-densify` tests `select()` only. No probe checks the
  `refine()`→scheduler wiring end-to-end (that allowance = B(t) − n is enforced on a real event),
  so a regression there would go unnoticed.

## Verified (no issue)
- Dilated intrinsics: fx/d, (c − o − ½)/d + ½ are exact. eps2d = 0.3/d² gives ds covariance
  (Σ_hr + 0.3)/d², with radius r/d. Merge probe: 151 dB, vs the paper's > 70 dB from its tile
  misalignment.
- d = 2 is fixed with no schedule, which matches the paper (§4.1 has no dilation schedule).
  One uniform offset per step.
- Target `image[oy::2, ox::2]`. Eval renders full resolution under the runtime's `@no_grad`
  `measure`.
- The gsplat NDC normalization of means2d gradients uses ds width, which is invariant to d.
- 3DGS schedule: refine at steps 600…14900 every 100 (`>500`, `<15000`); reset to 0.01 at
  3000…12000 through the plugin's own `reset_opa` (avoids the gsplat reset bug). Clone then split,
  grow_scale3d 0.01, prune opacity 0.005, world-size prune after the first reset, no screen-size
  prune.
- Top-k cap: scores are masked to candidates, `topk(allowance)`, and zero allowance gives no
  densification.
- requires_grad pitfall: not present. All topology/opacity edits go through
  `gsplat.strategy.ops` (Parameters are rebuilt with `requires_grad` preserved). Retained grads
  on non-leaf `means2d`/`colors` survive `zero_grad(set_to_none)`.

## Launch (after M1 is fixed or accepted)
`python3 tools/bench.py launch --exp turbo-gs-db --methods turbo-gs --dataset db --scenes drjohnson playroom --steps 30000 --recipe 3dgs`
