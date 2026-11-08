# Gap analysis 1 — confidence-densify (30k, 1M budget, mipnerf360 garden + kitchen)

## Gap table (test PSNR)
| scene | ours | paper | abs gap | our control (MCMC) | paper MCMC | backbone offset | our gain | paper gain | delta_gap |
|---|---|---|---|---|---|---|---|---|---|
| garden | 27.15 | 27.68 | −0.53 | 26.64 | 26.86 | −0.22 | +0.51 | +0.82 | −0.31 |
| kitchen | 31.59 | 32.39 | −0.80 | 31.63 | 31.63 | 0.00 | −0.04 | +0.76 | **−0.80** |

Classification: garden is mostly protocol/backbone (−0.22) plus −0.31 of method gap. Kitchen is
a **method** gap: the backbone matches exactly, and our method adds nothing.

## Evidence
- **Train-view PSNR at 30k** (every 6th train view; no test images used):
  - kitchen: ours 32.84 vs control 32.66 (+0.18).
  - garden: ours 28.55 vs control 28.11 (+0.44).
  - So indoors the method already under-delivers on the training views; this is not only a
    generalisation problem.
- **Kitchen per test view:**
  - Ours is +0.3 to +0.6 dB on most of the 35 views.
  - It is −0.6 to −2.5 dB on 5 views (test views 0, 7, 8, 15, 31).
  - The extra error sits in the farthest depth quartile and the top-left of the image: bright
    window / plant background (`scratchpad/gap/kitchen-15-diff.png`).
  - L1 is nearly equal overall, so a few high-MSE background regions cancel the gain.
- **Mechanics run as specified:**
  - The cap is reached at 3.5k (kitchen) / 4.6k (garden).
  - Relocations follow the share schedule: 100k per 400 it from 4k to 10k, 77k at 13.6k,
    13.5k at 20k, 3k at 23.2k.
  - Loss terms are active, and λ_reg stops at 25k.
- **Confidence:**
  - Per-primitive γ̄ median is 2.6 (kitchen) vs 1.8 (garden).
  - 20 % of kitchen pixels sit at the Ĉ = 5 clamp.
  - The hardest 5 % of pixels hold about 50 % of the training MSE in both scenes. Relative to
    the remaining pixels they are weighted 0.63× (kitchen) and 0.41× (garden).

## Hypotheses (ranked)
1. **The confidence-weighted loss under-fits hard pixels.** The paper's text is ambiguous on
   whether L_conf replaces L_rgb (CoMe) or only trains γ.
   - Tested as post-hoc variant `confidence-densify-sep`: plain L_rgb for geometry, L_conf on a
     detached L_rgb trains γ only. Kitchen, 7k steps, train-view PSNR: 30.457 vs 30.459 for the
     CoMe reading. **No effect**, so it is rejected as the cause. The variant stays in the plugin
     only as a recorded negative diagnostic and is not proposed for benchmarking.
2. **Floaters or overfitting in under-observed indoor background**, from relocation to
   high-error background sources plus GT colour init.
   - Consistent with the 5 bad kitchen test views.
   - It cannot explain the small train gain (+0.18).
   - No grounded fix: the paper's schedule and rules are implemented as written.
3. **Unstated implementation details of the authors' Faster-GS / 3DGS codebase.** Candidates are
   the exact SSIM-term reduction, how rank ties are handled, the growth-primitive w window, and
   moment resets (paper.md §7).
   - Each is plausibly worth ≤0.1–0.2 dB, and none has an official-code reference.
   - Testing them would amount to an ablation sweep.
4. **Single-seed noise.** Roughly ±0.15 dB; too small for −0.8 dB.

## Recommendation
No grounded fix expected to close ≥0.3 dB was found. Report kitchen as not matched. The
backbone matches the paper's MCMC exactly (kitchen) and to −0.22 dB (garden). The method's gain
is reproduced in part outdoors (+0.51 of +0.82) and not at all indoors (−0.04 of +0.76).
Diagnostic runs are kept in `runs/confidence-densify-diag/` (on `/bigdata`).
