# DWTGS gap analysis 1

**Benchmark:** `runs/dwtgs-llff-r8`: 8 LLFF 3-view scenes at 1/8 resolution, 10k steps, seed 0, plugin defaults after review 1 (averaging Haar, interpolated novel views).

**Result:** mean **20.12 / 0.698 / 0.232**. The paper reports **20.78 / 0.720 / 0.197**, so the absolute gap is **−0.66 dB**.

**Comparison base:** the paper's gain is +0.39 over its own DropGaussian re-run (20.39). Our DropGaussian (`runs/dropgaussian-i1-llff-r8`, same base plugin and protocol) scores **20.36**, matching the paper's re-run to within 0.03 dB.
- The baseline offset is therefore about 0. This is **not a protocol gap**.
- delta_gap = (20.12 − 20.36) − (+0.39) = **−0.63 dB**, so the gap comes from **the method** (the DWTGS terms).

## Per scene, DWTGS vs our DropGaussian (seed 0)
| scene | DWTGS | DropGaussian | Δ | #G DWTGS / DG |
|---|---|---|---|---|
| fern | 22.80 | 22.73 | +0.07 | 109k / 75k |
| flower | 20.52 | 20.82 | −0.30 | 110k / 73k |
| fortress | 21.36 | 22.38 | **−1.02** | 88k / 60k |
| horns | 18.73 | 18.78 | −0.06 | 104k / 68k |
| leaves | 17.56 | 17.56 | 0.00 | 256k / 163k |
| orchids | 16.01 | 16.03 | −0.01 | 139k / 88k |
| room | 21.76 | 22.31 | **−0.55** | 61k / 40k |
| trex | 22.22 | 22.24 | −0.03 | 93k / 61k |
| mean | 20.12 | 20.36 | −0.24 | ×1.5 |

- Five of the eight scenes are within noise (|Δ| ≤ 0.07). Flower is −0.30.
- The mean loss comes from **fortress and room**. In both, the deficit sits on the test views farthest from the training views:
  - fortress: the last two views are 19.76 vs 21.63 and 20.53 vs 23.45;
  - room: the last view is 23.30 vs 24.77.
- DWTGS reaches a lower final training photometric loss in every scene (for example fortress 0.0152 vs 0.0188) with 1.5× as many Gaussians. That is a better fit to the 3 training views with worse extrapolation.

## Diagnostics: train views only for decisions, 10k steps, labelled, `evidence/gap1/`
| run | fortress | room |
|---|---|---|
| DropGaussian s0 (bench) / s1 | 22.38 / 22.61 | 22.31 / 22.15 |
| DWTGS s0 (bench) / s1 | 21.36 / 21.74 | 21.76 / 21.61 |
| DWTGS-LF only (no HF), s0 | **22.90** | 22.00 |

- **Seed noise does not explain the gap.** The DWTGS − DG gap is −1.02 / −0.87 on fortress (s0 / s1) and −0.55 / −0.54 on room.
- **The HF term causes most of the loss.** On fortress LF-only is +0.52 over DropGaussian and the full method is −1.0, so HF costs ~1.5 dB. On room, HF costs ~0.25–0.4 dB and LF ~−0.2 to −0.3 dB.
- In the paper's Table III, the HF term adds +0.08 dB (20.70 → 20.78). Ours is clearly negative on these two scenes, which points at our interpretation of the HF term.
- Caveat: these are 2 scenes and one LF seed, and this is a diagnosis, not a tuning result. Test PSNR is used only to localise the failure; no value was chosen on it.

## Loss magnitudes (8 scenes, bench)
| term | step 1k | step 5k |
|---|---|---|
| photometric | 0.03–0.15 | 0.01–0.03 |
| LF | 0.014–0.050 (~0.3–0.5× photometric) | 0.005–0.013 |
| HF | 0.0014–0.006 | 0.0017–0.013 (comparable to LF late) |

All terms are active as scheduled: HF on steps with step%10 == 0 up to 5000, and 0 afterwards.

## Remaining interpretations (are they paper-grounded?)
1. **K for the novel views.** §V-A says "rendering novel views every 10 iterations", and that is the K of Fig. 2 note 3. It is fixed by the paper, so there is nothing to change.
2. **HF weight.** Eq. 6 has no weight, and the commented total (main.tex) is L_LF + L_HF, both unweighted. Weight 1 is the literal reading. Any other value would be tuning against test results, with no paper basis.
3. **Dropout during the novel render.** The paper is silent. The official `render()` defaults to `is_train=False`. With dropout, the novel render would see a random 0–10% of Gaussians removed by step 5000 (r ≤ 0.1). The effect on HH sparsity is small and its sign is unknown, and I expect far less than 0.5 dB. It is not grounded.
4. **λ split between levels 1 and 2.** Eq. 5 uses one λ_LL for all levels. Our averaging form gives every level equal per-pixel weight. Fixed by the paper.
5. **Interpolation details** (pair choice, t ~ U(0,1), and whether interpolation also runs beyond the pair). Unstated. Our reading (random pair, t ∈ (0,1)) never regularises the extrapolated region where fortress and room lose. A different reading cannot be justified from the text.
6. **Haar normalisation.** Averaging follows the paper text. Orthonormal was worse at 2k (18.04 vs 18.64 on horns) and would make HF weigh twice as much.

## Conclusion and recommendation
- **Classification: method gap.** About −0.6 dB comes from the HF term's effect on fortress and room under our interpretation.
- **No paper-grounded change is likely to close ≥0.5 dB.**
  - Every remaining knob is either fixed by the text (K, λ, levels, HF weight) or unstated with no grounded alternative (interpolation details, dropout on the novel render).
  - Changing the HF weight or dropping HF would be post-hoc tuning.
- **No iteration 1.** Report the outcome as not-matched.
- **Optional labelled diagnostic** (not a fix): run `dwtgs-lf` on all 8 scenes. It would show whether LF alone reproduces the paper's +0.31 dB over DropGaussian (Table III: 20.70 vs 20.39), which would localise the fidelity gap to the unspecified HF view sampling.
