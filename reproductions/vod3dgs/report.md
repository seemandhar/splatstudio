# VoD-3DGS[S] reproduction report (plan id 10, arXiv 2501.17978)

**Outcome: not-matched** (−0.52 dB vs paper; tolerance ±0.5). Most of the gap is a baseline offset.

## Result (T&T complete average, 30k steps, llffhold8; runs/vod3dgs-tandt)
| scene | vod3dgs-s | Gaussians | 
|---|---|---|
| truck | 25.40 | 2.08M |
| train | 21.90 | 1.34M |
| **mean** | **23.65** (SSIM 0.856, LPIPS 0.198) | |

| | ours | paper |
|---|---|---|
| VoD-3DGS[S] | 23.65 | 24.17 |
| 3DGS baseline | 23.36 (gsplat `baseline3dgs-tandt-30000`) | 23.75 (authors' 3DGS† rerun) |
| gain | **+0.29** | +0.42 |

The −0.52 dB splits into about −0.39 dB baseline offset (runtime vs the authors' Inria rerun) and
about −0.13 dB method gain. Our gain is 70% of the paper's. Deep Blending was not benchmarked
(the paper's [S] gain there is −0.02). Smoke test (truck, 2k steps): 22.47 vs baseline 22.14.

## Gap analysis (training-side evidence only, no sweeps)
- The mechanism is active. At 30k, median Ŝ eigenvalues are about −1.5…−2.1 (min) and +2.3…+2.7 (max),
  and 95–96% of Gaussians have an eigenvalue span > 1 (55–66% > 4). L_vc ends at 4e-4 (truck) and 1.2e-3 (train).
- **γ reset kept alongside the Ŝ reset.** §4.3 says Ŝ is reset "to facilitate a similar approach" to 3DGS's reset, and
  §5.1 says "we do not modify any existing parameters". Keeping the 3DGS γ reset is the reading the paper supports;
  removing it would be a guess, not a correction.
- **Ŝ Adam moments zeroed and means detached in L_vc.** The paper specifies neither. L_vc is about 1e-3 of the
  photometric loss, so its gradient path to the means cannot plausibly move PSNR by 0.3 dB. Zeroing
  the moments affects only the 4 reset steps.
- **SfM covisibility instead of LightGlue pairing.** §5.1 itself says the matches "could be collected during the SfM
  computation". The paper's ablation (Table 4) measures keypoint vs uniform sampling at +0.01 dB, so the
  pairing source is not a 0.3 dB lever.
- No paper-grounded change is likely to recover ≥0.3 dB, so I did not run an iteration 1.

## Deviations from the paper (complete list)
1. No official code exists. The port is written from the paper and runs on the gsplat rasterizer and SplatStudio runtime
   (white background, random view sampling with replacement, gsplat's gradient drop for α>0.999).
2. Pairs for L_vc come from SfM covisibility (shared triangulated points) instead of LightGlue matches.
3. Training views with no covisibility would fall back to uniform sampling, and that count is logged. No such
   views exist in T&T.
4. γ is also reset. Ŝ Adam moments are zeroed at reset. Means are detached in L_vc. Ŝ starts at 0.
5. The Eq. 10 max is taken over all training cameras, with no visibility test.
6. Only averages are reported, so we compare the complete T&T average.

## Files
`paper.md`, `paper_results.json`, `plan.md`, `plugin/` (`vod3dgs-s`, `vod3dgs-l`), `evidence/` (verify, smoke,
test.json), `bench_request.json`.
