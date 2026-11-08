# 3DGabSplat (arXiv 2508.05343) — native port report

**Outcome: not-matched.** Port `gabsplat`, written from the paper alone (no official code exists). Custom CUDA Gabor rasteriser, with forward and backward validated against a float64 dense reference.

| scene (30k) | ours | paper | Δ | our gain over our gsplat baseline | paper gain over its 3DGS* |
|---|---|---|---|---|---|
| garden | 27.07 | 27.66 | −0.59 | +0.18 (26.89) | +0.24 (27.43) |
| room | 31.51 | 32.27 | −0.76 | −0.01 (31.52) | +0.82 (31.46) |

- **Garden:** the method gain roughly matches the paper's; the absolute gap is our runtime offset (−0.52 dB on the baseline).
- **Room:** the paper's gain is missing.
- Run cost: 4.06M / 1.43M Gaussians; 671 / 221 min on shared GPUs.

**Why** (gap-analysis-1.md): the learned Gabor weights collapse (median 0.0025). Zeroing them costs only 0.11 dB on room train views, and the two kernels per primitive stay ~90% identical, so F=2 ≈ F=1. That comes from the paper's literal identical init plus its resets. No paper-grounded fix exists:
- Supp. B has no indoor-specific values.
- The main-text/supp. lr swap is not resolved by the text.
- The opacity settings already follow supp. B.

## Deviations / fidelity gaps
1. No official code; the projection and integration were re-derived. The renderer's formula equals paper Eq. 9-10 exactly (probe).
2. Identical init of the two kernels, as the paper literally states. This leaves symmetry that is broken only by float noise.
3. Main-text vs supplementary conflicts: supp. B used (init f 0.05; lr f 0.02 / w 0.01). The main text says 0.001 and 0.01 / 0.02.
4. The 2D frequency uses the conic dilated by 0.3 px²; the paper uses the undilated one.
5. Σω is not bounded, so the modulation m can go negative. Those contributions are dropped by the 1/255 alpha gate, as in the paper's 3DGS-based rasteriser.
6. The depth-axis amplitude factor (C1·C2 in the supp. derivation) is dropped, as in paper Eq. 12.
7. Runtime protocol differs from Inria 3DGS: white background, random view sampling, gsplat backward gate.

Evidence: evidence/verify.json, evidence/test.json, evidence/diag30k-{room,garden}.json, reviews/review-1.md.
