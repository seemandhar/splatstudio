# MSAA-GS gap analysis 1 (after benchmark `runs/msaa-gs-mipnerf360`, 30k)

## Gap table (PSNR, test)
| scene | ours msaa-gs | ours gsplat | paper Ours | paper 3DGS (Table 2, retrained) | abs gap | baseline offset | delta gap |
|---|---|---|---|---|---|---|---|
| garden | 26.90 | 26.89 | 27.41 | 27.28 | −0.51 | −0.39 | **−0.12** |
| kitchen | 30.66 | 31.34 | 31.45 | 31.32 | −0.79 | +0.02 | **−0.81** |

Classification: garden = protocol offset + noise (method delta +0.01 vs paper +0.13).
Kitchen = method gap (delta −0.81), see below.

## Evidence
1. **One kitchen test view is a floater outlier.** Per-view deltas vs gsplat: one view
   (DSCF0672) −10.87 dB (21.30 vs 32.17), the other 34 views mean −0.38 dB. Removing the
   Gaussians whose screen radius exceeds 25% of the image (100 of 1.07M) brings that view to
   29.78 dB — a few large semi-transparent Gaussians near this (test-only) viewpoint darken the
   whole frame (render mean RGB 0.53/0.42/0.35 vs target 0.60/0.49/0.42;
   `evidence/gap1/view0672/`). Contributes ≈ −0.30 dB of the kitchen mean. Single seed;
   occurrence is stochastic, but fewer/larger Gaussians make it more likely.
2. **Gaussian count.** 30k: kitchen 1.07M vs gsplat 1.42M (−24%), garden 4.14M vs 4.76M (−13%).
   The paper states it "employed a similar number of Gaussians as in 3DGS" (Sec. 4.2).
3. **Component ablation, kitchen 7k, train views (every 4th, never test)** — `evidence/gap1/*.trainviews.json`:

   | variant | Gaussians | train PSNR |
   |---|---|---|
   | gsplat | 850k | 29.26 |
   | MSAA only | 844k | **29.37 (+0.11)** |
   | AWS+GDC only | 651k | 28.58 (−0.68) |
   | full msaa-gs | 649k | 28.62 (−0.64) |

   MSAA (4x rotated grid, eps2d 0.3 per sample) is not the problem: it helps and leaves the
   count unchanged. The loss constraints cost −0.65 dB and −24% Gaussians.
4. **Mechanism.** Densification statistic (gsplat |∂L/∂means2d| in NDC, train views of the 7k
   gsplat checkpoint; `evidence/gap1/densify_stat.txt`): fraction of Gaussians above the 3DGS
   threshold 2e-4 = 1.70% gsplat, 1.60% MSAA-only, **1.08% AWS+GDC**, 1.02% full. On the 30k
   kitchen model the AWS weights average 0.52 and Σw|e|/Σ|e| = 0.54, i.e. with the α = 0.5
   floor (not from the paper; taken from the old Nerfstudio port) the weighted L1 is about half
   of 3DGS's L1. The photometric gradient, and with it the densification statistic, shrinks
   while the 3DGS thresholds stay fixed — fewer Gaussians, lower fidelity.

## Hypotheses (ranked)
1. **AWS weight scale (α = 0.5, unnormalised) halves the L1 gradient → under-densification.**
   For: items 2–4. Against: none observed. Expected effect: most of the −0.4 dB broad kitchen
   loss, plus a lower chance of floater outliers. Paper basis: Eq. 7 gives no α; Sec. 4.1 keeps
   the 3DGS hyper-parameters, and Sec. 4.2 says the Gaussian count matches 3DGS, which our
   implementation contradicts.
2. Sub-sample offsets / per-sample eps2d dilation: ruled out by the MSAA-only ablation (+0.11).
3. GDC weight: it adds gradient and does not remove it, so it cannot explain the lower
   densification. Not isolated separately.
4. The floater view: stochastic. Not a code fix.

## Recommended fix (iteration 1, post-hoc, labelled)
`msaa-gs-it1`: divide the AWS weight map by its image mean (`normalize_mean: true`). This keeps
Eq. 7's relative emphasis on high-error pixels and restores the 3DGS L1 scale that the 3DGS
densification thresholds assume. It is not in the paper; it is justified by the paper's
"3DGS hyper-parameters" and "similar number of Gaussians" statements. All other settings
unchanged. The α value is not tuned.
Validation (7k, train views only): see the section below.

### Iteration-1 validation (7k, train views only)
| scene | variant | Gaussians | train PSNR |
|---|---|---|---|
| kitchen | gsplat | 850k | 29.26 |
| kitchen | msaa-gs (iter 0) | 649k | 28.62 |
| kitchen | **msaa-gs-it1** | 900k | **29.42** (+0.80 vs iter 0, +0.16 vs gsplat) |
| garden | gsplat | 2.62M | 26.57 |
| garden | msaa-gs (iter 0) | 2.26M | 26.19 |
| garden | **msaa-gs-it1** | 2.85M | **26.62** (+0.44 vs iter 0, +0.05 vs gsplat) |

Normalisation restores the Gaussian count (≈ +6% / +8% vs gsplat) and the train-view fidelity.
The improvement over gsplat now has the sign and size of the paper's per-scene deltas (+0.13).
Checks: `ss-inspect` issues [] for both methods; 7 unit tests pass; ruff is clean; `ss-verify msaa-gs-it1`
and `msaa-gs` report declared_checks_passed (`evidence/verify-it1.json`, `evidence/verify.json`;
the iteration-0 verify file was renamed `verify-iter0.json`).
**Recommendation: benchmark `msaa-gs-it1` on garden + kitchen at 30k and label it post-hoc.**
Note: the plugin source changed, so the iteration-0 runs can no longer be re-evaluated with
ss-eval (manifest check). Their test.json files already exist and are kept.
