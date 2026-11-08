# BBSplat (arXiv 2411.08508) — reproduction report

**Outcome: partial.** Mip-NeRF 360 bicycle matches the paper (−0.16 dB). Tanks&Temples truck does not
(−1.07 dB after a post-hoc iteration; −1.99 dB at iteration 0).
Plugin: `reproductions/bbsplat/plugin` (`ss_bbsplat`). The official CUDA rasterizer
(diff-bbsplat-rasterization @ 2dbc336) is vendored unmodified and JIT-built. Training logic is ported from
BBSplat @ cff8d6a.

## Results (test split, llffhold 8, 32000 steps, single seed)
| scene | method / iteration | billboards | ours PSNR | paper PSNR | gap | run |
|---|---|---|---|---|---|---|
| bicycle | `bbsplat` (iter 0) | 60K | 23.12 | 23.28 | −0.16 | runs/bbsplat-m360 |
| truck | `bbsplat-tnt` (iter 0) | 30K (20K SfM + 10K sky) | 22.20 | 24.19 | −1.99 | runs/bbsplat-tnt |
| truck | `bbsplat-tnt-sky2k` (iter 1, **post-hoc**) | 30K (28K SfM + 2K sky) | 23.12 | 24.19 | −1.07 | runs/bbsplat-tnt-sky2k |

Paper references come from the fixed-storage experiment: Tab. 1 and the Supp. per-scene table. The large-budget
per-scene table is commented out in the LaTeX source.
| scene | paper BBSplat | paper 3DGS-MCMC (budget-matched) | paper 3DGS (budget-matched) |
|---|---|---|---|
| bicycle | 23.28 | 23.87 | 23.04 |
| truck | 24.19 | 23.91 | 20.43 |
The paper's 3DGS rows are budget-constrained (80K / 160K Gaussians), so they are not comparable with our
unconstrained gsplat baseline. Absolute numbers are compared instead.

Smoke test (bicycle, 2000 steps): bbsplat 20.73 vs gsplat 21.14 PSNR, with 60K vs 571K primitives.

## Verification
- `ss-verify` passes 7/7 probes.
- Kernel forward and backward match a dense float64 torch reference: forward error ≤ 2e-6, gradient relative error ≤ 3e-6.
- Probes cover the texture composition (Eq. 6), the impact-weighted texture regulariser, the MCMC relocation/add
  (alpha-only rescaling), the sky/FPS init and the lr schedule.
- 7 unit tests pass. Review-1 approved the port (0 critical/major).

## Gap analysis (see gap-analysis-1.md)
- **Sky-sphere budget share (iter 0).** On truck, 21 % of the 30K billboards ended on the sky sphere (radius ≈ 212,
  inflated by far SfM outliers). The other billboards alone cover 99 % of test pixels.
- **Underfitting.** Truck train-view PSNR is 23.11 vs 22.20 on test: the model is capacity-limited.
- **Iteration 1 (post-hoc).** Supp. l.736 says "for scenarios when we use 20K points and less, we reduce the number of
  sphere points to 2K". We read "points" as the SfM points read (20K at iter 0). That gives a 2K sphere, and with
  cap = SfM + sky the split becomes 28K SfM + 2K sky. The same method also uses the official xyz lr after 30K: lr is
  zero only at step 30001. This variant was chosen after seeing the truck result, so it is labelled post-hoc. It
  gained +0.92 dB and runs in 24 min. Bicycle is unaffected: 50K SfM is above 20K under either reading.
- **Remaining truck gap (−1.07 dB): data/protocol, not closable.**
  - Ours: the 3DGS-distributed T&T truck (979×546, 3DGS COLMAP).
  - Authors: their own COLMAP reconstruction (`Truck_COLMAP_big`, convert.py, "~900 px", likely 960×540).
  - A different SfM also changes the FPS init and the sky radius.
  - No further paper-grounded fix was found.

## Deviations from the paper / official code
1. **Unpublished Tab. 1 budgets.** We used cap = Tab. 1 "Points" (M360 60K, T&T 30K), and SfM = cap − sky (as in the
   published scripts). For truck, the sky-sphere size is the ambiguous reading above.
2. **σ = 100.** This is the code default; the paper text says 500 (Supp. Tab. shows ≤0.07 dB sensitivity).
3. **No 2DGS depth-normal regularisation for M360/T&T.** This follows the published scripts; one sentence in the
   paper claims Tabs. 1/2 used it.
4. **Order effects.** Densification runs after the optimizer step, and the Adam step is not skipped on relocation
   iterations (the official code skips it as a side effect).
5. **Runtime-wide differences.**
   - Views are sampled with replacement.
   - SSIM uses a valid convolution.
   - The SH degree increases one step later.
   - Rendered images are not 8-bit quantised before metrics (8-bit texture quantisation at evaluation *is*
     emulated, as in the official render.py).
6. **±inf alpha logits** are stored as ±1e4. The sigmoid output and the (zero) gradient are identical.
7. **xyz lr after 30K.** Iteration-0 methods zero it for all steps > 30000; only `bbsplat-tnt-sky2k` uses the
   official behaviour. The effect is tiny.
8. **FPS** follows the COLMAP point order of our conversion, starting at index 0 (as pytorch3d does).
9. **Data.** T&T is the 3DGS-distributed truck, not the authors' COLMAP.
10. **Provenance.** Iteration-0 runs predate the iteration-1 source edit and can no longer be re-evaluated with
    ss-eval; their stored test.json results stand.
