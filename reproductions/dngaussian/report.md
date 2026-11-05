# DNGaussian reproduction report

**Paper:** *DNGaussian: Optimizing Sparse-View 3D Gaussian Radiance Fields with Global-Local Depth Normalization* (CVPR 2024).

**Method:** `dngaussian`, plugin `plugin/`, frozen at `a7a65bbfabb9d3f8`. The control is `dngaussian-3dgs`, which matches the paper's 3DGS† row: the same hyperparameters and neural colour renderer, with no depth regularization.

**Outcome: matched.**
- The 8-scene mean PSNR on LLFF with 3 views is **18.93 against the paper's 19.12 (−0.19 dB)**. SSIM is 0.593 against 0.591.
- The control reproduces the paper's 3DGS† row: **16.52 against 16.46**.
- The gain from depth regularization is +2.41 dB for us and +2.66 dB in the paper.

## Results
Setup:
- 6000 steps on `llff-fewview3-r8-dng`: the FSGS/DNGaussian 3-view split at 1/8 resolution, with DPT depth priors for the training views.
- Test views are every 8th image.
- Runs are in `runs/dngaussian-llff-r8/`.

| method | fern | flower | fortress | horns | leaves | orchids | room | trex | **mean PSNR** | SSIM | LPIPS |
|---|---|---|---|---|---|---|---|---|---|---|---|
| dngaussian (ours) | 21.12 | 18.73 | 21.20 | 18.51 | 16.59 | 14.48 | 20.88 | 19.96 | **18.93** | 0.593 | 0.341 |
| paper DNGaussian | | | | | | | | | **19.12** | 0.591 | 0.294 |
| dngaussian-3dgs (ours) | 17.39 | 17.57 | 18.52 | 15.38 | 15.62 | 12.48 | 18.24 | 17.00 | **16.52** | 0.467 | 0.426 |
| paper 3DGS† | | | | | | | | | **16.46** | 0.440 | 0.401 |
| paper 3DGS (plain) | | | | | | | | | 15.52 | 0.408 | 0.405 |

The paper reports only dataset averages for LLFF, so the mean over the full 8 scenes is the right comparison here. That mean is a full-dataset average, not a subset.

LPIPS is higher than the paper's (0.341 against 0.294). The runtime uses the `lpips` package on float renders, while the official code uses lpipsPyTorch on 8-bit PNGs (gap 6). PSNR and SSIM agree.

## Fidelity gaps (kept; full list in plan.md)
1. **Rasterizer.** We use gsplat instead of the Inria variant. Visibility radius is 3.33σ instead of 3σ.
2. **Pass ordering.** The depth passes run after the photometric pass on the same camera. The official code runs the depth passes on the next camera, then the photometric pass. As a result, the hard depth pass of iteration 1 is lost.
3. **Optimizer step order.** The runtime takes the optimizer step before densification and pruning, which also means it takes an extra step at iteration 6000.
4. **View sampling.** Views are sampled at random with replacement instead of from a shuffled stack.
5. **Near-Gaussian cull at eval.** It is done per rendered view instead of over the union of all eval cameras.
6. **Metric implementations.** SSIM uses valid convolution, and LPIPS uses the `lpips` package on float renders.
7. **Empty selections.** Where the official code returns NaN, ours returns 0. Gradients are the same.
8. **Seeds.** One seed per scene. The official README warns that PSNR varies between runs.
9. **Depth model.** DPT comes from MiDaS master with timm 0.6.12. The weights are the same `dpt_hybrid_384` (sha256 `501f0c75…`), but it is not the authors' 2023 torch.hub snapshot.
10. **3DGS† control.** It keeps penalties, opacity-gated densification, cleaning and near pruning, and drops only the hard and soft depth passes.

## Review
`reviews/review-1.json` raised one major issue: no probe drove `after_backward` at the schedule boundaries. It was fixed with the `after-backward-schedule` probe before the benchmark. The 5 minor findings are recorded under the fidelity gaps.

## Reproduce
```bash
python3 tools/bench.py launch --exp dngaussian-llff-r8 --methods dngaussian dngaussian-3dgs --dataset llff-fewview3-r8-dng --steps 6000 --per-gpu 2
```
