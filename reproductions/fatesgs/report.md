# FatesGS reproduction report

**Paper:** *FatesGS: Fast and Accurate Sparse-View Surface Reconstruction Using Gaussian Splatting with Depth-Feature Consistency* (AAAI 2025, arXiv 2501.04628).

**Official code:** yulunwu0108/FatesGS @ `a721d318`. Its 2DGS CUDA rasterizer is vendored unchanged.

**Methods:**
- `fatesgs`: the full method, which adds three losses to 2DGS: depth rank (L_r), smoothness (L_s) and multi-view feature consistency (L_f).
- `fatesgs-2dgs`: the same pipeline without L_r, L_s and L_f. It corresponds to the paper's ablation row "none", not to its separately run 2DGS row.

The plugin is in `plugin/`.

**Metric:** DTU Chamfer distance (CD) in mm, lower is better. It is computed with the official TSDF mesh extraction, mask culling and DTUeval-python against the official DTU ground truth.

**Outcome: matched (after 1 iteration).**
- **Absolute CD matches.** FatesGS CD is within 0.10 mm of the paper on all 3 scans: **0.67 / 1.91 / 0.85 against 0.67 / 1.94 / 0.75**.
- **Control matches too.** Our control is close to the paper's 2DGS row: 1.20 / 2.99 / 2.00 against 1.26 / 2.95 / 1.93.
- **Relative claim reproduces.** FatesGS reduces CD against the control by 44 / 36 / 58 %; the paper reports 47 / 34 / 61 %.
- **What changed in iteration 1.** The previous run (0.85 / 2.17 / 1.05) was off because of a screen-size prune that the official code never applies (see below).

## Results
**Setup**
- DTU, 3 views (large overlap, views 23/24/33, the SparseNeuS/NeuSurf split).
- 15000 steps, the official setting.
- Scans 24, 37 and 106 (of the 15 in the paper).
- Each run directory has `dtu_cd.json` (accuracy, completeness, CD) and `dtu_eval.log`.

**Iteration 1** (`runs/fatesgs-i1-dtu/`, the reported result):

| scan | FatesGS CD ours / paper (Δ) | control CD ours / paper 2DGS row | reduction vs control: ours / paper | #G FatesGS / control |
|---|---|---|---|---|
| 24 | **0.67 / 0.67 (0.00)** | 1.20 / 1.26 | −44 % / −47 % | 198k / 165k |
| 37 | **1.91 / 1.94 (−0.03)** | 2.99 / 2.95 | −36 % / −34 % | 189k / 129k |
| 106 | **0.85 / 0.75 (+0.10)** | 2.00 / 1.93 | −58 % / −61 % | 95k / 54k |
| mean (3 scans) | 1.14 / 1.12 | 2.06 / 2.05 | −45 % / −45 % | |

- These are 3-scan subset means, not the paper's 15-scan averages (0.92 FatesGS, 1.69 2DGS).
- The control is the paper's "none" ablation (same pipeline without L_r/L_s/L_f). It is compared with the paper's separately run 2DGS row, and the two agree closely here.

**Iteration 0** (`runs/fatesgs-dtu/`, kept):

| scan | FatesGS CD | control CD |
|---|---|---|
| 24 | 0.85 | 1.65 |
| 37 | 2.17 | 3.44 |
| 106 | 1.05 | 3.08 |

## Iteration 1: the fix
- **Official behaviour.** In the official code, `densification_postfix` (called by clone and split) zeroes `max_radii2D` (`gaussian_model.py:344`) before `densify_and_prune` tests `max_radii2D > max_screen_size` (`:396`). The screen-size prune therefore never fires; only the world-size prune acts.
- **What the port did.** Our port applied a real screen-size prune. That removed large background and table surfels and degraded both arms.
- **How it was found.** The same bug was found first in the SparseGS port. There, a diagnosis on a training run showed test PSNR collapsing after step 3000.
- **The fix.** The screen-size prune is removed, keeping the world-size prune. Nothing was tuned.

## Fidelity gaps (kept)
1. **View sampling:** with replacement in our runtime; the official code uses a shuffled stack.
2. **Densify order:** densification runs after the optimizer step. The official code densifies before it, which skips the Adam step on densify iterations and the opacity step on reset iterations.
3. **Rotation init:** normalised randn here; the official code uses `torch.rand`.
4. **"Visibility" term:** the official code uses `torch.unique(dim=0)`, so this term is not a true z-buffer; it picks effectively arbitrary points. We reproduce it as written. A `visibility: zbuffer` variant follows the paper's Eq. 15.
5. **Code over paper:** where code and paper disagree, the code was followed: means instead of sums, a 1−cos < 0.5 mask, 777×581 images, and the official feature normalisation.
6. **Control ≠ paper's 2DGS row:** the paper's per-scan 2DGS row is a separately run 2DGS. The "none" ablation row is published only for the little-overlap set.
7. **Not reproduced:** the little-overlap set (no released Marigold depth), the NVS table (pixelNeRF 25-view images not released), BlendedMVS, and the other 12 scans.

## Review
`reviews/review-1.json` approved the plugin with 7 minors. The reviewer verified:
- the vendored CUDA is identical to the official code (`diff -r`);
- Vis-MVSNet features are bit-identical;
- L_f matches within 1.7e-5, and L_r, L_s and SSIM match exactly;
- TSDF, culling and CD parameters are identical;
- coordinate frames and scale matrices are consistent.

## Reproduce
```bash
python3 tools/bench.py launch --exp fatesgs-dtu --methods fatesgs fatesgs-2dgs --dataset dtu-fatesgs-23_24_33 --scenes scan24 scan37 scan106 --steps 15000
# then per run:
PYTHONPATH=$HOME/models/dtueval-pylib tools/ssenv fatesgs-dtu-eval --run <run> --data <data> --dtu ~/datasets/dtu_official/gt
```
