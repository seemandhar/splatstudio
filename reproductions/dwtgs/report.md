# DWTGS reproduction report

- **Paper:** *DWTGS: Rethinking Frequency Regularization for Sparse-view 3D Gaussian Splatting*, arXiv 2507.15690 v3. Plan id 18.
- **Official code:** none. A GitHub search found no repository, so this port follows the paper.
- **Base:** the paper is built on DropGaussian, so the port reuses the in-repo DropGaussian port (`reproductions/dropgaussian`, matched).
- **Methods:** `dwtgs`, plus the ablation `dwtgs-lf`. The plugin is in `plugin/` (`ss_dwtgs`).

**Outcome: not matched.** On LLFF 3-view, the 8-scene mean is **20.12 / 0.698 / 0.232** vs the paper's **20.78 / 0.720 / 0.197**, a gap of **−0.66 dB**.

Our DropGaussian scores 20.36, against the authors' DropGaussian re-run of 20.39. The protocol is therefore matched, and the gap belongs to the DWTGS terms: we measure −0.24 dB relative to DropGaussian where the paper reports +0.39.

## Results (10k steps, `llff-fewview3-r8`, seed 0; the paper reports averages only)
| run | fern | flower | fortress | horns | leaves | orchids | room | trex | mean | SSIM | LPIPS |
|---|---|---|---|---|---|---|---|---|---|---|---|
| dwtgs (`runs/dwtgs-llff-r8`) | 22.80 | 20.52 | 21.36 | 18.73 | 17.56 | 16.01 | 21.76 | 22.22 | **20.12** | 0.698 | 0.232 |
| our DropGaussian (`runs/dropgaussian-i1-llff-r8`) | 22.73 | 20.82 | 22.38 | 18.78 | 17.56 | 16.03 | 22.31 | 22.24 | 20.36 | 0.705 | 0.232 |
| paper DWTGS | | | | | | | | | **20.78** | 0.720 | 0.197 |
| paper DropGaussian (authors' re-run) | | | | | | | | | 20.39 | 0.706 | 0.198 |

- DWTGS ends with 1.5× DropGaussian's Gaussian count in every scene.
- It is within noise of DropGaussian on 5 of 8 scenes.
- It loses on fortress (−1.0) and room (−0.55), concentrated on the extrapolated test views.

**Diagnostics** (`evidence/gap1`, 10k steps, labelled, not used to select anything):
- Seed 1 reproduces the losses: −0.87 dB on fortress and −0.54 dB on room vs DropGaussian.
- LF-only (no HF) is 22.90 on fortress (+0.52 over DropGaussian) and 22.00 on room. The novel-view HF term is therefore what costs fortress about 1.5 dB.

Details are in `gap-analysis-1.md`.

## History
- **i0, before benchmarking:** orthonormal Haar and FSGS random poses. The 2k horns smoke gave 18.04 dB and 152k Gaussians.
- **Review 1:** C1 made averaging Haar the default (§IV-B "same space as the original image"; Figs. 2–3). M1 switched to interpolated novel views (Fig. 2 note 3). M2 made the probes pin these paper forms.
- **After the fixes:** the 2k smoke gave 18.64 dB vs DropGaussian's 18.88 dB. The superseded evidence is kept under `evidence/*-i0-*`.
- **Gap analysis 1:** none of the remaining paper-grounded changes is likely to close ≥0.5 dB, so there is no iteration 1.

## Fidelity gaps (all kept)
1. **Novel-view sampling details are unstated.** The paper says only "spherical interpolation between training views" (Fig. 2 note 3). Our reading: a random distinct pair of training views, t ~ U(0,1), slerp of the rotations, lerp of the centres and K. This is the most likely cause of the gap, because the HF term hurts the extrapolated views.
2. **HF weight is 1.** Eq. 6 has no weight, so 1 is the literal reading.
3. **The novel view is rendered without dropout.** The paper is silent; this is the official `render()` default.
4. **Haar normalisation is averaging** (LL = block mean), chosen from the paper's text. The orthonormal form is available as a labelled option.
5. **The LF gradient enters the densification statistic**, as it would when the term is added to the same backward pass. This gives 1.5× the Gaussians. The paper reports no counts.
6. **Inherited DropGaussian gaps:**
   - views are sampled with replacement;
   - the Adam step runs before densification;
   - gsplat rasterizer with the straight-through alpha cap;
   - the `lpips` package for LPIPS;
   - one seed.
7. **Not benchmarked:** Mip-NeRF 360 12-view and Blender 8-view. LLFF is the cheapest complete average.

## Reproduce
```bash
python3 tools/bench.py launch --exp dwtgs-llff-r8 --methods dwtgs --dataset llff-fewview3-r8 --steps 10000
```
