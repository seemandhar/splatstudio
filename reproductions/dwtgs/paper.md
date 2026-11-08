# DWTGS — implementation spec

arXiv 2507.15690 (v3, 8 Oct 2025), Nguyen, Li, Le, Nguyen (UCSD). Sources: `source/latex/main.tex`
(4-page IEEE paper, no supplementary), `source/paper.pdf`.

**Official code: none found.** The paper gives no code link. A GitHub repository search for "DWTGS"
(2026-11-07) returned 0 results. A web search turned up only the unrelated LGDWT-GS
(`Advanced-Vision-and-Learning-Lab/sparse-view-3dgs-pack`), which is a different method. This port
is therefore written from the paper. The base method's numerics come from the DropGaussian
release (`reproductions/dropgaussian/source/code`, read only).

## 1. Method summary
- The base method is DropGaussian: 3DGS with scheduled random Gaussian dropout and 1/(1−r)
  opacity compensation, trained for 10k iterations (§V-A, "Our implementations are built on
  DropGaussian").
- **C1, LF loss (§IV-B, Eq. 5).** An L1 loss between the render's and the ground truth's LL subbands
  at every Haar-DWT level n = 1..N, with N = 2 and λ_LL = 0.5. It is computed on the training view,
  and LL is the 2×2 block mean (see §7.1).
- **C2, HF sparsity loss (§IV-B, Eq. 6).** An L1 loss pushing the level-1 HH subband of a render
  from a *novel* (unobserved) view towards 0. It is self-supervised, so no ground truth is used.
  It runs every 10 iterations and stops after 5k. Novel views are spherically interpolated between
  training views (Fig. 2, note 3).
- The DropGaussian loss L1 + 0.2·D-SSIM and DropGaussian's density control are unchanged.
- The final loss, from the commented line in the LaTeX, is L = L_3DGS(DropGaussian) + L_DWTGS_LF + L_DWTGS_HF.

## 2. Equations (verbatim from the LaTeX)
- Eq. 2 (DWT): `X_LL = L0 X L1, X_LH = H0 X L1, X_HL = L0 X H1, X_HH = H0 X H1`. The filters are
  Haar (§V-A, "we used the Haar wavelet"). Multi-level means repeating on X_LL.
- Eq. 5: `\mathcal{L}_{DWTGS\_LF} = \sum_{n=1}^{N}\lambda_{LL}\mathcal{L}_1(\mathbf{X}^n_{LL}, \mathbf{X}_{LL}^{n\_{gt}})`
- Eq. 6: `\mathcal{L}_{DWTGS\_HF} = \mathcal{L}_{1}( \mathbf{X}^1_{HH}, \mathbf{0})`. Here X is a novel-view
  render. Eq. 6 has no weight, so the weight is 1.
- Base (DropGaussian Eq. 7 / official train.py): `L = L1 + 0.2 (1 - SSIM)`.

## 3. Hyper-parameters
| name | value | source |
|---|---|---|
| wavelet | Haar, averaging normalisation (LL = block mean) | §V-A; §IV-B and Figs. 2–3 (see §7.1) |
| novel views | spherical interpolation between training views | Fig. 2, note 3 |
| N (LL levels) | 2 | §V-A, Table III |
| λ_LL | 0.5 | §V-A |
| HF weight | 1 (none written) | Eq. 6 |
| HF level | 1 (HH only) | §IV-B |
| HF novel-view interval | every 10 iterations | §V-A |
| HF stop | after 5000 iterations | §V-A |
| HF start | not stated → iteration 1 | assumption |
| iterations | 10000 | DropGaussian release default (DWTGS does not state it) |
| DropGaussian γ, τ, resets, etc. | as in `reproductions/dropgaussian/paper.md` | DropGaussian release |

## 4. Schedules
- The LF loss is active at every step.
- The HF loss is active when `step % 10 == 0` and `step <= 5000`.
- The DropGaussian drop rate is r_t = 0.2·t/10000.
- Densification runs from 500 to 10000, every 100 steps, with opacity resets at 3000, 6000 and 9000.

## 5. Builds on
DropGaussian (official release, built on 3DGS/FSGS code). Everything except the two added losses is kept.

## 6. Protocol
- **LLFF, 3 training views.** Uses the FSGS/DropGaussian split and the official `-r 8` resolution
  (504×378). Test views are every 8th image, and the average is over all 8 scenes.
- **Mip-NeRF 360, 12 views.** Uses DropGaussian's 12-view protocol, downsampled ×8.
- **Blender, 8 views.**
- Metrics are PSNR, SSIM and LPIPS. The authors re-ran DropGaussian and FreGS; the 3DGS row is copied
  from earlier papers. Only averages are reported.

## 7. Ambiguities (and the choice made)
1. **Haar normalization** (fixed after review 1). We use the averaging form, k = 4: LL = 2×2 block
   mean, so LL_n stays in the image's value range. The evidence:
   - §IV-B says LL "is in the same space as the original image, only a coarser version", and one
     shared λ_LL across levels is only "easy to control" if every level has the same scale.
   - Figs. 2–3 show X¹_LL and X²_LL at image brightness; only HH is marked "contrast enhanced".
   - The paper names no library.

   The orthonormal form (k = 2, pywt/pytorch_wavelets) is kept only as the labelled option
   `haar_norm: orthonormal`. It would give about a 3× effective LF weight (level n scaled by 2^n)
   and a 2× HF weight.
2. **Odd sizes.** At 504×378, level 1 gives 252×189 and level 2 needs padding. We zero-pad at the
   end, which matches pytorch_wavelets `mode='zero'`.
3. **Novel-view poses** (corrected after review 1; the paper is not silent here). Fig. 2, note 3,
   says: "Novel view sampling is performed using spherical interpolation between training views,
   only at every K iterations". K = 10 per §V-A. The interpolation details are not stated, so we
   take the simplest literal reading:
   - pick a random unordered pair of distinct training cameras;
   - draw t ~ U(0,1);
   - slerp the camera-to-world rotations and lerp the camera centres and intrinsics.

   FSGS's `generate_random_poses_llff` is kept only as the labelled option `novel_views: fsgs_llff`.
4. **Dropout on the novel-view render.** Not stated. The official `render()` defaults to
   `is_train=False`, so the novel view is rendered with all Gaussians, black background and a clamp.
5. **HF weight and reduction.** No weight is written, so we use 1. Both L1 terms use a mean
   reduction, as in 3DGS `l1_loss`.
6. **The LF loss is computed on the dropout training render**, the same render the main loss uses.
7. **Iteration count** is not stated. We use DropGaussian's 10k.
