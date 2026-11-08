# EFA-GS — native SplatStudio reproduction report

**Paper**: Low-Frequency First: Eliminating Floating Artifacts in 3D Gaussian Splatting (Wang, Zhou,
Li, Quan, Qin; Cyberworlds 2025; [arXiv 2508.02493](https://arxiv.org/abs/2508.02493)). Plan id 128.
**Official code**: <https://github.com/jcwang-gh/EFA-GS> at commit `57f330f`. This port follows the
`Mip-Splatting/` variant, EFA-GS(Mip), which is the paper's default and the variant behind every
per-scene table. The code is under the Inria license (non-commercial research use) and is cited.
**Methods**: `efa-gs` and the control `efa-gs-base` (plugin `plugin/`, module `ss_efa_gs`). The control
is the same code path with LFCF off. No custom CUDA.
**Outcome**: **matched**, single seed:

| scene | ours | paper | gap |
|---|---:|---:|---:|
| garden | 27.95 | 28.02 | −0.07 dB |
| bonsai | 32.19 | 32.56 | −0.37 dB |

The small EFA − Mip-Splatting differences that the paper reports are **not resolved** at one seed
(§3.2). A matched PSNR is not proof of fidelity; deviations are listed in §4.

## 1. What was implemented

| contribution | paper / code | implementation |
|---|---|---|
| Mip-Splatting base | official EFA repo `Mip-Splatting/` | 3D filter √0.2·d/f, with filter-aware opacity reset, and the 2D Mip filter (gsplat antialiased mode, eps2d 0.1). The filter functions are vendored from `reproductions/mip-splatting`. On top of that, **GOF absolute-gradient densification**: clone or split when g ≥ 2e-4 ∨ g_abs ≥ quantile(g_abs, 1 − ratio). g_abs = \|absgrad_x\|·W/2 + \|absgrad_y\|·H/2, which equals the official rasterizer's grad.z. The paper-era Mip-Splatting plugin (commit 1ecf7d5) lacks this densification, so the EFA base was written fresh. |
| C1 LFCF (Alg. 1) | `densify_lff` | Runs on every 2nd densification (600, 800, …, 14800). The gradients used are the 200-step LFCF accumulators. A selected Gaussian is enlarged if it was not selected at the previous LFCF step or its gradient rose (tolerance 1e-5). Otherwise it is shrunk and possibly split. A split adds one child with the shrunk scales and keeps the parent. Scales are edited in place and the Adam moments are kept. |
| C2 depth strategy | §4.3 + code | c = θ·c_min + (1−θ)·c_max, where θ = log-minmax of (min visible depth / max focal), c_max = 1.5 and c_min = 1. The code's label order is followed; the paper's formula has the labels swapped. |
| C3 scale strategy | code diffscale | Enlarge: Δlog s = log c·(1, −1/3, −2/3) on axes from smallest to largest (volume preserving). Shrink: −½(log c + ln 2)·(1, ½, 0) on axes from largest to smallest. |
| C4 strategies 1–3 | App. C + train.py | LFCF interval r = 2. Linear decay of log c to 0 at step 15k; this is the code's direction, the paper's x^n formula would grow instead. Split probability θ(1 − lb) + lb, with lb = 1 − (it − 500)/14500. |
| unchanged | — | L1 + 0.2 D-SSIM, 3DGS learning rates, SH schedule, `--recipe 3dgs` initialisation. |

Probes: 10 declared checks pass for both methods (`evidence/verify-2.json`, `evidence/verify-base.json`).
They include an independent loop of Alg. 1, the strategy formulas, the schedule boundaries, the GOF
selection, and a GPU drive through LFCF steps 600 and 800. Unit tests: 15 pass.

## 2. Protocol
- Data: Mip-NeRF 360 `mipnerf360-pilr`. Full-resolution images are PIL-resized, `-r 4` outdoor and
  `-r 2` indoor, as in the official `scripts/run_mipnerf360.py`.
- Split: llffhold8.
- Training: 30k steps, `--recipe 3dgs`, black background (renderer config, as official).
- Metrics: LPIPS-VGG.
- Scenes: garden (outdoor) and bonsai (indoor). Bonsai has the paper's largest EFA gain and is the
  scene used in its ablation and speed tables.
- Paper numbers: Appendix table `resofmippsnr/ssim/lpips` (all 9 scenes reported; the EFA row is
  EFA-GS(Mip)).
- Runs: `runs/efa-gs-pilr/{efa-gs,efa-gs-base}/`.

## 3. Results

### 3.1 Per scene (PSNR / SSIM / LPIPS)

| scene | run | PSNR | paper | Δ paper | SSIM ours / paper | LPIPS ours / paper | Gaussians (M) | train min* |
|---|---|---:|---:|---:|---|---|---:|---:|
| garden | **efa-gs** | **27.95** | 28.02 | **−0.07** | 0.882 / 0.88 | 0.107 / 0.09 | 3.94 | 114 |
| garden | efa-gs-base (control) | 27.90 | 28.04 (Mip row) | −0.14 | 0.883 / 0.88 | 0.104 / 0.09 | 5.99 | 169 |
| garden | gsplat baseline (white bg) | 27.35 | 27.28 (3DGS row) | +0.07 | 0.867 / 0.87 | 0.128 / 0.11 | — | — |
| bonsai | **efa-gs** | **32.19** | 32.56 | **−0.37** | 0.951 / 0.95 | 0.221 / 0.16 | 1.17 | 77 |
| bonsai | efa-gs-base (control) | 32.39 | 32.34 (Mip row) | +0.05 | 0.951 / 0.95 | 0.218 / 0.16 | 1.70 | 74 |
| bonsai | gsplat baseline (white bg) | 32.18 | 32.26 (3DGS row) | −0.08 | 0.948 / 0.95 | 0.234 / 0.17 | — | — |

- The gsplat baseline is `runs/baseline3dgs-mipnerf360-pilr-30000`. It renders a white background,
  so it is context only, not a matched control.
- \*Wall-clock times were measured on a heavily shared host and are not comparable.
- LPIPS is consistently above the paper's values. This holds for all of our runs and is the known
  runtime/LPIPS-implementation offset; the paper uses lpipsPyTorch.

### 3.2 Effect of LFCF (method − control)

| scene | ours: efa-gs − efa-gs-base | paper: EFA − Mip-splatting | ours: Gaussians ratio |
|---|---:|---:|---:|
| garden | +0.05 | −0.02 | 0.66× |
| bonsai | −0.20 | +0.22 | 0.69× |

- The differences are at the level of single-seed noise. The README reports ±0.05 dB across GPUs on
  the average, and per-scene noise is larger. They neither confirm nor refute the paper's small
  Mip-NeRF 360 effect (+0.02 dB on average).
- On bonsai the sign is opposite to the paper's: −0.20 here vs +0.22 there. Resolving this would
  need several seeds.
- Robust observation: LFCF changes the optimisation substantially. It ends with about 0.66–0.69× the
  Gaussians of the base, consistent with the paper's claim that it is cheaper (Tab. speed).
- The control matches the paper's Mip-Splatting row within 0.14 dB on both scenes. This supports
  the GOF-densification base as the correct one: the paper-era Mip plugin reached 27.43 on garden.

## 4. Deviations and fidelity gaps
1. **Runtime**: views are sampled at random with replacement (official: shuffled stack). Densification
   runs after the optimizer step; the official code runs it before, which drops that step's Adam
   update. SSIM uses a valid rather than same convolution. LPIPS comes from the `lpips` package.
2. **Rasterizer**: gsplat instead of the Inria/Mip CUDA rasterizer. gsplat drops gradients where
   α ≥ 0.999 (see TODO-runtime). Tile and culling details differ, so the gradient statistics that
   drive LFCF and GOF selection differ slightly.
3. **3D-filter / sampling-interval projection** uses K's cx, cy (official: W/2, H/2). Only the
   in-frustum test is affected, which is negligible for the centred PINHOLE cameras.
4. **RNG streams** differ (split offsets, split-probability draws). Single seed, seed 0.
5. **Control** `efa-gs-base` is our port of the EFA repo's base, not the authors' runs.
6. Scenes are 2 of the 9 reported. T&T (6 advanced scenes) and RWLQ are not available, and the
   noisy-initialization experiments were not run.
7. Paper formulas vs code: the c(i) labels and the annealing direction differ, and the code was
   followed. The 3DGS variant, EFA-GS(3DGS), was not ported; the paper has no per-scene table for it.

## 5. History
- **Review 1** (`reviews/review-1.md`): 1 major and 4 minor findings.
  - Major: the renderer composited a white background, while the official code uses black. Fixed by
    a renderer `background` config defaulting to [0,0,0].
  - Minor: probe wording in plan.md, an LR-delay gap that does not exist (official
    lr_delay_steps = 0), and the cx/cy note. All addressed.
  - Minor: densify-after-step is documented.
- Smoke tests: pilr garden, 2k steps, black background.

  | run | PSNR | Gaussians |
  |---|---:|---:|
  | efa-gs | 23.62 | 0.67M |
  | efa-gs-base | 23.59 | 1.21M |
  | gsplat (white) | 23.32 | — |

  The first white-background smoke runs are kept in `evidence/smoke-1` and `evidence/smoke-base`.
- During the fix iteration, two smoke folders from a crashed run were deleted. They were
  `evidence/smoke-2` and `evidence/smoke-base-2`, which crashed on a background-shape assert. They
  held no usable data and were re-created by the successful re-run.
- The GPU probe was made robust: it disables opacity pruning on its own probe instance, and for
  `efa-gs-base` it checks that no LFCF step runs.
