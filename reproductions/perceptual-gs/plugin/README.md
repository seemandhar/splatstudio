# splatstudio-method-perceptual-gs

Native SplatStudio port of **Perceptual-GS: Scene-adaptive Perceptual Densification for Gaussian
Splatting** (Zhou & Ni, ICML 2025, [arXiv 2506.12400](https://arxiv.org/abs/2506.12400)), following
the official code [eezkni/Perceptual-GS](https://github.com/eezkni/Perceptual-GS) (built on Inria
3DGS; research/non-commercial licence, see `ss_perceptual_gs/csrc/LICENSE-3DGS.md`). The CUDA
statistics kernels (`csrc/ms_raster.cu`, `csrc/simple_knn.cu`) are vendored from the SPLATIFY
Mini-Splatting port, which adapts the `diff-gaussian-rasterization_ms` rasterizer that Perceptual-GS
also ships. Paper fidelity is unverified.

## Mechanism → code
| contribution | code |
|---|---|
| C1 sensitivity maps (Eqs. 5–7) | `core.sensitivity_map`, `sidecar.py` (`ss-pgs-prep`) |
| C2 dual-branch rendering + BCE (Eqs. 8–10) | `loss.py` (`pgs-photometric` = 0.9·photo, `pgs-bce` = 0.1·BCE until 15k) |
| C3 sensitivity-guided densification (Eqs. 11–16) | `density.PerceptualDensity.masks/_weights/_decide` |
| C4 scene-adaptive depth reinit (Eqs. 17–18) | `density._decide/_depth_reinit/_reinit` |
| C5 opacity decline (Eqs. 19–20) | `core.opacity_decline`, `density._densify` |

Per-Gaussian field `sensitivity` (logit, lr 0.05, init σ=0.1). All hyper-parameters are in the
component configs with the paper/official values (Table 8, `arguments/__init__.py`).

## Data prerequisite
The GT sensitivity maps are computed from the **full-resolution** images and PIL-resized to the
training size, as in the official `preprocess.py` + loader:
```bash
tools/ssenv ss-pgs-prep --data ~/datasets/mipnerf360/pilr/bicycle --raw ~/datasets/mipnerf360/360_v2/bicycle
```
writes `/bigdata/sejain/perceptual-gs-sens/mipnerf360/pilr/bicycle/`. Training finds it through
`--data` (or `PGS_DATA`), frames are matched by viewmat.

## Run
```bash
tools/ssenv ss-verify perceptual-gs --device cuda
tools/ssenv ss-train perceptual-gs --data ~/datasets/mipnerf360/pilr/bicycle --output <run> \
  --steps 30000 --recipe 3dgs --num-gaussians 10000000 --max-gaussians 12000000
tools/ssenv ss-eval --run <run> --split test
```
The official protocol is `-r 4` (outdoor) / `-r 2` (indoor) PIL resizing = dataset `mipnerf360-pilr`.

## Fidelity gaps
1. RGB branch over a white background (runtime); the sensitivity branch uses black as official.
2. Runtime steps the optimizer before density control (official skips the Adam step on densify
   iterations).
3. Statistics kernels use gsplat numerics (0.999 alpha cap, +0.5 px centres) instead of 0.99.
4. Views sampled with replacement (official shuffled stack); depth reinit colours come from a
   cache of rendered training targets and fails loudly if a view was never rendered.
5. SH degree increments one step later (runtime); SSIM valid convolution / `lpips` package.
6. OD output clamped to [1e-7, 1-1e-7] before the logit (official would produce ±inf).
7. Gamma correction uses the installed Pillow's `Image.point` rounding.
