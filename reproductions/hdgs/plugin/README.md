# splatstudio-method-hdgs

HDGS: Textured 2D Gaussian Splatting for Enhanced Scene Rendering (arXiv 2412.01823) as a native
SplatStudio plugin, built on the official code (github.com/TimSong412/HDGS: `main` b13a359 =
stage 1, `texture` 01c6fd4 = stage 2) with both official CUDA rasterizers vendored in
`ss_hdgs/csrc/` (Inria licence; only change: `SORT_WINDOW` made a build flag).

Methods: `hdgs` (Mip-NeRF 360 outdoor, tex_res 200) and `hdgs-indoor` (indoor, tex_res 500).
Train with `--steps 60000 --recipe 3dgs` (40000 stage 1 + 20000 stage 2).

## Mechanism → code
| paper | code |
|---|---|
| 2DGS surfels, per-ray k-buffer sort (k=16), tile-centre depth sort, frustum 5-sample G | `csrc/stage1` (official), `raster.py` |
| 2DGS losses L_d (λ 100, >3000), L_n (λ 0.01, >7000) | `losses.py` (official train.py / run_360.py) |
| 2DGS densification (grad 2e-4, every 150, 500–20000, cull 0.01, reset 3000) + xyz expon LR | `density.py` |
| Fisher pruning 10 % (Eq. U_i) at step 40000 | `fisher.py` (Hutchinson diag-F; code unreleased) |
| per-surfel texture (T s, cap 500, bilinear, r 4.5, DC init), SH at intersection direction | `csrc/texture` (official), `texture.py` |
| stage 2: geometry/opacity/DC frozen, SH rest after 1000 its, texture Adam 8e-3 StepLR(8000, 0.1), L_c only | `density.py`, `losses.py` |

The texture buffer is not a per-Gaussian field: it lives in `texture.STORE`, is saved through the
density `state` into `checkpoint.pt["density_state"]`, and a fresh `ss-eval --run <dir>` reloads
it (the renderer reads `--run` from argv; params carry the `hdgs_textured` flag so a textured run
can never be rendered untextured silently).

## Fidelity gaps
See `../report.md` / `../plan.md`: Hutchinson Fisher estimate and score sign convention; runtime
view sampling with replacement and valid-conv SSIM; SH-rest delay implemented as lr 0; stage 2 at
stage-1 resolution (official script; paper text says higher); clamp of zero-sized textures to 1.

## Commands
```
tools/ssenv pip install -e reproductions/hdgs/plugin
tools/ssenv ss-verify hdgs --device cuda
tools/ssenv ss-train hdgs --data ~/datasets/mipnerf360/native/garden --output runs/x --steps 60000 \
    --recipe 3dgs --num-gaussians 10000000 --max-gaussians 12000000
tools/ssenv ss-eval --run runs/x --split test
```
