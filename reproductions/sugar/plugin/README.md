# splatstudio-method-sugar

SuGaR (Guédon & Lepetit, CVPR 2024, arXiv 2311.12775), a native SplatStudio port of the
official `sdf` pipeline (https://github.com/Anttwo/SuGaR, `train.py -r sdf --high_poly True
--refinement_time long`). Paper fidelity is unverified.

One `ss-train sugar --steps 30000` run covers:
| steps | stage | official source |
|---|---|---|
| 1–7000 | vanilla 3DGS (gsplat DefaultStrategy, refine_stop 7000, opacity reset 3000/6000) | gaussian_splatting/train.py --iterations 7000 |
| 7001–15000 | coarse SuGaR: entropy 0.1 (7000<it<9000), prune opacity<0.5 at 9000, SDF R 0.2 + normal R_Norm 0.2 after 9000, 16-NN updated every 500 | sugar_trainers/coarse_sdf.py |
| after 15000 | λ=0.3 level-set points (10M), fg/bg Poisson depth 10, 10 % density trim, 1M-triangle decimation each, vertex snapping | sugar_extractors/coarse_mesh.py |
| 15001–30000 | one bound Gaussian per triangle (barycentre, normal axis thickness E·1e-6, 2 scales, complex 2D rotation), learnable vertices (LR 1.6e-4·E/100, exp decay over 30k), SH warm-up, normal consistency 0.1 | sugar_trainers/refine.py |

All values live in the component configs (`ss-inspect sugar`). `sugar-smoke` uses stage
boundaries 1000/1300/2000 for smoke tests only.

Mesh extraction needs Open3D, installed outside the env:
`pip install --no-deps --target ~/models/sugar-pylib open3d==0.19.0` plus its pure-Python import
deps (dash, flask, werkzeug, plotly, addict, pandas, pytz, sklearn, ...). The plugin appends
that directory to `sys.path` only if `import open3d` fails (env packages keep precedence).

Fidelity gaps (see ../plan.md): gsplat expected depth + point K-NN instead of the PyTorch3D
diamond z-buffer for level-set search, white background (official black), view sampling with
replacement, gsplat rasterizer (with a straight-through opacity cap at 0.999 so saturated
Gaussians keep gradients, as in the Inria rasterizer), sqrt floor at d = 1.

```
pip install -e .
ss-verify sugar --device cuda
ss-train sugar --data ~/datasets/mipnerf360/w1600/garden --output runs/x --steps 30000 \
  --recipe 3dgs --num-gaussians 10000000 --max-gaussians 12000000
ss-eval --run runs/x --split test
```
