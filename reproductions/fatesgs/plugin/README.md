# splatstudio-method-fatesgs

FatesGS (Huang, Wu et al., AAAI 2025, arXiv 2501.04628), native SplatStudio port of the
official code https://github.com/yulunwu0108/FatesGS (commit a721d318). Paper fidelity
unverified.

Methods: `fatesgs` (full), `fatesgs-2dgs` (same pipeline without L_r/L_s/L_f).

| component | what | source |
|---|---|---|
| `fatesgs-surfel` renderer | official diff-surfel-rasterization (2DGS CUDA) vendored in `ss_fatesgs/csrc`, JIT-built; black background; surf_depth = expected depth / alpha; surf_normal = depth_to_normal·alpha.detach() | repo submodule, gaussian_renderer/__init__.py |
| `fatesgs-density` | 2DGS clone / in-plane split / prune (<0.1, >20 px, >0.1·extent after 1000), reset min(o,0.1) every 1000, densify 500–15000 every 100, grad 2e-4; xyz LR 1.6e-4→1.6e-6 over 30000 | scene/gaussian_model.py, arguments |
| `photometric` | 0.8 L1 + 0.2 (1 − SSIM, same padding) | utils/loss_utils.py |
| `surfel-reg` | 1e4·dist (>3000) + 0.05·(1 − n·N) (>7000) | train.py |
| `depth-rank` (Eq. 8) | 10 × patch rank, M=32, m=1e-4, crops 576×768 & 544×736 | loss_utils.py |
| `depth-smooth` (Eq. 9) | 1 × TVLoss(m_e=1e-2, m_t=1e-4) | loss_utils.py |
| `feature` (Eq. 14) | 1.5 × Vis-MVSNet levels 1,2 (weights 1, 1/2), 2 source views, diff<0.5, official visibility bookkeeping (`visibility: zbuffer` = paper Eq. 15) | utils/feat_utils.py |

Data: `reproductions/fatesgs/tools/prepare_dtu.py` → `~/datasets/dtu_fatesgs/native/scanX`
(sidecar `fatesgs.npz`: Marigold depth, pairs, alpha masks; `images_full/` for features).
Weights: Vis-MVSNet feature extractor `~/models/vismvsnet/vismvsnet.pt`
(sha256 4e172887e429e77759c56df2201f2b3c8978cca656992107947c6987a97be05a, from the FatesGS repo
`utils/vismvsnet.pt`), loaded with `weights_only=True`.

```bash
T=/home/sejain/splatify/tools/ssenv
$T ss-train fatesgs --data ~/datasets/dtu_fatesgs/native/scan24 --output RUN --steps 15000 \
   --recipe 3dgs --num-gaussians 10000000 --max-gaussians 12000000
PYTHONPATH=~/models/dtueval-pylib $T fatesgs-dtu-eval --run RUN \
   --data ~/datasets/dtu_fatesgs/native/scan24 --dtu ~/datasets/dtu_official/gt   # -> RUN/dtu_cd.json
```

Fidelity gaps: views sampled with replacement (official: shuffled stack); densification after
the optimizer step (official before); runtime's normalised-randn rotation init (official
torch.rand); PSNR split (views 22 / 25, 28) is informational, not the paper's NVS protocol.
