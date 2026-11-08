# splatstudio-method-fsgs

Native SplatStudio port of **FSGS: Real-Time Few-shot View Synthesis using Gaussian Splatting**
(Zhu et al., ECCV 2024, https://arxiv.org/abs/2312.00451). Numerics follow the official code
(https://github.com/VITA-Group/FSGS, Inria Gaussian-Splatting license, research use; read only).
Paper fidelity unverified.

Method `fsgs`:
- `fsgs-renderer`: gsplat with accumulated depth Σ z α T (Eq. 7), black background, SH +1 every
  500 steps, straight-through opacity cap 0.999 (Inria-style backward).
- `fsgs-density`: official densify_and_prune (τ 5e-4, clone/split, FSGS distance split
  `mean sq. 3-NN dist > 10·extent & max scale > extent`, **proximity unpooling** for step < 2000
  `> 5·extent & max scale > extent` → midpoints to the 3 NN with destination scale/opacity,
  identity rotation, zero SH), prune opacity < 0.005, no size pruning, opacity reset to 0.05 at
  2001/5001/8001; owns 10 000 RegNeRF random LLFF pseudo poses (official near/far table, 8 scenes).
- `photometric`: 0.8 L1 + 0.2 D-SSIM.
- `fsgs-depth`: min(1−corr(−D_dpt, D), 1−corr(1/(D_dpt+200), D)), weight 0.05 (0.001 from 9502).
- `fsgs-pseudo`: 1−corr(D_pseudo, −DPT(render_pseudo)) with gradient through DPT,
  weight 0.5·min((t−2000)/500, 1), 2000 < t < 9500, every step.

DPT: MiDaS DPT_Hybrid (`~/models/dpt/midas_repo`, `~/models/dpt/dpt_hybrid_384.pt`, timm 0.6.12
from `~/models/dpt/pylib`, loaded lazily in-process).

Data: `llff-fewview3-r8-fsgs` (official images_8, 3-view split, MVS init of the train views).
Fidelity gaps: see `../plan.md`.

```bash
tools/ssenv ss-train fsgs --data ~/datasets/llff/fewview3-r8-fsgs/horns --output runs/x --steps 10000 \
  --recipe 3dgs --num-gaussians 10000000 --max-gaussians 12000000
```
