# splatstudio-method-dofgs

DOF-GS (Wang, Chakravarthula, Chen, CVPR 2025, arXiv 2405.17351) as a native SplatStudio
method `dofgs`, ported from github.com/rongduo/DOFGS_implementation @cfd635c.

## Mechanism
- `ss_dofgs/csrc/`: official `diff-gaussian-rasterization-dof`, **unmodified**
  (Inria/MPII Gaussian-Splatting licence, non-commercial research use: `csrc/LICENSE-DOFGS.md`),
  JIT-built by `ss_dofgs.raster.extension()` (needs `tools/ssenv` for nvcc/gcc 11).
  Thin-lens CoC per Gaussian `R = Q/2 |1/f - 1/z|`, covariance `+ R^2/(2 ln 4) I`, renders RGB,
  ray-distance depth, CoC map; returns dL/df, dL/dQ.
- `bank.py`: per-view f_m / Q_m (official reparameterisation `f = 10 clamp(Fs)`,
  `Q = 100 clamp(Ks) + 100`, supp. B.1 initialisation) and the ILN, process-local
  (`bank.ACTIVE`), created and stepped by the density component; exported to the checkpoint
  (`density_state["bank"]`).
- `renderer.py`: training views → defocused render (+ AiF after step 5000); every other camera →
  All-in-Focus (Q = 0), which is what `ss-eval` measures (official test protocol).
- `losses.py`: `dofgs-objective` (L_rec warm-up ≤ 5000, then the two composite terms of
  training.py) and `dofgs-mask` (Pearson L_mk + entropy L_reg, λ 0.002 / 0.0002).
- `density.py`: official 3DGS densification (percent_dense 0.03, prune < 0.005, until 25k),
  compactness densification every 1k, extra points at 2k, Mip 3D filter, xyz LR (max 80k).

Extra dependency: `torch_cluster` (same KNN as official) in `/home/sejain/models/dofgs-pylib`
(`pip install --target`, sys.path fallback).

## Data
`reproductions/dofgs/tools/prepare_deblurnerf.py <scene>` builds
`~/datasets/deblurnerf/native/<scene>` from the Deblur-NeRF real defocus set (600×400, official
per-scene holdout, validation = copies of the test images).

## Commands
```
tools/ssenv ss-verify dofgs --device cuda
tools/ssenv ss-train dofgs --data ~/datasets/deblurnerf/native/cake --output runs/x \
    --steps 40000 --recipe 3dgs --num-gaussians 10000000 --max-gaussians 12000000
tools/ssenv ss-eval --run runs/x --split test
```

## Fidelity gaps
View sampling with replacement; optimizer step before densification; SplatStudio SSIM
(valid conv) vs skimage SSIM in the paper; final-step evaluation (official val scripts pick
per-scene iterations); log-space Mip filter coefficient (GOF port) instead of the direct
determinant ratio. See `reproductions/dofgs/report.md`.
