# splatstudio-method-decomp-gs

Native SplatStudio port of **DecompGS — Decomposing Densification in Gaussian Splatting for
Faster 3D Scene Reconstruction** (Huang, Liu, Wong; https://arxiv.org/abs/2507.20239).
No official code is released; written from the paper. Renderer/target downsampling adapted from
the DashGaussian port in this workspace. Paper fidelity unverified.

Mechanism (config defaults = paper values):
- G2L densification: split only before T_2, 3DGS split + clone after (`g2l`, `global_mode`).
- Energy-guided C2F resolution: T_r = round(E_r/E_1 · 25000), r = 8..1 (`c2f`,
  `max_scale_factor`, `densify_until`); position LR held until T_2 then log-linear decay.
- Adaptive opacity pruning: τ = min(sorted α[⌊N·0.03⌋], 0.05) (`prune_ratio`, `prune_upper`).

Fidelity gaps: see `../plan.md`.

```bash
pip install -e .
ss-inspect decomp-gs && ss-verify decomp-gs --device cuda
ss-train decomp-gs --data ~/datasets/tandt_db/native/truck --output runs/x --recipe 3dgs --num-gaussians 10000000
```
