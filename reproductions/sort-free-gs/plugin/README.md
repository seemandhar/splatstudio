# splatstudio-method-sort-free-gs

Sort-free Gaussian Splatting via Weighted Sum Rendering (Hou et al., ICLR 2025,
[arXiv 2410.18931](https://arxiv.org/abs/2410.18931)). Native SplatStudio port; paper fidelity
unverified. No official code; unstated learning rates/inits taken from the unofficial
re-implementation github.com/LiYukeee/sort-free-gs (read only).

## Mechanism
- `wsr-renderer` (custom CUDA, JIT, `csrc/wsr_raster.cu` adapted from gsplat 1.5.3
  RasterizeToPixels3DGS{Fwd,Bwd}.cu): per pixel S_f = Σ f_i α_i w_i, S_w = Σ α_i w_i with
  α_i = u_i G_i; no sorting (tile binning with a constant depth key), no transmittance;
  C = (S_rgb + w_B c_B)/max(S_w + w_B, 1e-4) (Eq. 7), clamped to [0,1].
- LC-WSR weight w_i = max(0, 1 − z_i/σ)·v_i (Eq. 10), σ global (init 10), v_i = sigmoid(opacities)
  (init 0.1). `sort-free-gs-dir`: w_i = v_i (DIR-WSR ablation).
- View-dependent unclamped opacity u_i = 0.1 + SH_deg(dir; H_i) (Eq. 11), fields `opa_sh0`,
  `opa_shN`; Gaussians with u ≤ 1/255 or w ≤ 0 are culled (unofficial behaviour).
- `wsr-density`: gsplat DefaultStrategy (3DGS clone/split, world-size prune), no opacity prune,
  no opacity reset; sets v_i lr 0.01, w_B lr 0.01, σ lr 0.1→0.001 × scene scale (exp, 30k).
  σ and w_B are global scalars stored as identical per-Gaussian rows (`wsr_sigma`, `wsr_wbg`);
  the renderer gives each row the full scalar gradient (straight-through on the row mean) and the density re-synchronises rows and Adam moments each step, which matches scalar Adam exactly (probe-checked). σ, w_B, the culled fraction and the near-zero-denominator pixel fraction are logged every 1000 steps.

## Config (defaults = paper / unofficial)
renderer: weight lc, u_offset 0.1, cull_nonpositive true (False is unsafe: the denominator clamp maps negative values to 1e-4), den_eps 1e-4, clamp_output true, background 0.0 (c_B black, as in the paper/unofficial).
density: grow_grad2d 2e-4, prune_opa 0, opacity_reset false, vi_lr 0.01, sigma_lr_init 0.1,
sigma_lr_final 0.001, sigma_lr_max_steps 30000, wbg_lr 0.01.

## Fidelity gaps
See `../plan.md` (u ≤ 0 culled; α threshold on u·G with 3.33σ footprint; world-size prune kept;
culled Gaussians never pruned; EXP-WSR not implemented; lrs/inits not in the paper).

## Commands
```bash
pip install -e .
ss-inspect sort-free-gs && ss-verify sort-free-gs --device cuda
ss-train sort-free-gs --data ~/datasets/mipnerf360/native/garden --output runs/x --recipe 3dgs --num-gaussians 10000000
```
