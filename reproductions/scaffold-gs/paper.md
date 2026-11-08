# Scaffold-GS: Structured 3D Gaussians for View-Adaptive Rendering (CVPR 2024, arXiv 2312.00109)

Sources: LaTeX `source/latex/` (main + `sec/X_suppl.tex`); official code `source/code/` →
symlink to `reproductions/hard-gs/source/code/scaffold-gs` (github.com/city-super/Scaffold-GS @59c833b,
same clone the HGS port was written against). **Code defaults override the paper.**

## 1. Summary and contributions
- Anchors from a voxelized SfM cloud; each anchor has a 32-D feature f_v, a scaling l_v and k=10 offsets.
- Per view, only frustum-visible anchors spawn k neural Gaussians whose opacity/colour/scale/quaternion
  are decoded by small MLPs from [feature, view direction] ("on the fly", view adaptive).
- Neural Gaussians with opacity ≤ τ_α (=0, tanh output) are dropped before rasterization.
- Anchors grow at multi-resolution voxels where neural-Gaussian average gradients are large, and are pruned when
  their neural Gaussians stay transparent.

C1 anchor init by voxelization (§3.2.1, Eq. 2). C2 view-adaptive neural Gaussian derivation (§3.2.2, Eq. 4–6) incl.
visibility prefilter + opacity selection. C3 anchor growing/pruning (§3.3, Eq. 7). C4 loss with volume regulariser (Eq. 8).

## 2. Equations
- Eq. 2: $\mathbf{V} = \{\lfloor \mathbf{P}/\epsilon \rceil\}\cdot\epsilon$ (dedup).
- Eq. 4: $\{\mu_0..\mu_{k-1}\} = x_v + \{O_0..O_{k-1}\}\cdot l_v$ (code: $l_v$ = exp(scaling[:, :3])).
- Eq. 5–6: $\{\alpha_i\} = F_\alpha(\hat f_v, \delta_{vc}, \vec d_{vc})$, likewise $F_c, F_q, F_s$. Code (scripts):
  input is $[f_v, \vec d_{vc}]$ only (add_*_dist False, use_feat_bank False, appearance_dim 0);
  opacity = tanh, colour = sigmoid, scale = scaling[:, 3:]·sigmoid(F_s), quat = normalize(F_q).
- Eq. 7: $\epsilon_g^{(m)} = \epsilon_g/4^{m-1}$, $\tau_g^{(m)} = \tau_g\,2^{m-1}$, m∈{1,2,3}; random elimination
  of candidates (code keeps with prob 1−0.5^m). Code: ε_g^(1) = voxel_size·update_init_factor(16), /4 per level;
  τ_g = densify_grad_threshold 2e-4 (NDC view-space grad norm averaged over denom), ×2 per level.
- Eq. 8: $L = L_1 + \lambda_{SSIM} L_{SSIM} + \lambda_{vol} \sum \mathrm{Prod}(s_i)$. Code train.py:145:
  0.8·L1 + 0.2·(1−SSIM) + **0.01**·mean(prod(scale)) (paper text says λ_vol = 0.001 — code wins).

## 3. Hyper-parameters (code: arguments/__init__.py, train_*.sh)
| name | value | source |
|---|---|---|
| feat_dim / n_offsets | 32 / 10 | args |
| voxel_size | 0.001 mip360, 0.01 T&T, 0.005 DB | train_mip360/tnt/db.sh (paper supp.: 0.005/0.01 or 1-NN median) |
| update_init_factor / hierachy / depth | 16 / 4 / 3 | scripts, args |
| start_stat / update_from / interval / until | 500 / 1500 / 100 / 15000 | args |
| min_opacity (prune: accum < 0.005·visits) | 0.005 | args (paper: 0.5) |
| success_threshold | 0.8 (offset denom > 40, anchor visits > 80) | args |
| densify_grad_threshold | 2e-4 | args |
| lr offset | 0.01→1e-4 ×spatial_lr_scale (exp) | args |
| lr feature / scaling / position | 0.0075 / 0.007 / 0 | args |
| lr mlp_opacity / cov / color | 0.002→2e-5 / 0.004 const / 0.008→5e-5 | args |
| iterations | 30000 | train.sh |
| lambda_dssim / vol | 0.2 / 0.01 | train.py |
| background | black | args white_background False |

## 4. Schedules
Exponential LR decay over 30k (offset, MLPs). Stats from step 500, anchor adjust every 100 steps in (1500, 15000).
Prune also clamps raw scaling[:, 3:] ≤ 0.05 (in `_prune_anchor_optimizer`).

## 5. Builds on
3DGS rasterizer (fork with visible_filter); SH not used (MLP colour). Adam eps 1e-15.

## 6. Protocol
- Mip-NeRF 360: 7 scenes (bicycle garden stump room counter kitchen bonsai); official `train.sh` passes
  `-s data/mipnerf360/<scene>` with default `images` and `--resolution -1` → full-res PIL-resized to width 1600
  (camera_utils.loadCam) = our `mipnerf360-1600`. The paper says "adhered to 3D-GS" for preprocessing (ambiguous);
  code wins. T&T / DB: inria tandt_db images (< 1600 wide, no resize).
- llffhold 8 (`--eval`), 30k iterations, LPIPS-VGG (lpipsPyTorch) in metrics.py.
- Paper's 3DGS rows are the 3DGS paper's own numbers (images_4/images_2), not rerun at 1600.

## 7. Ambiguities
- Resolution (above). Feature bank / distance inputs described in the paper but disabled in the released scripts.
- λ_vol 0.001 (paper) vs 0.01 (code); prune 0.5 (paper) vs 0.005·visits (code); τ_g = 64ε (paper) vs fixed 2e-4.
- Growth happens before `optimizer.step()` in the official loop.
