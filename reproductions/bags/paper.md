# BAGS: Blur Agnostic Gaussian Splatting through Multi-Scale Kernel Modeling

Peng, Tang, Zhou, Wang, Liu, Li, Chellappa — ECCV 2024, arXiv 2403.04926.
Project https://nwang43jhu.github.io/BAGS/ ; official code https://github.com/snldmt/BAGS
(cloned at `source/code`, commit ed04144, 2024-07-05; built on Mip-Splatting; README: "initial
version, will be updated more to match the exact performances reported in the manuscript").
LaTeX in `source/latex/` (no supplementary in the e-print).

## 1. Method summary
- Mip-Splatting scene (3D filter + 2D Mip filter, kernel 0.1) is rendered sharp: C (RGB) and D (depth).
- A **Blur Proposal Network** (BPN) predicts, per training view i and pixel x, a dense K×K kernel
  h(x,i) (softmax-normalised) and a mask m(x,i) ∈ [0,1] from a view embedding l(i), a 2-D sinusoidal
  positional encoding p(x) and CNN features of the rendered RGB-D (input detached).
- Training image = (1−m) C + m (h ⊛ C) (spatially varying convolution of the *rendered* image);
  loss = 0.8 L1 + 0.2 D-SSIM + mask sparsity (+ optional TV terms in code).
- **Coarse-to-fine**: train at 1/4, then 1/2, then full resolution with kernels 5, 9, 17
  (the kernel MLP grows a head per scale).
- Test views are rendered **sharp** (BPN bypassed; it is per-training-view anyway).

Contributions
- C1 Blur Proposal Network: per-pixel dense kernels + residual quality mask, MMF input (§3.1, Eq. 2–3).
- C2 Multi-scale BPN + coarse-to-fine optimisation (§3.2, Eq. 4–5).
- (base) Mip-Splatting renderer and 3DGS densification, with code-only schedule changes (§4 impl. details, code).

## 2. Equations (verbatim)
Forward blur model (Eq. 1, introduction.tex):
```
\tilde{C}(x) = \sum_{x_k\in \mathcal{N}(x)}C({x_k})h(x_k) \text{ s.t } \sum_{x_k\in \mathcal{N}(x)}h(x_k)=1
```
BPN (Eq. 2):
```
h(x,i), m(x,i) = \mathcal{F}_\textrm{kernel}(l(i)\oplus p(x)\oplus f_\textrm{RGBD}(x,i)),
\textrm{where } f_\textrm{RGBD}(x,i) = \mathcal{F}_\textrm{feat}(C(x,i) \oplus D(x,i)), 0 \leq m \leq 1.
```
Mask blend (Eq. 3): `C_out(x,i) = (1-m(x,i)) C(x,i) + m(x,i) \tilde{C}(x,i)`
Multi-scale BPN (Eq. 4):
```
f_inter(x,i) = F^s_base(l(i) ⊕ p(x) ⊕ f_RGBD(x,i))
h^s(x,i) = softmax(F^s_blur(f_inter(x,i))),   m^s(x,i) = sigmoid(F^s_mask(f_inter(x,i)))
```
Scale loss (Eq. 5):
```
\mathcal{L}_s = \lambda_{photo}\lVert C^s_{out}-C^s_{obs}\rVert + \lambda_{DS}\mathcal{L}_{D-SSIM}(C^s_{out}, C^s_{obs}) + \lambda_{mask}\lVert m^s\rVert
```
Scale s has downsampling 2^{s-1}; S = 3, K_3 = 5, K_2 = 9, K_1 = 17.

### Code form (train.py, scene/kernelnet_multires.py) — authoritative
- Convolution: `patches = Unfold(K, padding=K//2)(C)` (zero padding) → `[1,3,K²,H,W]`;
  `rgb = Σ_k patches[:, :, k] * h[:, k]` (correlation, k = ky·K + kx, offset (ky−K//2, kx−K//2));
  `blur = m·rgb + (1−m)·C`. Gradient flows into C through the patches; the BPN input
  `cat(C, D_norm).detach()`.
- `D_norm = (D − min D) / max(D − min D)`; D = alpha-blended **Euclidean distance** to the camera
  (Mip-Splatting rasterizer `depths = dist`, backward via |p − campos|), black background.
- p(x): `get_2d_emb(1, H, W, 16)` sinusoidal (8 ch for rows, 8 for cols, freqs 1/10000^{2j/8}, 2π/H·row).
- l(i): `nn.Embedding(num_train, 32)`.
- F_feat: Conv(4→64,5,p2) ReLU IN(64) Conv(64→64,5,p2) ReLU IN(64) Conv(64→32,3,p1) (bias on).
- F_kernel (1×1 convs, **no bias**): base1 = Conv(80→64) ReLU; scale 3: head1(64→25), mask1(64→1)
  on base1; scale 2: base2 = Conv(64→64) ReLU, head2(64→81), mask2; scale 1: base3 Conv(64→64) ReLU,
  head3(64→289), mask3 on base1∘base2∘base3. softmax over K², sigmoid mask.
- Optional `use_another_mlp`: after ms_steps a separate single-scale net (kernelnet_single_res:
  base Conv(80→64)ReLU Conv(64→64)ReLU, head(64→K_ss²), mask) is used instead of the ms net.
- Loss (`iteration > 250`): `0.8 L1(blur, gt) + 0.2 (1 − SSIM(blur, gt)) + 0.001·mean(m)`
  `+ [use_depth_loss] α_d·TV(D_norm) + [use_rgbtv_loss] 0.001·TV(C)`;
  `TV(g) = 2 (Σ(Δ_h g)²/(C(H−1)W) + Σ(Δ_w g)²/(CH(W−1)))`.
  For `iteration ≤ 250` the loss is the plain 3DGS loss on the sharp render C (no BPN).
- `iteration` = ori_iter for ori_iter ≤ ms_steps, else ori_iter − ms_steps (phase-2 restart).

## 3. Hyper-parameters
| name | value | source |
|---|---|---|
| kernel sizes K3/K2/K1 | 5 / 9 / 17 | paper §4 impl.; code arguments kernel_size1..3 |
| K_ss (use_another_mlp net) | 17 | code arguments kernel_size_ss |
| view embedding | 32 | code kernelnet_multires.py |
| pos. enc. channels | 16 | code train.py get_2d_emb(..,16) |
| CNN | 3 layers, 64 ch, k 5,5,3, out 32 | paper §4 + code |
| kernel MLP | 4 linear (1×1 conv) layers, 64 ch, no bias | paper §4 + code |
| BPN lr | 5e-4 (env KERLR; 1e-3 for tools and Mip-NeRF 360) | code gaussian_model.py:264, scripts |
| Adam eps | 1e-15 (BPN shares the Gaussian Adam) | code |
| λ_photo / λ_DS | 0.8 / 0.2 | code lambda_dssim 0.2 |
| λ_mask | 0.001 (use_mask_loss True) | code arguments |
| depth TV α | 0.001 default; 0.01 for girl, heron, daisy, seal | code / scripts |
| rgb TV α | 0.001 | code |
| BPN start | iteration > 250 (each phase) | code train.py |
| resolution switches | ori_iter 3000 (1/4→1/2), 6000 (→ full) | code upsample_iter |
| kernel switches | ori_iter <3000: 5, <6000: 9, ≥6000: 17 | code kernelnet_multires.forward |
| ms_steps | per scene (6000–18000) | scripts/*.sh |
| iterations | per scene (10000–58000) | scripts/*.sh |
| init_dgt (grad thr while ori_iter ≤ ms_steps) | 0.0002/0.0006/0.0008 per scene | scripts |
| densify_grad_threshold (phase 2) | 0.0002 | code |
| min_opacity (prune) | 0.1 (all deblur scenes) | scripts |
| densify from / until / interval | 500 / 35000 / 100 (on `iteration`) | code |
| opacity reset | every 3000 of `iteration`, filter-aware min(·,0.01) | code (Mip-Splatting) |
| percent_dense | 0.01 | code |
| screen-size prune | `max_radii2D > 20` after 3000, but max_radii2D is zeroed by densification_postfix before the prune → never fires; world-size prune only if min_opacity == 0.005 → off | code |
| xyz lr | 1.6e-4·extent → 1.6e-6·extent, max_steps 60000, on `iteration` (reset at ms_steps) | code |
| other lrs | f_dc 2.5e-3, f_rest 1.25e-4, opacity 0.05, scaling 5e-3, rotation 1e-3 | code (3DGS) |
| Mip filters | 3D filter coef 0.2 (cameras of the current scale), 2D kernel 0.1 | code |
| background | black | code (white_background False) |
| SH | +1 degree every 1000 `iteration` up to 3 | code |
| init | SfM points, opacity 0.1, 3-NN scales | code (3DGS) |

Per-scene official scripts (scripts/*.sh):
| scene | hold | init_dgt | iters | ms_steps | extra flags |
|---|---|---|---|---|---|
| ball | 7 | 8e-4 | 49000 | 9000 | – |
| basket | 7 | 6e-4 | 40000 | 6000 | depth |
| buick | 7 | 8e-4 | 38000 | 9000 | – |
| coffee | 6 | 6e-4 | 12000 | 6000 | – |
| decoration | 6 | 8e-4 | 49000 | 9000 | – |
| girl | 7 | 8e-4 | 49000 | 9000 | depth 0.01, rgbtv |
| heron | 8 | 8e-4 | 49000 | 9000 | depth 0.01, rgbtv |
| parterre | 6 | 8e-4 | 35000 | 9000 | – |
| puppet | 6 | 8e-4 | 28000 | 9000 | – |
| stair | 6 | 8e-4 | 25000 | 9000 | – |
| cake | 9 | 8e-4 | 58000 | 12000 | depth, rgbtv, another_mlp |
| caps | 5 | 6e-4 | 12000 | 9000 | depth, rgbtv |
| cisco | 7 | 6e-4 | 49000 | 6000 | depth |
| coral | 10 | 6e-4 | 22000 | 18000 | another_mlp |
| cupcake | 6 | 6e-4 | 30000 | 10000 | – |
| cups | 8 | 6e-4 | 22000 | 9000 | – |
| daisy | 8 | 2e-4 | 18000 | 9000 | depth 0.01, another_mlp |
| sausage | 6 | 6e-4 | 12000 | 9000 | – |
| seal | 6 | 2e-4 | 10000 | 7000 | depth 0.01, rgbtv |
| tools | 7 | 6e-4 | 40000 | 6000 | depth, rgbtv, KERLR 1e-3 |

## 4. Schedules (ori_iter = global step, it = phase-local iteration)
- 1–250: no BPN (3DGS loss at 1/4 res). 251–2999: BPN K=5 at 1/4 res (150×100).
- 3000–5999: 1/2 res (300×200), K=9. ≥6000: full (600×400), K=17.
- ori_iter = ms_steps+1: `it` restarts at 1 → xyz LR restarts, densification/reset clocks restart,
  BPN off again for it ≤ 250, grad threshold switches init_dgt → 2e-4; with use_another_mlp the
  single-scale net replaces the ms net.
- Densify at it > 500, it % 100 == 0, it < 35000; reset at it % 3000 == 0 (it < 35000); 3D filter
  recomputed at each resolution change, after each densification, and every 100 it for
  35000 < it < (iterations − ms_steps) − 100.
- Final model = state after the last iteration (saved before that iteration's densification).

## 5. Builds on
Mip-Splatting (3D smoothing filter + 2D Mip filter), 3DGS densification and losses.
Unchanged: SH, init, Adam LRs except xyz schedule, clone/split rules.

## 6. Protocol
- Deblur-NeRF real camera-motion (10 scenes) and real defocus (10 scenes) sets, authors' Drive folder
  (same as Deblur-NeRF; `sparse/0` COLMAP, `hold=N`). Images `images/` full-res read with `-r 4` →
  600×400 (PIL resize). Split: names sorted, idx % llffhold == 0 → test (sharp references), rest train.
  SfM points of `sparse/0` initialise. No low-res / defocus-specific variant sets for these tables.
- Mixed-resolution Mip-NeRF 360 (nerfstudio re-calibration, 4 groups ×1,2,3,4 down) and the
  unbounded drone set (not released) are additional experiments — out of scope here.
- Metrics (`metrics.py -t deblur-nerf`): renders saved as 8-bit PNG; PSNR/SSIM from skimage on
  [−1,1] images (data_range 2 → identical to [0,1] PSNR; SSIM 7×7 uniform, sample covariance),
  LPIPS = `lpips.LPIPS()` default **AlexNet** on [−1,1]. Final iteration (`render.py --iteration -1`).

## 7. Ambiguities
- Paper: "train first two scales 3000 steps each, 40K iterations overall" vs code: per-scene
  iterations 10k–58k and ms_steps 6k–18k; per-scene flags (depth TV, rgb TV, another_mlp, KERLR).
  We follow the per-scene scripts (code overrides paper).
- README warns the release may not exactly reproduce manuscript numbers.
- Paper's mask loss ‖m‖ = code 0.001·mean(m). TV losses are code-only.
- Coarse-level target images: official PIL-resizes the 2400×1600 JPG directly to 150×100 / 300×200;
  our dataset stores 600×400, so coarse targets are antialiased resizes of the 600×400 image.
- Test-time evaluation renders sharp C; training-time `testing_iterations` only print PSNR (no
  checkpoint selection in the scripts; render.py takes the last iteration).
