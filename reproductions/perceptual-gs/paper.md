# Perceptual-GS — implementation spec

Paper: Zhou & Ni, "Perceptual-GS: Scene-adaptive Perceptual Densification for Gaussian Splatting",
ICML 2025, arXiv 2506.12400. Official code: https://github.com/eezkni/Perceptual-GS (cloned into
`source/code/`; built on Inria 3DGS, Inria non-commercial research license; the depth-reinit
rasterizer `submodules/diff-gaussian-rasterization_ms` comes from Mini-Splatting).
**Code defaults override the paper where they differ (listed in §7).**

## 1. Summary and contributions
- 3DGS + a per-Gaussian learnable sensitivity `ε_i` (sigmoid), rendered as an extra branch and
  supervised by binary perceptual-sensitivity maps (Sobel + threshold + pooled threshold).
- Densification adds sensitivity-guided clone/split of high-sensitivity (σ(ε)≥0.9, every 1000 it)
  and medium-sensitivity (0.3<σ(ε)<0.9, every 1500 it) Gaussians whose max-over-views summed
  blending weight exceeds 25 / 10.
- Scene-adaptive behaviour decided at iteration 600: split-only mode if the scene sensitivity β>0.85;
  Mini-Splatting-style depth reinitialization if γ (fraction of medium-sensitivity Gaussians among
  the 25% largest) > 0.55.
- Opacity decline (OD) of cloned Gaussians: α̂ = 1 − sqrt(1 − α^k), k = 1.2.

Contributions:
- C1 Perceptual sensitivity extraction (§3.3, Eqs. 5–7) — offline preprocessing.
- C2 Dual-branch rendering + BCE sensitivity loss (§3.4, Eqs. 8–10).
- C3 Perceptual sensitivity-guided densification (§3.5, Eqs. 11–16).
- C4 Scene-adaptive depth reinitialization (§3.6, Eqs. 17–18) (+ code: no opacity reset, reinit
  from own centres at 15k).
- C5 Opacity decline for clone (§3.7, Eqs. 19–20).

## 2. Equations (verbatim) and code
Eq. 5–6: `G = \sqrt{(I \otimes G_x)^2 + (I \otimes G_y)^2}`, Sobel `G_x=[[-1,0,1],[-2,0,2],[-1,0,1]]`,
`G_y=[[-1,-2,-1],[0,0,0],[1,2,1]]`.
Eq. 7: `G_E(u) = \mathbb{I}(G(u) > \tau_e)`; then average pooling with threshold τ_s.
Code (`preprocess.py`): image = full-resolution `images/` file, PIL `convert("L")`, gamma
`255*(x/255)**(1/1.5)` via `Image.point`, `ToTensor` ([0,1]); conv2d padding 1; `>0.05`;
`avg_pool2d(k=5)` (stride 5, floor), nearest upsample to the original size, `>0.3`; saved as uint8
PNG. Training loads it with `PIL.resize((round(W/r), round(H/r)))` (bicubic default) /255 — the
training target is therefore NOT binary near edges.

Eq. 8: `R_v^S(u) = \sum_i \omega_i^v(u) \sigma(\epsilon_i)`.
Eq. 9: `L_v^S = L_{BCE}(I_v^S, R_v^S)`; Eq. 10: `L_v = (1-\lambda_S) L_v^C + \lambda_S L_v^S`.
Code (`train.py`): only while `iteration <= densify_until_iter (15000)`:
`r = render(override_color=(1-σ(ε)).repeat(1,3), bg=black)[0]`, `L_S = nn.BCELoss()(1-r, gt_sens)`
(mean; torch clamps log at −100). So the predicted map is `1 − Σ w(1−σ(ε)) = (1−A) + Σ w σ(ε)`
(empty pixels predict "sensitive"). Total = 0.9·(0.8 L1 + 0.2 DSSIM) + 0.1·L_S; after 15000 the
plain 3DGS loss. The RGB render's viewspace gradient (used for densification) therefore carries the
0.9 factor until 15k.

Eq. 11–12: `G_h = {ε_i > τ_h}`, `G_m = {ε_i ∈ [τ_l, τ_h]}` (code: σ(ε)≥0.9 and 0.3<σ(ε)<0.9).
Eq. 13–14: `G_D = {ω_i^{max} > τ^ω} ∩ (G_h ∪ G_m)`, `ω_i^{max} = MAX_v \sum_{u} ω_i^v(u)`.
Code: at `it % 1000 == 0 or it % 1500 == 0` (it ≤ 15000): optimizer step first, then
`trans = max_v accum_weights_v` over all training views (Mini-Splatting `render_imp`:
`accum_weights_i = Σ_pixels α_i T_i`), then the regular densify step uses it, then `trans = 0`.
- split (`densify_and_split`): mask = 3DGS (grad ≥ 2e-4 & max scale > 0.01·extent)
  ∪ [it≥1000, it%1000==0] (σ≥0.9 & trans>25 & (split_only or max scale > 0.01·extent))
  ∪ [it≥1000, it%1500==0] (0.3<σ<0.9 & trans>10). N=2, scales/1.6, opacity/ε copied.
- clone (`densify_and_clone`, runs first): mask = 3DGS (grad ≥ 2e-4 & small)
  ∪ [it≥1000, it%1000==0, not split_only] (σ≥0.9 & trans>25 & small).
Eq. 15–16: `β = mean_v (#pixels with sens>0)/(#pixels)` (code: `count(sens>0)/count(sens<=1)`),
`split_only = β > 0.85` (τ_β).

Eq. 17–18: `γ = |G_l ∩ G_m| / |G_l|`, `G_l` = Gaussians with max scale ≥ the k-th largest
(k = int(0.25 N)). Code: at it == 600, `mix_rate = count(0.3<σ<0.9 within G_l)/|G_l|`,
`sparse = mix_rate > 0.55`.
Sparse scenes (code): at it % 5000 == 0 (< 15000: 5000, 10000) the densify step is skipped and the
model is re-initialized from depth points: per training view, render_depth (Mini-Splatting) gives
per-pixel the point on the ray closest (in the 3σ-ellipsoid metric) to the max-weight Gaussian;
pixel probability ∝ 1 − T_final; `np.random.choice(H·W, int(H·W·3.5e6/(H·W·|V|)), p)` (with
replacement) then unique; colours = GT pixels; `reinitial_pts` (simple-knn scales, identity rot,
opacity 0.1, ε = logit 0.1, SH rest 0) + fresh optimizer. Opacity reset is disabled for sparse
scenes. At it == 15000 sparse scenes are re-initialized again from their own centres and DC colour.
Non-sparse: 3DGS opacity reset every 3000 (it < 15000): opacity = min(α, 0.01).

Eq. 19–20: `\hat{α} + (1-\hat{α})\hat{α} = OD(α)` → `\hat{α} = 1 - \sqrt{1 - OD(α)}`, `OD(x)=x^k`,
k = 1.2. Code: applied in place to the selected originals before cloning (clone copies it):
`opacity = logit(α^k / (1 + sqrt(1 − α^k)))` (identical algebraically).

## 3. Hyper-parameters
| name | value | source |
|---|---|---|
| τ_e, τ_s, gamma, pool | 0.05, 0.3, 1.5, 5 | Tab. 8; code:preprocess.py:10-13 |
| λ_S | 0.1 | Tab. 8; arguments `lambda_bce` |
| sensitivity lr / init | 0.05 / σ=0.1 | code arguments `sensitivity_lr`; gaussian_model.py:154 |
| Iter_h, Iter_m | 1000, 1500 | Tab. 8; `hd_interval`, `md_interval` |
| τ_h, τ_l | 0.9, 0.3 | Tab. 8 |
| τ_h^ω, τ_m^ω | 25, 10 | Tab. 8; `complex_transaction`, `mix_transaction` |
| τ_β | 0.85 | Tab. 8; train.py:206 |
| τ_γ | 0.55 | Tab. 8; train.py:189 |
| decision iteration | 600 | code train.py:178 |
| OD k | 1.2 | §3.7; `od_factor` |
| depth samples | 3.5M total | train.py:241 |
| depth reinit every | 5000 (<15000) | train.py:222 |
| 3DGS defaults | grad 2e-4, densify 500–15000 every 100, reset 3000, percent_dense 0.01, min opacity 0.005, 30k iters, lr as 3DGS | arguments/__init__.py |

## 4. Schedules
1–15000: BCE branch on (≤15000). 501..14900 every 100: densify (>500, <15000). 600: β, γ decision.
1000·k / 1500·k: weight statistics + sensitivity-guided densification. 3000·k (<15000, non-sparse):
opacity reset. 5000, 10000 (sparse): depth reinit instead of densify. 15000 (sparse): self-reinit.

## 5. Builds on
Vanilla 3DGS (losses, LR schedule, SH schedule, densification/pruning) plus Mini-Splatting's
`render_imp`/`render_depth` statistics and depth reinitialization.

## 6. Protocol
Mip-NeRF 360 (9), T&T (train, truck), DB (2), BungeeNeRF (8). `run.py`: M360 outdoor `-r 4`,
indoor `-r 2` (PIL resize of full-res images = our `mipnerf360-pilr`), T&T/DB `-r 1`; `--eval`
(llffhold8), black background, 30k iterations, LPIPS-VGG (3DGS metrics.py), RTX4090.
Per-scene PSNR/SSIM/LPIPS/#G/FPS in Appendix E (Tables 12–17); 3DGS* is the authors' retrain.

## 7. Ambiguities / paper vs code
- Paper says clone-only for G_h when β < τ_β; code splits large and clones small ones when not
  split_only, and splits all when split_only. Follow code.
- Paper G_m interval closed, code open (0.3, 0.9). Follow code.
- BCE target is the bicubic-resized (non-binary) map; prediction includes the empty-pixel term.
- Sensitivity maps are computed on full-resolution images (not the training resolution).
- The official extra `alpha` branch is inactive (no alpha masks in these datasets).
- 3DGS screen-size prune is inert in the code (`densification_postfix` zeroes max_radii2D).
