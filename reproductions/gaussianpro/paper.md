# GaussianPro: 3D Gaussian Splatting with Progressive Propagation — implementation spec

Paper: Cheng et al., ICML 2024 (PMLR v235, cheng24f), arXiv 2402.14650. Sources in `source/`:
`latex/` (arXiv v1 — its appendix is empty), `icml.pdf`/`icml.txt` (camera-ready, appendix A–D
with per-scene tables), `code/` (github.com/kcheng1021/GaussianPro @ 86bb048, non-commercial
licence), `simple-knn/` (gitlab.inria.fr/bkerbl/simple-knn @ 86710c2, Inria licence).

## 1. Method summary
- 3DGS + a second densification route: every m=50 iterations the sampled training view's rendered
  depth seeds a PatchMatch (ACMH checkerboard) propagation of plane hypotheses against its
  neighbouring views (official CUDA `submodules/Propagation`).
- Propagated depth is filtered with a multi-view geometric consistency check; pixels whose
  rendered depth deviates from the filtered propagated depth by abs-rel > σ are back-projected
  (1/8 sub-sampled) and inserted as new Gaussians.
- Planar constraint: L1 + angular loss between rendered normal (shortest Gaussian axis,
  alpha-blended) and the filtered propagated normal, plus a NeuSG flatten loss on min scale.

Contributions
- **C1 hybrid geometric representation** (§4.1): render depth z_i and normal n_i (shortest axis)
  maps by alpha blending.
- **C2 progressive propagation** (§4.2, App. A): plane definition, checkerboard candidates,
  homography patch matching with NCC, u iterations, geometric filtering, selection
  (abs-rel > σ), back-projection to new Gaussians.
- **C3 plane-constrained optimisation** (§4.3): L_normal (Eq. 8) + γ L_scale (Eq. 9).
- **C4 training strategy** (§4.4): propagation every m=50 iterations, Eq. 10 total loss.

## 2. Equations (camera-ready numbering)
- Eq. 3 depth: $\mu_i' = W\mu_i + t$, $z_i$ = third component.
- Eq. 4 normal: $n_i = R_i[r,:],\ r = \arg\min([s_1,s_2,s_3])$ (code: column `R[:, :, r]`, flipped
  to face the camera: `n *= sign` with sign=+1 iff $(\mu_i - C)\cdot n < 0$, then $R_{w2c} n$).
- Maps: Eq. 2 with $c_i$ replaced by $z_i$ / $n_i$; code renders them with the Inria rasterizer as
  precomputed colours over the (black) background, normal map `F.normalize(dim=0)`.
- Eq. 5 plane: $d = z\, n^\top K^{-1}\tilde p$ (kernel: `GetDistance2Origin` = $-n\cdot X$).
- Eq. 6/7 homography: $\tilde p' \simeq H\tilde p$, $H = K(W_{rel} - t_{rel} n^\top / d) K^{-1}$.
- NCC cost: bilateral-weighted NCC over a patch (kernel `ComputeBilateralNCC`, cost = 1 − NCC
  clipped to [0,2]; top-k=4 views averaged).
- Eq. 15/16 geometric consistency: warp ref depth into target g and back; keep pixel if reprojection
  error < 2 px (code `thre1=2`) and relative depth error < 0.01 (`thre2`) in ≥ τ=2 of ≤4 targets.
- Selection: $|D_{prop} - D_{render}| / D_{prop} > σ$.
- Eq. 8: $\mathcal L_{normal} = \sum_{p\in Q} \|\hat N(p) - \bar N(p)\|_1 + \|1 - \hat N(p)^\top \bar N(p)\|_1$
  (code: L1 summed over channels, both terms **mean** over Q, separate weights 0.001 / 0.001).
- Eq. 9: $\mathcal L_{planar} = β \mathcal L_{normal} + γ \mathcal L_{scale}$,
  $\mathcal L_{scale}$ = mean over Gaussians of clamp(min_j s_j, 0, 30) (code `flatten_loss`).
- Eq. 10: $\mathcal L = (1-λ)\mathcal L_1 + λ \mathcal L_{D-SSIM} + \mathcal L_{planar}$, λ=0.2.

## 3. Hyper-parameters
| name | value | source |
|---|---|---|
| iterations | 30000 | §5.1 |
| propagation interval m | 50 | §4.4, §5.1, demo.sh (`--propagation_interval 50`); code default 20 |
| propagation iterations u | paper 3; **kernel 6** (`PatchMatchParams.max_iterations = 6`, not exposed) | §5.1 vs `PatchMatch.h:44` |
| σ (abs-rel threshold) | 0.8 final; linearly annealed from 1.0 (max) to 0.8 (min) over the propagation window | §5.1; demo.sh `--depth_error_max_threshold 1.0 --depth_error_min_threshold 0.8`; train.py:131 |
| propagation window | (1000, 12000) exclusive; demo.sh uses 6000 | arguments default `propagated_iteration_begin=1000, _after=12000` |
| β (lambda_l1_normal, lambda_cos_normal) | 0.001 each | §5.1, demo.sh (code default 0.01) |
| γ (lambda_flatten) | 100 | §5.1, code default |
| patch size | 20 (radius 10, step 2 → 11×11 samples) | demo.sh, default; `radius_increment=2` |
| sigma_spatial / sigma_color | 5 / 3 (grey levels 0–255) | PatchMatch.h |
| top_k views in cost | 4 | PatchMatch.h |
| neighbour views (360) | 4 training views with most shared SfM points (score > 0) | App. A; `load_pairs_relation` |
| depth range | [0.1, 20] (dataset '360') → kernel [0.06, 24] | graphics_utils.depth_propagation, PatchMatch.cpp |
| geometric check | thre1 = 2 px, thre2 = 0.01, ≥ 2 views | train.py:162, :172 |
| invalid marker | 300 | train.py |
| min new pixels | > 100 masked pixels | train.py:186 |
| back-projection stride | every 8th pixel in x and y | gaussian_model.py:441 |
| new Gaussian init | colour = GT pixel (SH0 = RGB2SH), SH rest 0, scale = log sqrt(distCUDA2 over new ∪ existing), identity rotation, **opacity inverse_sigmoid(1.0) = +inf** | densify_from_depth_propagation |
| opacity mask | L1 and SSIM only over pixels with rendered opacity > 0.999 for iteration < 1000 (when propagation on), > 0 afterwards | train.py:197-203 |
| 3DGS params | position_lr 1.6e-4→1.6e-6 ×extent, feature 2.5e-3, opacity 0.05, scaling 5e-3, rotation 1e-3, percent_dense 0.01, densify 500–15000/100, grad 2e-4, reset 3000 | §5.1 "3DGS schedule and hyper-parameters" (demo/waymo scripts change lrs for large scenes — not used for 360) |

## 4. Schedules
- iteration < 1000: opacity-masked photometric loss (alpha > 0.999).
- 1000 < it < 12000, it % 50 == 0 and the sampled view not yet propagated: propagate that view
  **once** (`propagation_dict`); each view propagates at most once in training.
- Normal loss: every iteration whose view has a stored propagated normal (from its propagation
  onwards, to 30000); flatten loss every iteration.
- σ annealed linearly 1.0 → 0.8 over (1000, 12000).
- 3DGS densification/opacity reset unchanged.

## 5. Builds on
Inria 3DGS (diff-gaussian-rasterization unmodified, simple-knn), all 3DGS training kept.

## 6. Protocol
- Waymo (9 segments, custom data — not available here), Mip-NeRF 360 (all 9 scenes, per-scene in
  App. D Tables 13–15), llffhold8, 30k iterations, RTX 3090. Resolution "follows 3DGS"
  (images_4 outdoor / images_2 indoor). 3DGS row "(Retrained)" is 3DGS trained on the authors'
  re-run COLMAP poses and points ("we retrain 3DGS using our generated SfM point clouds") —
  GaussianPro presumably uses the same; those inputs are not released. Black background (code).
- Official repo also ships `results/` CSVs (Baseline vs GaussianPro, per scene, 360/T&T/DB/ETH3D);
  these are not paper tables and do not match Table 14 (e.g. garden 27.24 → 27.12).

## 7. Ambiguities / code-vs-paper (code wins)
1. u = 3 in the paper, 6 in the kernel.
2. Paper says plane uses the rendered normal; the kernel ignores the input normal (zeros) and
   initialises each pixel with a random normal at the rendered depth (random depth where ≤ 0).
3. Propagation window end for 360 not stated (default 12000; demo 6000).
4. New Gaussians get opacity +inf (fully opaque until the next opacity reset).
5. Colour of new Gaussians is taken from `gt_image` of the **previous** iteration (variable reuse).
6. When a source view has no cached depth it is propagated with the *reference view's* source list
   (which contains the source itself) and cached; the reference view's cache is the tensor that
   is later overwritten in place with 300 outside the inserted pixels; caches are never refreshed.
7. `densification_postfix` zeroes gradient accumulators / max radii for all Gaussians on every
   propagation insertion.
8. The normal pass shares `screenspace_points` with the RGB pass, so the normal-loss gradient adds
   to the densification statistic.
9. Neighbour selection for 360: "number of shared SfM points" (App. A); the pair file generator is
   not released — the loader takes the first 4 training entries with score > 0.
