# Compact 3D Gaussian Representation for Radiance Field — implementation spec

Lee, Rho, Sun, Ko, Park. CVPR 2024 (highlight). arXiv **2311.13681** (v2 LaTeX in `source/latex/main.tex`).
Official code: https://github.com/maincold2/Compact-3DGS (cloned depth 1 in `source/code/`, Inria 3DGS licence).
Pinned dependencies read for exact semantics: `vector-quantize-pytorch==1.8.1`, `dahuffman==0.4.1`
(sources in `source/deps/`), tiny-cuda-nn (HashGrid, SphericalHarmonics, FullyFusedMLP).

Not to be confused with plan #42 (Niedermayr, "Compressed 3DGS", slug `c3dgs`). The *extended* journal
version (arXiv 2408.03822) adds G-PCC position coding; **the CVPR paper and the official code store
positions as fp16 and do not use G-PCC**. No external codec binary is needed.

## 1. Method summary
- 3DGS with a learnable per-Gaussian **volume mask**: a binary (STE) mask multiplies both scale and opacity;
  a mean-sigmoid penalty drives masks off; masked Gaussians are removed during training.
- **View-dependent colour from a neural field** instead of per-Gaussian SH: a multiresolution hash grid
  (Instant-NGP, tcnn) at the contracted Gaussian position + SH encoding of the view direction → tiny MLP →
  0-degree SH coefficient → RGB.
- **Geometry codebooks**: scale and rotation represented by residual VQ (6 stages × 64 codes), learnt only
  during the last 1000 iterations with K-means initialisation.
- Storage: positions/opacity fp16, R-VQ indices 6 bits/stage, codebooks fp32, hash + MLP fp16 ("Ours").
  "Ours+PP": 8-bit min–max quantisation of opacity and hash, hash pruning |θ|≤0.1, Huffman coding of
  quantised opacity, hash and R-VQ indices.

Contributions
- **C1 Gaussian volume mask** (§3.1, Eq. 1–3).
- **C2 Compact view-dependent colour** (hash grid + MLP) (§3.3, Eq. 7–8).
- **C3 Geometry codebook** (R-VQ for scale and rotation) (§3.2, Eq. 4–6).
- **C4 Storage / post-processing** (§4.1 "Ours+PP", Table 7 (tab:size)).

## 2. Equations (verbatim from main.tex)
Mask (Eq. 1–2):
```
M_n = \operatorname*{sg}(\mathds{1}[\sigma(m_n) > \epsilon] - \sigma(m_n)) + \sigma(m_n),
\hat{s}_n = M_n s_n,\quad \hat{o}_n = M_n o_n,
```
Mask loss (Eq. 3): `L_m = {1\over N} \sum_{n=1}^{N}\sigma({m_n})`.
R-VQ (Eq. 4–5):
```
\hat{r}_n^{l} = \sum_{k=1}^{l}\mathcal{Z}^k[i^k],\quad
i_n^l = \operatorname*{argmin}_{k} ||\mathcal{Z}^{l}[{k}]-(r_n - \hat{r}_n^{l-1})||_2^2,\ \hat{r}_n^0 = \vec{0}
```
Codebook loss (Eq. 6): `L_r = {1 \over NC} \sum_{k=1}^{L} \sum_{n=1}^{N} ||\operatorname*{sg}[r_n -\hat{r}_n^{k-1}] - \mathcal{Z}^k[i_n^k]||_2^2`
(and L_s for scale, applied to s *before* masking).
Colour (Eq. 7–8): `c_n(d) = f(contract(p_n), d; θ)`,
`contract(p) = p if ||p||≤1 else (2 - 1/||p||)(p/||p||)`; f outputs the 0-degree SH, converted to RGB.
Total loss (Eq. 9): `L = L_ren + λ_m L_m + L_r + L_s`, L_ren = 0.8 L1 + 0.2 D-SSIM (3DGS); L_r, L_s = 0
except during the last 1K iterations.

### What the code actually does (code overrides paper)
- `gaussian_renderer/__init__.py` (training branch): `mask = ((σ(m) > 0.01).float() - σ(m)).detach() + σ(m)` → ε = 0.01;
  `scales = exp(s)*mask`, `opacity = σ(o)*mask`, `rotations = normalize(q)`.
  For `iteration > rvq_iter (29000)`: `scales = vq_scale(exp(s))*mask`, `rotations = vq_rot(normalize(q))` (not re-normalised).
  `xyz = contract_to_unisphere(means.detach(), aabb=[-1,1]^3)` → paper contraction then `x/4 + 0.5` ∈ [0,1];
  `dir = (means - cam_center)/||·||` (not detached); `shs = mlp_head(cat[recolor(xyz), direction_encoding(dir)])`
  rendered as SH degree 0 (rasterizer: rgb = max(0.28209479·sh + 0.5, 0)).
- Mask loss in `train.py`: `opt.lambda_mask * torch.mean(torch.sigmoid(gaussians._mask))`, λ_m = 5e-4.
- L_r, L_s are **not** in the total loss: vector-quantize-pytorch 1.8.1 with `learnable_codebook=True,
  ema_update=False, commitment_weight=0, in_place_codebook_optimizer=Adam(lr=1e-4)` performs, in every
  training forward of every stage, `loss = mse(quantize, x.detach())` (mean over elements), one Adam step on
  that stage's codebook, re-quantises with the updated codebook, then straight-through
  `quantize = x + (quantize - x).detach()`. ResidualVQ: `residual -= quantized.detach();
  out += quantized` — so the STE gradient reaching the input is **L = 6 times** the render gradient.
  K-means init (`kmeans_init=True, kmeans_iters=1`): on the first forward, per stage, sample 64 vectors
  (randperm) from that stage's residuals, one assignment + mean update (empty clusters keep the sample).
  Distances: Euclidean (`cdist` via x²+y²−2xy, sqrt), argmin.
- Hash grid (tcnn `HashGrid`): 16 levels, 2 features, log2 T = 19, base resolution 16, per-level scale 1.447,
  tcnn scale `2^(l·log2 1.447)·16 − 1`, resolution `ceil(scale)+1`, pos = x·scale + 0.5, trilinear,
  dense indexing while `res^3 ≤ T_l`, else coherent prime hash (1, 2654435761, 805459861) mod T_l, level
  size `min(next_multiple(res^3, 8), 2^19)`. Init U(−1e-4, 1e-4). Total 6,299,960 entries × 2 = 12,599,920
  parameters (= 25.2 MB at fp16, paper Table tab:size ✓).
- Direction encoding: tcnn `SphericalHarmonics` with `"degree": 3` → degree² = **9** outputs (bands 0–2).
  tcnn maps its input from [0,1] to [−1,1] (`2x−1`) internally, but the official code feeds the unit
  direction in [−1,1] directly, so the SH polynomials are evaluated at `2d−1` (official quirk, kept).
- MLP: tcnn `FullyFusedMLP`, ReLU, 64 neurons, 2 hidden layers, no biases, output activation none,
  n_input 41 (32+9) padded to 48 (padding inputs = 1 in tcnn, i.e. an effective first-layer bias),
  output 3 padded to 16. tcnn param count 48·64 + 64·64 + 64·16 = 8192 (= 0.016 MB at fp16 ✓).
- Net optimiser: `Adam(hash+MLP, lr=net_lr=0.01, eps=1e-15)`, `ChainedScheduler([LinearLR(start_factor=0.01,
  total_iters=100), MultiStepLR(milestones=[5000,15000,25000], gamma=0.33)])`, stepped every iteration
  after `optimizer_net.step()`; no optimiser step at the final iteration.
- Mask parameter: `_mask` init **1.0** (σ = 0.731), Adam lr `mask_lr = 0.01`, in the 3DGS optimiser group.
- Pruning: at each densification (`iteration > 500`, every 100, `< 15000`):
  `prune = σ(m) ≤ 0.01 or opacity < 0.005 [or big]`; afterwards (`iteration ≥ 15000`) `mask_prune()` every
  `mask_prune_iter = 1000`; at 30000 `final_prune`.
- `final_prune(compress)`: prune σ(m)≤0.01; (compress: Morton sort — reorder only); xyz → fp16;
  VQ layers `training=False`: `_scaling, sca_idx = vq_scale(exp(s))`, `_rotation, rot_idx = vq_rot(normalize(q))`
  (activated values stored); storage (MB = 1e6 bytes):
  ```
  position = N*3*16/8/1e6;  opacity = N*16/8/1e6
  scale    = N*6*6/8/1e6 + 64*6*3*32/8/1e6;  rotation = N*6*6/8/1e6 + 64*6*4*32/8/1e6
  hash     = 12,599,920*16/8/1e6;  mlp = 8192*16/8/1e6
  ```
  compress (Ours+PP): `post_quant(σ(o))` and `post_quant(hash, prune=True)`:
  `min,max` from the *unpruned* tensor; `x *= (|x| > 0.1)` (hash only); `q = round((x−min)/(max−min)·255)`;
  decoded `(max−min)·q/255 + min`. Huffman (dahuffman `HuffmanCodec.from_data`, EOF symbol of frequency 1
  added to the tree) on `sca_idx`, `rot_idx` (all N·6 indices in one codec), `quant_opa`, `quant_hash`;
  size = Σ freq·codelen / 8 / 1e6 (tables not counted) + codebooks (fp32); position, MLP fp16 as above.
  Without compress: opacity → fp16. `precompute()`: hash features at `contract(xyz.half())` are cached and
  test rendering uses `mlp(cat[feature, SH(dir)])` with the stored attributes.

## 3. Hyper-parameters
| name | value | source |
|---|---|---|
| iterations | 30000 | supp. §A; code arguments |
| λ_m (lambda_mask) | 5e-4 | supp. §A; `arguments/__init__.py` (README says 0.01: wrong, code wins) |
| mask threshold ε | 0.01 | code renderer + `densify_and_prune` |
| mask lr | 0.01 | supp. §A; code |
| mask init | 1.0 | `create_from_pcd` |
| net lr | 0.01, ×0.33 at 5k/15k/25k, linear warm-up ×0.01→1 over 100 | supp. §A; `training_setup` |
| max hashmap | 2^19 | supp. §A; code |
| hash levels / features / base res / per-level scale | 16 / 2 / 16 / 1.447 | supp. §A ("16 to 4096"); code |
| MLP | 2 hidden × 64, ReLU | supp. §A; code |
| R-VQ C / L | 64 / 6 | supp. §A; code |
| rvq_iter | 29000 (VQ on for iterations 29001–30000) | §3.4 "last 1K"; code |
| codebook Adam lr | 1e-4 | code |
| kmeans iters | 1 | code |
| 3DGS lrs | pos 1.6e-4→1.6e-6 ×extent, opacity 0.05, scale 0.005, rotation 0.001 | 3DGS defaults |
| densify | from 500, every 100, until 15000, grad 2e-4, percent_dense 0.01, opacity reset 3000, min opacity 0.005 | 3DGS defaults |
| sh_degree | 0 (no per-Gaussian SH) | `ModelParams.sh_degree = 0` |
| PP | 8-bit opacity/hash, hash prune 0.1, Huffman | §4.1, code `final_prune(compress=True)` |

## 4. Schedules
- 1–100: net lr linear warm-up. 5000 / 15000 / 25000: net lr ×0.33.
- 500 < it < 15000, it % 100 == 0: densify + prune (incl. mask). it % 3000 == 0 (< 15000): opacity reset 0.01.
- it ≥ 15000, it % 1000 == 0: mask prune.
- 29001–30000: R-VQ on scale & rotation (K-means init at 29001, in-place codebook learning).
- 30000: no optimiser step; final prune → quantise/encode → evaluate the stored representation.

## 5. Builds on
3DGS (official code base), unchanged: densification, opacity reset, position lr schedule, L1+D-SSIM, SfM init,
SH-degree schedule irrelevant (degree 0). Removed: per-Gaussian SH (features_dc/rest not present).

## 6. Protocol
Mip-NeRF 360 (9 scenes; 3DGS convention: outdoor images_4, indoor images_2), Tanks&Temples (train, truck),
Deep Blending (drjohnson, playroom); `--eval` llffhold 8; black background (`white_background=False`);
30K iterations; LPIPS-VGG (3DGS metrics.py); A100 for their runs (3DGS* retrained by the authors, same GPU).
Per-scene numbers: supp. Tables `tab:per_360`, `tab:per_td`. Storage reported in MB (10^6 bytes).

## 7. Ambiguities and choices
- G-PCC: not in this paper (extended version only). Positions fp16. → nothing to install.
- Ours vs Ours+PP: both reported; PP changes PSNR by ≤0.16 dB. Port evaluates **Ours+PP** (the decoded
  PP model) and reports both storage formulas.
- Repository CSVs (`source/code/results/`) "w/ PP" sizes are `point_cloud_pp.npz` files (Aug-2024 Morton
  sort + npz compression), smaller than the paper's formula numbers (e.g. bicycle 38.32 vs 42.42 MB); the
  paper values are the target. "w/o PP" CSV sizes equal the paper's "Ours" storage.
- tcnn runs the hash grid and MLP in fp16; precision of the trained parameters is fp16 in tcnn.
- README `lambda_mask 0.01` vs code 0.0005 vs paper 5e-4: 5e-4.
- Huffman total bits exclude the EOF symbol's code but the EOF participates in tree construction (dahuffman).
