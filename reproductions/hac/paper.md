# HAC: Hash-grid Assisted Context for 3D Gaussian Splatting Compression — implementation spec

- arXiv **2403.14530** (ECCV 2024). Yihang Chen, Qianyi Wu, Weiyao Lin, Mehrtash Harandi, Jianfei Cai.
- Sources (untrusted, never executed): `source/latex/main_arxiv.tex` (main paper and supplementary in one file),
  `source/paper.pdf` and `source/paper.txt`, and the official code `source/code/` (github.com/YihangChen-ee/HAC, commit
  `110dcf0a`, 2025-11-09). I unzipped the submodule sources only to read them, in the scratchpad.
- Citations: `code:<file>:<line>` points into `source/code/`, and `§`/`Eq.` point into the paper.
- **Where code and paper disagree, the code values are the ones to use.** Both are listed.

---------------------------------------------------------------------------------------------------
## 1. Method summary and contributions

- HAC is **Scaffold-GS plus entropy coding**. The Scaffold-GS anchors stay as they are. An anchor has a position
  `x^a`, a feature `f^a ∈ R^50`, a scaling `l ∈ R^6` and offsets `o ∈ R^{10×3}`. MLPs decode each anchor into K=10
  neural Gaussians.
- A compact **binary hash grid** is learned jointly. It is a mixed 3D + three 2D-plane grid with parameters in
  {-1,+1}. The grid is queried at each anchor position to get the hash feature `f^h` (96-D). `f^h` is used only as
  **context**: it is never fed to the renderer.
- A context MLP (`mlp_grid`) maps `f^h` to three things for every attribute dimension of f^a, l and o:
  - a Gaussian mean μ and scale σ;
  - one scalar **quantization-step refinement** r per attribute group (AQM).
  Attributes are quantized with step `q = Q0·(1+tanh r)`. Their bits are estimated as `-log2` of the Gaussian
  probability mass on `[x̂-q/2, x̂+q/2]`, and that estimate becomes a rate loss.
- **Adaptive offset masking**: a learnable, STE-binarized mask on each of the 10 offsets per anchor, with a mask loss
  as in Lee et al. 2024. An anchor whose offsets are all masked is dropped at encoding time.
- **Real bitstream at test time**: f^a, l and o are arithmetic-coded with the predicted Gaussians. The hash grid and
  the masks are arithmetic-coded as Bernoulli symbols with their empirical frequency. Anchor xyz is stored as 16-bit
  min-max quantized values and the MLPs as fp32. Rendering metrics are computed on the **decoded** model.

Contributions:
- **C1 (§3.2, §3.3 "Hash Grid Compression")**: a structured binary hash grid used as context for the unorganized
  anchors, giving `p(A|f^h)·p(H)` (Eq. 3). The grid is binarized by STE and its rate is estimated from the frequency
  of +1 values (L_hash).
- **C2a (§3.3 "Adaptive Quantization Module")**: AQM, a per-anchor quantization step `q_i = Q0(1+tanh r_i)` with
  `r_i = MLP_q(f^h_i)`. Training uses uniform-noise quantization and testing uses rounding.
- **C2b (§3.3 "Gaussian Distribution Modeling")**: Gaussian entropy model with `(μ_i, σ_i) = MLP_c(f^h_i)` and the
  entropy loss L_entropy.
- **C2c (§3.3 "Adaptive Offset Masking")**: STE binary offset masks with mask loss L_m, plus anchor pruning when all
  offsets of an anchor are masked.
- **C3 (§4, supp. A)**: a staged training schedule and a real entropy-coded bitstream. The paper reports about 75×
  smaller than 3DGS and about 11× smaller than Scaffold-GS, at the same or better fidelity.

---------------------------------------------------------------------------------------------------
## 2. Equations (verbatim from LaTeX, numbers as in the paper)

Eq. (1)–(2) are standard 3DGS: `G(x)=exp(-½(x-μ)^T Σ^{-1}(x-μ))`, `Σ=RSS^TR^T`, `C=Σ_i c_i α_i Π_{j<i}(1-α_j)`.

Scaffold-GS anchor (§3.1):
`$\mathcal{A}=\{\bm{f}^a\in\mathbb{R}^{D^a}, \bm{l}\in\mathbb{R}^6, \bm{o}\in\mathbb{R}^{3K}\}$`

Hash feature: `$\bm{f}^h:= {\rm Interp}(\bm{x}^a, \mathcal{H})$`, where
`$\mathcal{H}=\{\bm{\theta}_i^l\in\mathbb{R}^{D^h}|i=1,\dots,T^l|l=1,\dots,L\}$`.

Eq. (3), context model:
```latex
p(\mathcal{A}, \bm{x}^a, \mathcal{H}) = p(\mathcal{A}|\bm{x}^a, \mathcal{H})\times p(\bm{x}^a, \mathcal{H}) \sim p(\mathcal{A}|\bm{f}^h)\times p(\mathcal{H})
```

Eq. (4), quantization. Here `f_i ∈ {f^a_i, l_i, o_i}`:
```latex
\hat{\bm{f}_i} = \bm{f}_i + \mathcal{U}\left(-\frac{1}{2}, \frac{1}{2}\right)\times \bm{q}_i \quad \text{(training)};\qquad
\hat{\bm{f}_i} = {\rm Round}(\bm{f}_i / \bm{q}_i)\times \bm{q}_i \quad \text{(testing)}
```

Eq. (5), AQM:
```latex
\bm{q}_i = Q_0\times\left(1+{\rm Tanh}\left({\bm{r}}_i\right)\right), \qquad {\bm{r}}_i = {\rm MLP_q}\left(\bm{f}^h_i\right)
```
`r_i ∈ R^1` is one scalar per anchor **per attribute group** (code: 3 scalars, `Q_feat_adj`, `Q_scaling_adj` and
`Q_offsets_adj`), so `q_i ∈ (0, 2Q0)`.

Eq. (6), probability:
```latex
p(\hat{\bm{f}}_i) = \Phi_{\bm{\mu}_i, \bm{\sigma}_i}\left(\hat{\bm{f}}_i+\frac{1}{2}\bm{q}_i\right) - \Phi_{\bm{\mu}_i, \bm{\sigma}_i}\left(\hat{\bm{f}}_i-\frac{1}{2}\bm{q}_i\right), \qquad
\bm{\mu}_i, \bm{\sigma}_i = {\rm MLP_c}\left(\bm{f}^h_i\right)
```

Eq. (7), entropy loss:
```latex
L_{\text{entropy}} = \sum_{\bm{f}\in\{\bm{f}^a, \bm{l}, \bm{o}\}} \sum_{i=1}^N \sum_{j=1}^D \left(-\log_2 p(\hat{{f}}_{i,j})\right)
```

Eq. (8), hash rate. `h_f` is the frequency of +1, and `M_+` and `M_-` are the counts of +1 and -1:
```latex
L_{\text{hash}} = M_+\times\left(-\log_2(h_f)\right) + M_-\times\left(-\log_2(1-h_f)\right)
```

Eq. (9), total loss:
```latex
Loss = L_{\text{Scaffold}} + \lambda_e\frac{1}{N(D^a+6+3K)}(L_{\text{entropy}} + L_{\text{hash}}) + \lambda_m L_m
```

What the code computes (`code:train.py:190-202`, `code:gaussian_renderer/__init__.py:60-96`):
```
L_Scaffold = 0.8*L1 + 0.2*(1-SSIM) + 0.01 * mean_over_rendered_gaussians( prod(scaling_gaussian, dim=1) )
if step > 10000:
   bit_per_param = (Σbits_feat + Σbits_scaling + Σ(bits_offsets*mask)) / (numel_feat+numel_scaling+numel_offsets)
                   * mask_anchor_rate                    # over a ~5% random sample of visible, unmasked anchors
   loss += lmbda * (bit_per_param + bit_hash_grid / (N_anchor_total*(50+6+30)))
   loss += 5e-4 * mean(sigmoid(_mask))                   # L_m with λ_m = 5e-4
```
- `mask_anchor_rate` is the fraction of visible anchors that still have at least one unmasked offset. The per-param
  mean is therefore taken over the sampled set and then scaled by the valid-anchor fraction. This approximates Eq. 9's
  `L_entropy / (N(D^a+6+3K))`.
- Offset bits are multiplied by the binary mask, so masked offsets cost 0 bits.
- `bit_hash_grid` is Eq. 8 + 32 bits. It is computed on `(STE_binary(params)+1)/2` (`code:utils/encodings.py:16-33`).
  The gradient reaches the hash params through the STE.

Entropy model details (`code:utils/entropy_models.py:30-50`):
- `bits = -log2( clamp_low(|Φ(x+Q/2) - Φ(x-Q/2)|, 1e-6) )`. `Low_bound` passes the gradient through only where
  `x ≥ 1e-6` or `g < 0`.
- `scale` is the **raw** MLP output clamped to ≥1e-9. There is no softplus or exp.
- Before the CDF, x is clamped to `mean(x) ± 15000·Q`. The mean is passed in as `_anchor_feat.mean()`,
  `get_scaling.mean()` or `_offset.mean()`.
- During training x is the noise-added value, not a rounded one. The Gaussian entropy model is the only one in use:
  `Entropy_factorized`, `Entropy_bernoulli` and the `*_gaussian_mixed` codecs exist in the files but are unused (the
  mixed codecs are leftovers from HAC++).

Mask (`code:scene/gaussian_model.py:350-366`):
- `mask = STE( sigmoid(_mask) > 0.01 )`, with the hard value in the forward pass and the sigmoid gradient in the
  backward pass.
- `_mask` has shape `[N,10,1]` and is initialized to 1.0, so `sigmoid(1)` ≈ 0.73.
- `neural_opacity *= mask` (`code:gaussian_renderer/__init__.py:136`). The mask is active from step 0, but `L_m` is
  added only after step 10000.
- `mask_anchor = any(mask)`.

Hash grid binarization (`code:utils/encodings.py:35-52`):
- Forward pass: `sign(clamp(p,-1,1))`, with 0 mapped to +1.
- Backward pass: the gradient passes where |p| ≤ 1.
- Params are initialized U(-1e-4, 1e-4).

Anchor xyz quantization (`code:utils/encodings.py:73-89`):
- `get_anchor` is **always** 16-bit min-max quantized inside `[x_bound_min, x_bound_max]` using STE:
  `q = floor((x-min)/((max-min)/(2^16-1)+1e-6))`, clamped to [0, 65535].
- The bounds come from `update_anchor_bound()`, which sets min/max over anchors and then pads them outward: ×1.2 when
  the value has the outward sign, ×0.8 otherwise (`code:scene/gaussian_model.py:409-418`).
- The bounds are computed at init and recomputed once at step 10000. They also normalize positions to [0,1] for the
  hash grid.

---------------------------------------------------------------------------------------------------
## 3. Hyper-parameters

### 3a. Scaffold-GS backbone as configured in HAC's code
| name | value | source |
|---|---|---|
| anchor feature dim D^a (`feat_dim`) | **50** (Scaffold-GS default is 32) | §4.1; `code:arguments/__init__.py:50` |
| offsets per anchor K (`n_offsets`) | **10** | `code:arguments/__init__.py:51` |
| feature bank | **disabled** (`use_feat_bank=False`) | §4.1; `code:arguments/__init__.py:57` |
| anchor fields | `_anchor [N,3]` (fixed, lr 0), `_offset [N,10,3]` (init 0), `_mask [N,10,1]` (init 1.0), `_anchor_feat [N,50]` (init 0), `_scaling [N,6]` (log, init `log(sqrt(knn_dist2))` repeated ×6), `_rotation [N,4]` and `_opacity [N,1]` (present, `requires_grad=False`, unused) | `code:scene/gaussian_model.py:464-485` |
| anchor init | SfM points voxelized: `unique(round(p/voxel_size))*voxel_size` | `code:scene/gaussian_model.py:442-463` |
| `voxel_size` | default 0.001. Run scripts: **mip360 0.001**, **tandt 0.01**, **db 0.005**, synthetic 0.001, bungee 0 (0 means median 1-NN distance) | `code:run_shell_*.py` |
| `update_init_factor` | 16 (mip360/tandt/db), 4 (synthetic), 128 (bungee) | run scripts |
| `update_depth`, `update_hierachy_factor` | 3, 4 | `code:arguments/__init__.py:53-55` |
| view input to MLPs | `[f^a (50), ob_view = (x^a - cam_center)/‖·‖ (3), ob_dist (1)]` = 54-D | `code:gaussian_renderer/__init__.py:115-132` |
| `mlp_opacity` | Linear(54,50)-ReLU-Linear(50,10)-**Tanh**. Gaussians with opacity > 0 are kept after the mask multiply | `code:scene/gaussian_model.py:235-240` |
| `mlp_cov` | Linear(54,50)-ReLU-Linear(50,70), no activation. scale = `l[3:6] * sigmoid(s[:3])`, rot = normalize(s[3:7]) | `:242-247`, renderer `:162-164` |
| `mlp_color` | Linear(54,50)-ReLU-Linear(50,30)-**Sigmoid**, given as `colors_precomp` (no SH) | `:249-254` |
| Gaussian position | `x^a + o_k * l[0:3]` | renderer `:166-167` |
| scaling as stored or coded | `get_scaling = exp(_scaling)`, i.e. the activated value is the one quantized and entropy-coded | `:344-348` |
| photometric loss | 0.8·L1 + 0.2·D-SSIM (11×11 Gaussian window, σ=1.5, same padding) | `code:train.py:191-195` |
| scaling regularizer | 0.01 · mean(prod(gaussian scales)) | `code:train.py:194-195` |
| background | black (`white_background=False`) | `code:arguments/__init__.py:62` |
| prefilter | anchors culled by frustum with `rasterizer.visible_filter` (Scaffold-GS rasterizer fork) before decoding | renderer `:250-305` |

### 3b. Anchor growing and pruning (Scaffold-GS logic, `code:scene/gaussian_model.py:749-959`, `code:train.py:228-245`)
| name | value |
|---|---|
| start of statistics accumulation (`start_stat`) | step > 500 |
| growing/pruning window | step > 1500 (`update_from`), every 100 (`update_interval`), while step < 15000 (`update_until`). **Paused for steps 3000–3999** (`iteration not in range(3000,4000)`). Statistics keep accumulating during the pause |
| gradient stat | per neural Gaussian, `‖∇ screen-space xy‖`, averaged over `offset_denom`. Only offsets with `offset_denom > 100*0.8*0.5 = 40` are eligible |
| growing | for i in 0..2: threshold `0.0002 * 2^i` (`(hier_factor//2)^i`); keep with probability `rand > 0.5^(i+1)`; candidate voxel size `voxel_size * 16 // 4^i` (16→4→1 × voxel_size for mip360/tandt/db); new anchors at voxel centers not already occupied |
| new-anchor init | `offset=0`, `mask=1`, feat = **scatter_max** of the parents' features in the voxel, `scaling = log(cur_size)` (×6) |
| pruning | anchors with `opacity_accum < 0.005 * anchor_demon` and `anchor_demon > 80`. `opacity_accum` sums the clamped-≥0 neural opacities of the K offsets. Counters reset for anchors with demon > 80 |
| quirk carried from Scaffold-GS | on every prune, raw `_scaling[:,3:]` is clamped to ≤ 0.05 in log space (`:782-795`) |
| no opacity reset, no 3DGS-style densification | — |

### 3c. Learning rates
Adam with eps=1e-15. Schedules use `get_expon_lr_func`: log-linear from init to final over 30k steps, with
`lr_delay_steps=0`, so `delay_mult` has no effect. Source: `code:arguments/__init__.py:81-139` and
`code:scene/gaussian_model.py:519-622`.

| group | init → final | note |
|---|---|---|
| anchor xyz | **0 → 0** | anchors never move |
| offset | 0.01 → 0.0001 | × `spatial_lr_scale` (3DGS camera-extent radius) |
| mask | 0.01 → 0.0001 | × `spatial_lr_scale` |
| anchor_feat | 0.0075 | constant |
| scaling | 0.007 | constant |
| opacity / rotation | 0.02 / 0.002 | params frozen (`requires_grad=False`) |
| mlp_opacity | 0.002 → 0.00002 | |
| mlp_cov | 0.004 → 0.004 | |
| mlp_color | 0.008 → 0.00005 | |
| encoding_xyz (hash grid) | 0.005 → 0.00001 | `step_sub=0` because `ste_binary=True`, so the decay runs from step 0 even though the grid is only used after 10k |
| mlp_grid (context MLP) | 0.005 → 0.00001 | same as above |
| mlp_deform | 0.005 → 0.0005 | **dead code**: built and optimized but never called, and excluded from the size count |

### 3d. HAC-specific
| name | value | source |
|---|---|---|
| hash grid | 3D grid with 12 levels and 3 2D grids (xy, xz, yz) with 4 levels each. `n_features_per_level = D^h = 4` | §4.1; `code:scene/gaussian_model.py:41-107,136-137` |
| 3D resolutions | code list `(18,24,33,44,59,80,108,148,201,275,376,514)`. The kernel uses `pos = x*(res-2)+0.5`, so the effective resolution is 16…512, matching the paper's "16 to 512" | `code:scene/gaussian_model.py:136`; `gridencoder.cu:183` |
| 2D resolutions | `(130,258,514,1026)`, effective 128…1024 | `code:scene/gaussian_model.py:137` |
| table size | 3D `2^13` (`--log2 13`), 2D `2^15` (`--log2_2D 15`) per level. Entries per level = `min(2^T, res^dim)` rounded up to a multiple of 8. Indexing is dense when res^dim fits the table; otherwise the instant-NGP spatial hash with primes 1, 2654435761, 805459861 is used. Interpolation is (bi/tri)linear | `code:train.py:603-605`; `code:utils/encodings.py:250-268`; gridencoder.cu |
| f^h dimension | 12·4 + 3·4·4 = **96** | |
| hash input | `(x^a - x_bound_min)/(x_bound_max - x_bound_min)` with the quantized anchor | `code:scene/gaussian_model.py:420-426` |
| context MLP `mlp_grid` (= MLP_q + MLP_c combined) | Linear(96,100)-ReLU-Linear(100,175). The output splits as `[μ_feat 50, σ_feat 50, μ_scal 6, σ_scal 6, μ_off 30, σ_off 30, r_feat 1, r_scal 1, r_off 1]`. The paper says "3-layer MLP with ReLU"; the code has 2 Linear layers | §4.1; `code:scene/gaussian_model.py:256-260,1040-1041` |
| Q0 | f^a: **1**, l: **0.001**, o: **0.2** | §4.1; `code:gaussian_renderer/__init__.py:47-49` |
| λ_e (`--lmbda`) | {0.004, 0.003, 0.002, 0.001, 0.0005}. The run scripts default to 0.004. Table 1 uses lowrate 0.004 and highrate 0.001 for real scenes, and 0.002 / 0.0005 for Synthetic-NeRF | §4.1; supp. Tables D–H; `code:run_shell_*.py`; train.py default 0.001 |
| λ_m | 5e-4 on `mean(sigmoid(_mask))` | §4.1; `code:train.py:202` |
| mask threshold | sigmoid > 0.01 | `code:scene/gaussian_model.py:355` |
| entropy sampling ratio | 5% of visible anchors per iteration (`rand <= 0.05`), restricted to anchors where `mask_anchor` is true | supp. A.2; renderer `:73-74` |
| anchor xyz coding | 16 bits/coordinate (`anchor_round_digits=16`) | `code:utils/encodings.py:11-12` |
| MLP storage | 32 bits per param of mlp_opacity, mlp_cov, mlp_color and mlp_grid (`'mlp' in name and 'deform' not in name`) | `code:scene/gaussian_model.py:285-290` |
| iterations | 30,000 | run scripts |

---------------------------------------------------------------------------------------------------
## 4. Schedules (the `step` passed to `render`, 1-based)
| steps | what happens | source |
|---|---|---|
| 1–3000 | Plain Scaffold-GS: no quantization and no rate loss. Masks already multiply opacity (STE threshold, but no L_m). Anchor xyz is 16-bit quantized using the init bounds | supp. A.1; renderer `:50-56` |
| 501+ | accumulate growing/pruning statistics | train.py:229 |
| 1600–2900, every 100 | grow and prune anchors | |
| **3001–10000** | Uniform noise `U(-½,½)·Q0` is added to f^a, the activated scaling and the offsets (fixed Q0, no r, no hash grid). There is no rate loss in this phase | supp. A.1; renderer `:51-55` |
| 3000–3999 | **anchor spawning and pruning paused** | supp. A.1; train.py:237 |
| 4000–14900, every 100 | grow and prune again | |
| **10000** | `update_anchor_bound()` recomputes the hash-grid and xyz-quantization bounds from the current anchors | supp. A.1; renderer `:57-58` |
| **10001–30000** | Full HAC: hash grid, context MLP, AQM noise with step q, entropy loss on a 5% sample, hash rate loss and mask loss | renderer `:60-96`; train.py:197-202 |
| 15000 | stop growing/pruning (statistics buffers deleted) | train.py:241-245 |
| 30000 | Run `estimate_final_bits`, then **real encode → decode**, swap the decoded params into the model, save, render the test set from the decoded model, and evaluate | train.py:294-308, 658-675 |

The "step flag" behaviour comes entirely from the `step` checks in `generate_neural_gaussians`: `>3000 and <=10000`,
`==10000` and `>10000`. No other flag exists. Test-time rendering, when not decoded, applies
`STE_multistep(round(x/q)*q)` with q from the context MLP.

---------------------------------------------------------------------------------------------------
## 5. Bitstream and size measurement (what "size" means in the tables)
The encoder is `conduct_encoding` (`code:scene/gaussian_model.py:1080-1199`).

1. Keep only anchors with `mask_anchor` true (anchors whose 10 masks are all 0 are dropped).
2. **Anchor xyz**: uint16 quantized values saved with `np.save`. The size is counted as `N*3*16` bits, not as the
   file size.
3. Process the anchors in chunks of 3000. For each chunk, compute `f^h`, then mlp_grid gives μ, σ (clamped ≥ 1e-9)
   and q. For f^a (50/anchor), the scaling (6/anchor, activated) and the **unmasked** offsets only, round
   `x/q` (with the ±15000q clamp) and **arithmetic-code** the symbols with the per-symbol Gaussian CDF table:
   `lower[i] = Φ((min+i-0.5)·q)` for i in `[min, max+1]`.
   - The coder is the custom CUDA extension `arithmetic` (`arithmetic.calculate_cdf`, `arithmetic_encode` and
     `arithmetic_decode`). It uses 16-bit precision with independent GPU chunks of 10000 symbols.
   - Each file stores min, max, a length (3×32 bits) and per-chunk byte counts.
   - Bits = `(len(bytes)+len(cnt_bytes))*8 + 96`, so the real coded size including headers is counted.
4. **Hash grid**: all binary params (3D + 3×2D, `{0,1}`) are arithmetic-coded with a single Bernoulli p equal to the
   fraction of 1s, plus 64 header bits.
5. **Masks** `[N,10,1]`: arithmetic-coded the same way, with a single Bernoulli p.
6. **MLPs**: number of params × 32 bits.
7. The reported **Total = anchor + feat + scaling + offsets + hash + masks + MLPs**, in MB = bits/(8·1024·1024)
   (MiB). The repo's `results/*.csv` give Size [Bytes]; for example, bicycle highrate is 41048080 B = 39.15 MiB, which
   matches the supplementary value of 39.15.

Further notes:
- Decoding rebuilds everything from the bitstreams (`conduct_decoding`, `:1201-1343`). The model then switches to
  `decoded_version=True`, and the reported PSNR, SSIM and LPIPS are on the decoded model.
- The paper itself used `torchac` as the AE. The README says a "cuda-based codec" replaced torchac on 2024-08-08, which
  made coding about 10× faster; sizes should be essentially unchanged.
- `estimate_final_bits` (`:1023-1078`) logs the differentiable estimate for comparison.
- Paper bit allocation (BungeeNeRF, λ=4e-3), in MB: A 14.90 (f^a 8.76, l 2.52, o 3.62), hash 0.15, masks 0.52,
  anchors 2.77, MLPs 0.16.
- Per-param bits (supp. Table A): BungeeNeRF f^a 3.03 / l 7.27 / o 4.56; Synthetic f^a 1.33 / l 3.58 / o 3.76.

---------------------------------------------------------------------------------------------------
## 6. What it builds on, and CUDA dependencies
- **Base: Scaffold-GS**, with these changes:
  - D^a = 50 and no feature bank;
  - masks multiply the neural opacity;
  - activated scaling is the coded quantity;
  - anchor xyz is 16-bit quantized;
  - the 3000–4000 spawning pause is added;
  - the rate, hash and mask losses are added.
- **Kept unchanged from Scaffold-GS**: the MLP decoders, the view-dependent input, anchor growing and pruning, the
  loss with the scaling regularizer, LR schedules and the voxel sizes. Nothing from 3DGS densification is used.
- **CUDA extensions the official code needs.** All four are zips in `submodules/`, built via pip from
  `environment.yml` (py3.7, torch 1.12.1, cudatoolkit 11.6):
  1. `diff-gaussian-rasterization`: the Scaffold-GS fork, which adds `GaussianRasterizer.visible_filter` (anchor
     frustum prefilter) and takes `colors_precomp`.
  2. `simple-knn` (`distCUDA2`): 3-NN distances for the initial anchor scaling.
  3. `gridencoder` (`_gridencoder`): a torch-ngp-derived hash grid with an explicit resolution list, supporting 2D and
     3D grids and binary embeddings.
  4. `arithmetic`: a custom CUDA arithmetic coder, plus `calculate_cdf`.
  - Also required: `torch_scatter` (`scatter_max` for new-anchor features), `einops`, `plyfile` and `lpips`.
  - **No tiny-cuda-nn is used.**
- Our environment already has gsplat for rasterization, so a port needs the following:
  - a pure-PyTorch (or JIT) multi-resolution hash grid that takes the explicit resolution lists, applies the `res-2`
    scaling and supports 2D planes;
  - an arithmetic coder. Python `torchac` or `constriction` would do, or the size can be measured as the sum of
    `-log2 p` under the quantized CDF. The real coder adds only about 0.1% overhead plus headers, and this must be
    labeled.

---------------------------------------------------------------------------------------------------
## 7. Protocol
- **Datasets** (Table 1):
  - Synthetic-NeRF (8 scenes);
  - Mip-NeRF360, **all 9 scenes** (bicycle garden stump room counter kitchen bonsai flowers treehill);
  - Tanks&Temples (truck, train);
  - DeepBlending (playroom, drjohnson);
  - BungeeNeRF (6 scenes).
- **Split**: `--eval`, llffhold=8 on name-sorted images (`idx % 8 == 0` is test), with `--lod 0`. BungeeNeRF uses
  `--lod 30`, which takes the first 31 images as test.
- **Resolution**:
  - The run scripts pass no `-r` and no `-i`, so the loader reads `images/` with `--resolution -1`. Any image wider
    than 1600 px is rescaled to **width 1600** (`code:utils/camera_utils.py:25-34`).
  - Mip-NeRF360 is therefore at ~1.6K for both outdoor and indoor scenes. **This differs from the 3DGS
    images_4/images_2 protocol** used by our runtime.
  - T&T and DB from the 3DGS `tandt_db.zip` are below 1600 px wide, so they run at native resolution.
- **Iterations**: 30k. Black background. Training views are sampled from a shuffled stack without replacement.
- **Metrics**:
  - Test images are rendered from the **decoded** model, saved as PNGs and re-read.
  - PSNR on [0,1].
  - SSIM: 3DGS implementation (11×11 Gaussian window, same padding).
  - **LPIPS-VGG** from `lpipsPyTorch` (3DGS's copy).
  - Per-scene metrics are averaged over views.
  - FPS is also logged.
- **Seeds**: `safe_state` seeds torch, numpy and random with 0 (3DGS convention). There is one run per setting and no
  multi-seed averaging.
- **Baselines in the paper**:
  - The 3DGS and Scaffold-GS per-scene rows in supp. Tables G (3DGS) and H (Scaffold-GS) of the PDF.
  - The 3DGS numbers are **not** the 3DGS paper's own. Mip-NeRF360 averages 27.49 here vs 27.21 in the 3DGS paper,
    and T&T 23.69 vs 23.14. Their protocol is an **unverified assumption**: they may come from the Scaffold-GS paper (Scaffold-GS sizes are rounded integers); that they share HAC's 1.6K protocol is not stated.
  - Judge fidelity against HAC's own rows and our own baselines on the same images; deltas vs these rows are indicative only.
- **Paper averages**. Table 1 values, also all in `paper_results.json`:

| dataset | Ours-lowrate (PSNR / SSIM / LPIPS / MB) | Ours-highrate (PSNR / SSIM / LPIPS / MB) |
|---|---|---|
| Mip360 | 27.53 / 0.807 / 0.238 / 15.26 (λ 0.004) | 27.77 / 0.811 / 0.230 / 21.87 (λ 0.001) |
| T&T | 24.04 / 0.846 / 0.187 / 8.10 | 24.40 / 0.853 / 0.177 / 11.24 |
| DB | 29.98 / 0.902 / 0.269 / 4.35 | 30.34 / 0.906 / 0.258 / 6.35 |

  Scaffold-GS rows (same order): Mip360 27.50 / 0.806 / 0.252 / 253.9; T&T 23.96 / 0.853 / 0.177 / 86.50;
  DB 30.21 / 0.906 / 0.254 / 66.00.
- **Training time** (paper, RTX 4090): about 1.9× Scaffold-GS, e.g. 27.6 min on BungeeNeRF and 9.0 min on Synthetic.

---------------------------------------------------------------------------------------------------
## 8. Ambiguities and choices made by the code
1. **Scaling `l` activation**: the paper footnote defines l *after Sigmoid*. The code quantizes, codes and renders
   `exp(_scaling)`, and Q0=0.001 applies to that exp value. Follow the code.
2. **MLP depth**: the paper says the combined MLP_q/MLP_c is a "3-layer MLP". The code is
   Linear(96→100)-ReLU-Linear(100→175), i.e. 2 Linear layers (3 layers if you count input, hidden and output).
3. **σ parameterization**: raw linear output clamped ≥ 1e-9, with no softplus. Negative raw values become 1e-9 and
   give a ~1e-6 likelihood floor, which keeps the gradients meaningful.
4. **Rate normalization**: the code computes per-param bits on a 5% sample and multiplies by the valid-anchor rate. The
   hash bits are divided by the **total** anchor count × 86. This approximates Eq. 9.
5. **λ_m mask loss**: `5e-4 * mean(sigmoid(_mask))`, added only after step 10000. Lee et al.'s L_m is the same mean
   of sigmoid. The mask is applied to opacity from step 1.
6. **Hash grid "resolutions 16–512"**: the code lists 18…514 and the kernel subtracts 2. The 3D levels are spaced
   approximately geometrically (×~1.36 per level).
7. **Hash bounds**: computed at init and again at step 10000. After that the bounds stay fixed even though anchors
   keep growing until step 15000, so new anchors outside the padded box get clamped by the 16-bit quantizer and the
   grid input. The ±20% padding exists for this case.
8. **Noise before step 10000** uses the fixed Q0 without the hash grid and adds **no rate loss** (the loss requires
   `bit_per_param`).
9. **Q refinement shape**: one scalar r per anchor per attribute group (3 per anchor), not one per dimension.
10. **Anchor pruning by mask** happens only at encode time. During training, fully masked anchors still exist; they
    contribute nothing to rendering because their opacities are zeroed.
11. **Mip-NeRF360 resolution**: 1.6K width in the official code vs images_4/images_2 in our protocol. Expect absolute
    PSNR and size differences, especially on outdoor scenes. Whether the paper's 3DGS and Scaffold-GS rows
    use the same 1.6K protocol is an unverified assumption.
12. **Entropy codec**: torchac in the paper vs the custom CUDA AE in the current repo. The size definition is the same
    (real coded bytes plus small headers).
13. `mlp_deform`, `Entropy_factorized` and the mixed-Gaussian codecs are unused leftovers. Ignore them.
14. The paper reports **Synthetic-NeRF lowrate as λ=0.002** while the real-scene datasets use 0.004. This is confirmed
    by matching the Table 1 averages against supp. Table B (PDF label).
15. The supplementary's Mip-NeRF360 λ=0.0005 PSNR average is 27.83, while the per-scene mean is 27.837. This is a
    rounding difference only.
