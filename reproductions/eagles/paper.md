# EAGLES: Efficient Accelerated 3D Gaussians with Lightweight EncodingS — implementation spec

- arXiv: 2312.04564 (ECCV 2024). Sources: `source/latex/main_arxiv_eccv.tex`, `source/latex/arxiv_suppl_eccv.tex`.
- Official code: https://github.com/Sharath-girish/efficientgaussian, commit
  `b9b50db075ca14e797a557300d66f52cc3fef7e5`, cloned to `source/code/` (Inria 3DGS licence:
  non-commercial research use). Run config = `configs/efficient-3dgs.yaml` (README command).

## 1. Method summary
- Rotation (4), SH-rest (45) and opacity (1) of every Gaussian are stored as **integer latents**
  (dims 8, 16, 1) decoded by tiny per-attribute **linear decoders** shared by all Gaussians;
  continuous latents are trained with a straight-through rounding estimator. Position, scale and
  SH DC are not quantized (fp32).
- After training the rounded latents are **entropy coded** (per-latent-dimension empirical
  probabilities) and stored with the decoders.
- **Progressive (coarse-to-fine) training**: renders and targets are downsampled by a factor that
  grows from 0.3 to 1.0 on a cosine schedule over the first 70% of iterations.
- **Controlled densification + influence pruning**: densification interval 125 (code; paper says
  175), every 5100 steps the 15% Gaussians with lowest accumulated influence sum_p alpha*T are pruned.

Contributions:
- C1 Attribute quantization with learned decoders + entropy coding (§3.1 "Attribute quantization",
  supp. §1). Includes opacity quantization (§3.1 "Opacity quantization for improved optimization").
- C2 Progressive training (§3.2).
- C3 Influence pruning (§3.3) with controlled densification (§4.1, ablation Table 3).

## 2. Equations
Eq. (STE decoding, §3.1):
$$\boldsymbol{a} = D(STE(\widehat{\boldsymbol{q}}))$$
STE = round in forward, identity gradient (`compress/decoders.py:StraightThrough`).
Decoder for all three attributes is a single `DecoderLayer` (`num_layers_dec = 0`, `use_shift=1`):
- `ldecode_matrix='sq'` (rotation, opacity): $a = \mathrm{round}(\hat q)\,S + b$, $S\in\mathbb{R}^{l\times k}$, $b\in\mathbb{R}^{1\times k}$.
- `ldecode_matrix='dft'` (SH rest): $a = (\mathrm{round}(\hat q)\,F)\odot s + b$, with fixed
  $F_{ij} = \cos(\pi/k\,(i+\tfrac12)\,j)/\sqrt{k}\cdot(\sqrt2 \text{ if } j>0)$ ($16\times45$, DCT-II basis, not trained),
  learned $s\in\mathbb{R}^{1\times45}$, $b\in\mathbb{R}^{1\times45}$.
- `div` normalisation is the identity (`latent_norm='none'` → div = 1).
- Decoded rotation is normalised by the rasterizer (`F.normalize`); decoded opacity is a logit
  (`sigmoid` activation); decoded SH rest is reshaped `[N,15,3]` (coefficient-major, channel-minor).

Latent init (supp. Eq. 1): $\widehat{\boldsymbol q} = D^{-1}(\boldsymbol a)$; linear decoder →
least squares `torch.linalg.lstsq(S.T, (a-b).T).solution.T`; 1×1 decoder →
$(a-b)/(S+10^{-6})$. SH-rest latents are initialised to **zeros** (code). Rotation target = identity
quaternion, opacity target = logit(0.1).

Latent LR (supp. §1 + `update_learning_rate`): for `latent_scale_norm='div'` (rotation, SH rest)
$\mathrm{lr} = \mathrm{lr}_{3DGS}\cdot \mathrm{lr\_scaling} / \lVert S\rVert_F$ (Frobenius norm of
the decoder `scale`, recomputed every iteration); opacity (`'none'`) uses
$\mathrm{lr}_{3DGS}\cdot\mathrm{lr\_scaling}$.

Influence (Eq. 3, §3.3):
$$W_{i,p} = \alpha_iT_i = \alpha_i\prod_{j=1}^{i-1}(1-\alpha_j), \quad W_i = \sum_p W_{i,p}$$
accumulated over training iterations; every `infl_prune_interval` prune
`W <= quantile(W, quantile_threshold)` and reset W to 0. Official CUDA: `forward.cu:364`
`out_infl[id] += alpha * T` (non-atomic). New Gaussians from clone/split start with W = 0
(`densification_postfix`).

Progressive schedule (`DecayScheduler('cosine')`, `downsample_image`):
$s(t) = 1 + \tfrac12(s_0 - 1)(1+\cos(\pi t/T_r))$ for $t\le T_r$, else 1;
$T_r = \lfloor 0.7\,(30000+1)\rfloor = 21000$, $s_0 = 0.3$. Target:
`F.interpolate(gt, scale_factor=s, mode='bilinear', align_corners=False)`; render at the
downsampled image size with the same FoV.

Opacity reset (`reset_opacity`): new logit = logit(min(sigmoid(D(q)), 0.01)); latent =
$D^{-1}$(new logit); Adam moments of the opacity latent zeroed.

Size (`GaussianModelSQ.size`, logged as "Size (MB)" = bits/8/1e6):
$$\text{bits} = \sum_{\text{decoders}}32\,|\theta| + \sum_{\text{unquantized}} 32\,N d + \sum_{\text{attr}}\sum_{\text{dim}}\sum_{v} c_v\,\mathrm{clamp}(-\log_2(c_v/N + 10^{-10}), 0, 1000)$$
(decoder parameter count includes the fixed DFT matrix and the `div` vector; unquantized =
xyz, f_dc, scaling). The on-disk pkl uses torchac arithmetic coding with one flattened
per-attribute CDF (values with prob ≤ 1e-4 moved to the mode, positions stored separately).

## 3. Hyper-parameters
| name | value | source |
|---|---|---|
| iterations | 30000 | paper §4.1, code |
| SH rest latent dim / decoder | 16 / dft | supp. Tab. 1, `efficient-3dgs.yaml` |
| rotation latent dim / decoder | 8 / sq | supp. Tab. 1, yaml |
| opacity latent dim / decoder | 1 / sq | supp. Tab. 1, yaml |
| decoder init std (shN, rot, opa) | 0.0005, 0.01, 0.5 | supp. Tab. 1, yaml |
| decoder LR (shN, rot, opa) | **1e-4 all** (yaml lists 0.005/0.01/0.3 under `quantize_params`, never read: only `opt_params` feeds OptimizationParams) | supp. Tab. 1 = effective code default |
| latent LR scale (shN, rot, opa) | **1.0** (yaml's 0.1 likewise ignored) | supp. Tab. 1 = effective code default |
| latent scale norm | div, div, none | yaml |
| base LRs | xyz 1.6e-4·extent→1.6e-6, f_dc 0.0025, f_rest 0.0025/20, opa 0.05, scale 0.005, rot 0.001 | `arguments/__init__.py` |
| Adam eps | 1e-15 (single Adam over latents + decoders) | `training_setup` |
| resize_scale / period / transform | 0.3 / 0.7 / downsample | paper §4.1, yaml |
| densify from / until / interval | 500 / **18000** / **125** (paper: 175) | yaml |
| densify grad threshold | 0.0002 | default |
| opacity reset interval | 2500 (until 18000) | paper, yaml |
| influence prune interval / until / quantile | **5100 / 27000 / 0.15** (paper: 5000 / 25000 / 15%) | yaml |
| min opacity prune | 0.005 | `densify_and_prune` call |
| SH degree | 3 (one band / 1000 its) | paper |
| lambda_dssim | 0.2 | default |

## 4. Schedules
- s(t): cosine 0.3 → 1.0 over steps 1..21000, full resolution afterwards.
- Densify every 125 steps for 500 < t < 18000; opacity reset at multiples of 2500 below 18000.
- Influence accumulated every step t < 27000; prune at t ∈ {5100, 10200, 15300, 20400, 25500}.
- Latent LR of rotation / SH rest recomputed every step from the decoder norm.

## 5. Builds on
Official 3DGS code (photometric loss, densification, SfM init, exponential xyz LR, SH schedule)
unchanged apart from the above. Final model = iteration 30000 (`search_best_iters = 0` → the
"best train PSNR" checkpoint is evaluated only at the last iteration).

## 6. Protocol
Mip-NeRF 360 (9 scenes, images_4 outdoor / images_2 indoor per 3DGS), Tanks&Temples (truck,
train), Deep Blending (drjohnson, playroom); llffhold 8; black background (3DGS default);
metrics PSNR/SSIM/LPIPS (`lpipsPyTorch`, VGG); evaluation renders the decoded model after
loading the compressed pkl. Per-scene numbers in supp. Tables 2–4; "3D-GS" rows there are the
authors' own 3DGS runs (3D-GS* in main Table 1).

## 7. Ambiguities / paper-vs-code differences
- Densification interval, prune interval/until differ between paper text and config
  (`opt_params`, effective); the config is used. Decoder LR / latent LR scale in the yaml are
  dead keys (wrong section); effective values = paper supp. Table 1 (found after a smoke run with
  the yaml values saturated every opacity to 1 and froze SH latents).
- "MLP decoder" in the paper is a single linear layer in the config (`num_layers_dec = 0`).
- Influence accumulation in the official CUDA is a non-atomic `+=` (data race, undercounts);
  the formula (Eq. 3) is a plain sum.
- Paper "Storage Mem" is the `size()` Shannon estimate (logged); the pkl adds pickle overhead
  and torchac's range-coding overhead.
- The official training samples cameras without replacement (stack pop, `Random(0)`).
