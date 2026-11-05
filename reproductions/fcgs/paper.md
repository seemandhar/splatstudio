# FCGS — Fast Feedforward 3D Gaussian Splatting Compression (ICLR 2025)

arXiv 2410.08017 · Chen, Wu, Li, Lin, Harandi, Cai · official code
github.com/YihangChen-ee/FCGS (commit 31e59a46, Apache-2.0 + non-commercial clause), cloned
into `source/code/` (read only). **Pretrained weights are released in the repository**
(`checkpoints/checkpoint_{1e-4,2e-4,4e-4,8e-4,16e-4}.pkl`, 10 MB each, plain state dicts;
copied to `~/models/fcgs/`, SHA256 in `~/models/fcgs/SHA256SUMS`).

## 1. Method summary
- A single pretrained network compresses an *already trained* vanilla 3DGS (SH degree 3,
  56 attributes / Gaussian + xyz) in one feed-forward pass; no per-scene optimisation.
- Positions: 16-bit voxelisation (per-axis min/max) + lossless GPCC (MPEG `tmc3`).
- Geometry attributes f_geo = (opacity logit, 3 log-scales, 4 raw quaternion) are directly
  scalar-quantised with learned steps; colour f_col (DC + 45 SH-rest) goes either through an
  MLP autoencoder (m=1) or direct quantisation (m=0), chosen per Gaussian by MEM.
- Entropy coding of every quantised stream with a GMM whose components come from a
  hyperprior, an inter-Gaussian grid context and an intra-Gaussian channel context; real
  bitstreams via a CUDA arithmetic coder (torchac-derived).
- λ ∈ {1e-4 (high rate), 2e-4, 4e-4, 8e-4, 16e-4 (low rate)} selects a separately trained model.

Contributions
- **C1 MEM** (§3.2, Eq. 3): per-Gaussian binary mask m = Sig(MLP_m(f_gau)) > ε_m routes
  colour through g_s(Q(g_a(x))) (m=1) or Q(x) (m=0); geometry always Q(x).
- **C2 Inter-Gaussian context** (§3.3, Eqs. 4–5): N_s=4 random batches (1/6,1/6,1/3,1/3);
  previously decoded batches are splatted into created 3-D (70,80,90) and 2-D (300,400,500,
  three axis planes) grids (`gridcreater` CUDA, trilinear-weight average), interpolated
  (`gridencoder` CUDA) at the next batch's positions + freq. embedding → MLP_s → (μ,σ,π).
- **C3 Intra-Gaussian context** (§3.3, Eq. 6): m=1 latent split into 4 chunks of 64 channels,
  m=0 colour split into 3 RGB chunks; MLP_c predicts each chunk from earlier ones.
- **C4 GMM + hyperprior entropy coding** (§3.3, Eq. 7; §3.4): mixture of h/s/c (geometry:
  h/s) Gaussians with softmax weights; factorised prior for z; arithmetic coding; mask coded
  with a single Bernoulli probability. Size = sum of all bitstreams.
- (Training of the codec, Eq. 8, on DL3DV-GS 6670 scenes / 60 GPU-days: **out of scope**,
  released weights are used.)

## 2. Equations (verbatim from the LaTeX)
Eq. 3: `\hat{\bm{x}}_i = g_s(\texttt{Q}(g_a(\bm{x}_i)))\times m_i + \texttt{Q}(\bm{x}_i)\times (1-m_i), \; m_i=\texttt{Sig}(\text{MLP}_\text{m}(\bm{f}^{\text{gau}}_i)) > \epsilon_m`
(q = 1 in latent space; learned decimal step otherwise; test-time rounding.)

Eq. 4 (grid creation): `\bm{f}^{\bm{v}}_i = \frac{\sum_{k} w_k\hat{\bm{y}}_k}{\sum_{k} w_k}, w_k=\prod_{\text{dim}}(1-|\bm{\mu}^\text{g}_{k,\text{dim}}-\bm{v}_{i,\text{dim}}|)`

Eq. 5: `\bm{\mu}^\text{s}_i, \bm{\sigma}^\text{s}_i, \bm{\pi}^\text{s}_i = \text{MLP}_\text{s} (\oplus[\bm{f}^{\bm{\mu}^\text{g}}_i, \texttt{emb}(\bm{\mu}^\text{g}_i)])`

Eq. 6: `\bm{\mu}^\text{c}_{i, n^\text{c}}, \bm{\sigma}^\text{c}_{i, n^\text{c}}, \bm{\pi}^\text{c}_{i, n^\text{c}} = \text{MLP}_\text{c}(\hat{\bm{y}}_{i, [0,n^\text{c}c-c)})`

Eq. 7: `p(\hat{y}_{i,j})=\sum_{l\in\{h,s,c\}}\theta^l_{i,j}(\Phi(\hat{y}+q^l/2|\mu^l,\sigma^l)-\Phi(\hat{y}-q^l/2|\mu^l,\sigma^l)), \theta^l=\mathrm{softmax}(\pi^l)`

Eq. 8 (training only): `L = L_{fid} + \lambda L_{entropy}/(N^g\cdot 56)`.

## 3. Hyper-parameters (inference)
| name | value | source |
|---|---|---|
| quant. base steps Q_fe, Q_op, Q_sc, Q_ro | 1e-3, 1e-3, 1e-2, 1e-5, adapted by learned `ad_*`: (Q·a0+a1)(1+tanh a2) | code:model/FCGS_model.py:150,372 |
| latent dim (m=1) / hidden | 256 / 256 | paper §4.1, code |
| z dims geo / col m=0 / col m=1 | 16 / 24 / 64 | paper §4.1 (code agrees) |
| 2-D grid res | 300, 400, 500 | paper, encode_single_scene.py |
| 3-D grid res | 70, 80, 90 (class default 60,80,100 overridden) | paper, encode_single_scene.py |
| grid feature dim | 48 (latent → 48 by a linear layer; geo grids 8) | code |
| freq. embedding | degree 4 on 3-D (output 27) | code:FCGS_model.py:160 |
| N_s batches | 4, boundaries N//6·{1,2,4} | paper, code:393 |
| ε_m | 0.01 | paper, code:358 |
| xyz normalisation for grids | mean ± 3·std per axis, clamped to [0,1] (`--nr 3`) | code:grid_utils.normalize_xyz |
| shuffle seed | `torch.manual_seed(1); randperm(N)` after sort | code:349 |
| voxelisation | 16 bits per axis, per-axis min/max (float32 header) | code:gpcc_utils.py |
| GPCC command | `tmc3 --mode=0 --trisoupNodeSizeLog2=0 --mergeDuplicatedPoints=0 --neighbourAvailBoundaryLog2=8 --intra_pred_max_node_size_log2=3 --positionQuantizationScale=1 --inferredDirectCodingMode=3 --maxNumQtBtBeforeOt=2 --minQtbtSizeLog2=0 --planarEnabled=0 --planarModeIdcmUse=0 --cabac_bypass_stream_enabled_flag=1` | code:gpcc_utils.py:17 |
| per-step chunk | 1,000,000 Gaussians per independent compress call (1.1M if 1.0–1.1M) | encode_single_scene.py:22 |
| AE chunk sizes fea/feq/geo | 2e6 / 1e6 / 1e6 symbols per file | encode_single_scene.py:31 |
| determ | 1 (integer-atomic deterministic grid creation at encode) | README, atomic_statement.md |
| clamp | quantised values clamped to mean ± 15000·Q | code:325 |

## 4. Schedules
None at inference. Applied once to the final (30 K) 3DGS.

## 5. Builds on
Vanilla 3DGS (Kerbl 2023) trained 30 K iterations, SH degree 3, unchanged. FCGS does not
change the number or structure of Gaussians; only positions (16-bit grid) and attribute
values (quantised / autoencoded colour) change.

## 6. Protocol
- Base models: "Each scene is initially trained for 30K iterations to produce the vanilla
  3DGS" (Fig. 3 caption); Gaussians counts in `results/*.csv` (garden 5.62 M, bicycle 5.68 M,
  counter 1.18 M) match official 3DGS 30 K models.
- Datasets: DL3DV-GS (own test split), Mip-NeRF 360 (9 scenes), Tanks&Temples (truck,
  train), Deep Blending (Fig. 7 / appendix). Standard 3DGS resolution (images_4 outdoor,
  images_2 indoor), llffhold 8.
- Evaluation (`decode_single_scene_validate.py`): decoded ply rendered with the 3DGS
  rasterizer, **black background**, renders rounded to 8 bit, PSNR/SSIM (3DGS same-padded
  Gaussian SSIM), LPIPS-VGG (`lpips` package).
- Size: sum of all bitstream bits (`bits_xyz + bits_mask + bits_fea + bits_feq + bits_geo`,
  each = payload bytes·8 + small headers) / (8·1024·1024) → "MB" are **MiB**. Check:
  mean of the 9 Mip-NeRF 360 CSV byte counts (high rate) = 67.16e6 B = 64.05 MiB = Table
  "Ours-highrate 64.05 MB". "Ours-highrate" = λ=1e-4, "Ours-lowrate" = λ=16e-4.
- Per-scene numbers: `source/code/results/<dataset>/<scene>.csv` (3dgs-compression-survey
  format; PSNR, SSIM, LPIPS, Size [Bytes], #Gaussians) for high and low rate.

## 7. Ambiguities
- GPCC version unspecified (README: "refer to tmc3 github"); flags such as
  `--inferredDirectCodingMode=3` require a recent TMC13 (≥ v14).
- Paper's 3DGS baseline row (Mip-NeRF 360 27.52 dB) is retrained by the authors; no per-scene
  baseline published.
- `compress()` uses `determ=determ_codec` for encode grids, decode always `determ=True`;
  with `--determ 1` both are identical (needed for lossless decoding).
- Scenes > 1 M Gaussians are coded as independent 1 M chunks in the file order of the ply
  (sorting/shuffle happen inside each chunk).
