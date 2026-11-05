# RDO-Gaussian — implementation spec

Paper: *End-to-End Rate-Distortion Optimized 3D Gaussian Representation*, H. Wang, H. Zhu, T. He,
R. Feng, J. Deng, J. Bian, Z. Chen, ECCV 2024. arXiv 2406.01597.
Official code: https://github.com/USTC-IMCL/RDO-Gaussian (commit 710ebab8), cloned in
`source/code/` (3DGS licence; entropy coder = CompressAI rANS, Apache-2.0, + ryg_rans, public domain).
LaTeX in `source/latex/` (main.tex + appendix.tex with hyper-parameters and per-scene tables).

## 1. Method summary
- Plain 3DGS training (densification etc.) for 15k iterations; from iteration 15001 to 30000 the
  model is trained *compression-aware* with a rate–distortion objective (no densification any more).
- Learnable per-Gaussian mask (STE, sigmoid > 0.1) zeros scale and opacity (dynamic pruning;
  masked Gaussians may come back; removed only at the end).
- Learnable per-Gaussian, per-SH-degree mask (3 masks) zeros SH bands 1/2/3 (adaptive SH).
- Entropy-constrained VQ (ECVQ) of scale, rotation, DC, SH1, SH2, SH3 with six codebooks and a
  learned unconditional softmax entropy model per codebook; codeword choice minimises
  distance + rate/λ; rate (bits) is part of the loss.
- Final encoding: remove masked Gaussians, sort by SH-mask pattern (8 boundaries in the header),
  shrink codebooks to used codewords, rANS-code all indices with the learned pmfs, 8-bit uniform
  opacity quantization with histogram pmf (also rANS), fp16 positions.

Contributions
- **C1 RD formulation / training objective** (§3.2 "Training Objective", Eq. 11): L = λ_GS L_GS +
  λ_SH L_SH + L_rate + L_VQ + L_render.
- **C2 Gaussian pruning** (§3.2 "Gaussian Pruning", Eq. 2–5).
- **C3 Adaptive SH pruning** (§3.2 "Adaptive Spherical Harmonics Pruning", Eq. 6–8).
- **C4 ECVQ with learned entropy model** (§3.2 "Entropy-Constrained Vector Quantization", Eq. 9–11).
- **C5 Parameter encoding** (§3.3: removal, SH-mask rearrangement, arithmetic coding, fp16 xyz,
  8-bit opacity) — defines the reported size.

## 2. Equations (paper LaTeX, then code reality)

Eq. 2–3 (Gaussian mask):
```
\phi_i^{\rm soft} = \texttt{sigmoid}(\phi_i)
\phi_i^{\rm hard} = \texttt{sg}(\mathbbm{1}(\phi_i^{\rm soft} > \phi_{\rm thres}) - \phi_i^{\rm soft}) + \phi_i^{\rm soft}
```
Eq. 4: `\hat{s}_i = \phi_i^{hard} s_i, \hat\alpha_i = \phi_i^{hard}\alpha_i` (s, α activated:
code `torch.mul(self.get_scaling, hard)`, `torch.mul(self.get_opacity, hard)`).
Eq. 5: `\mathcal{L}_{GSprune} = \frac{1}{N}\sum_i \phi_i^{soft}` (code: mean over all N Gaussians).

Eq. 6–7 (SH mask, l = 1..3): same sigmoid/STE with θ_thres; `\hat c_i^{(l)} = \theta^{(l)hard} c_i^{(l)}`.
Eq. 8: `\mathcal{L}_{SHprune} = \frac{1}{N}\sum_i\sum_{l=1}^k \frac{2l+1}{(k+1)^2-1}\theta_i^{(l)soft}`, k=3.
Code: the soft mask is expanded to the 15×3 coefficient grid (3/5/7 coefficients × 3 channels)
and `mean` is taken over **Gaussians kept by the GS bit mask** (`sh_soft_mask[gs_bitmask]`), which
equals the (2l+1)/15 weighting averaged over the kept Gaussians (N = number of GS-kept Gaussians).

Eq. 9 (entropy model): paper `p_j = e^{-w_j} / \sum_m e^{-w_m}`; code `log_softmax(+logits)`
(sign convention irrelevant), log-pmf clamped at `log(1e-9)` when it requires grad.

Eq. 10 (ECVQ assignment):
`j = \arg\min_m -\frac{\log p_m}{\lambda^{(s)}} + (s_i - CB^{(s)}[m])^2`
Code (`vq/ecvq.py:quant`): `dist = cdist(x, codebook, p=2)` (**Euclidean, not squared**) `+
log2_pmf / lmbda` (bits = −log2 p), the rate term only once `rate_constrain` is on (iteration
≥ 20001); before that plain nearest neighbour. Indices are recomputed every 10 iterations
(`update_index=(iteration-1) % 10 == 0`), otherwise taken from a global cache.

Eq. 11 (losses): `L_rate = 1/N Σ_i Σ_attr r_{i,j}/λ^{attr}`, `L_VQ = 1/N Σ_i Σ_attr d_{i,j}`.
Code (`gaussian_renderer/__init__.py`):
```
rate_loss = Σ_key bits_key / lmbda_key / num_gs / vq_dim
vq_loss   = Σ_key Σ_rows ||x_hat − x||_2 / num_gs / vq_dim
```
with `num_gs` = total Gaussians N (incl. masked), `vq_dim` = 3+4+3+9+15+21 = 55, bits = Σ −log2 p
of the chosen codewords (gradient only into the logits), d = **unsquared L2 norm** (gradient into
both the codebook and the Gaussian attribute). Rendering uses STE `x_q = sg(x_hat − x) + x`
(gradient of the render only into x). Rotation codeword is normalised after STE.

Eq. 12/13: `L_render = (1−λ_SSIM) L1 + λ_SSIM D-SSIM`, λ_SSIM = 0.2;
`L_total = λ_GS L_GS + λ_SH L_SH + L_rate + L_VQ + L_render`.

VQ inputs (code `vector_dict`): scale = **activated** `exp(_scaling)` masked by the GS hard mask
(rows with GS bit mask only); rot = raw `_rotation` (unnormalised); dc = `_features_dc` (3);
sh1/sh2/sh3 = masked `features_rest[:, 0:3 / 3:8 / 8:15, :]` flattened (9/15/21) on rows with
`sh_l bitmask ∧ gs bitmask`.

## 3. Hyper-parameters

| name | value | source |
|---|---|---|
| iterations | 30000 | run_real_world.sh |
| densification | 3DGS defaults (500–15000, every 100, grad 2e-4, opacity reset 3000) | arguments/__init__.py |
| gs/sh prune & VQ start | 15001 | arguments `gs_prune_start_iter`, `sh_prune_start_iter`, `vq_start_iter` |
| rate_constrain_iter | 20001 | arguments |
| reactivate_codeword_period | 1000 (only while it < 20001; prob threshold 1e-5) | train.py |
| φ_thres, θ_thres | 0.1, 0.1 | App. A1; gaussian_model.py:64-65 |
| GS mask lr | 0.01 (Adam eps 1e-15), init 0 | App. A1; run script |
| SH mask lr | 0.05 (real scenes), init 0 | App. A1; run script |
| codebook lr / logits lr | 2e-4 / 2e-3, Adam eps 1e-15, MultiStepLR([12000], 0.1) on VQ steps | App. A1; gaussian_model.py:223-228 |
| codebook sizes | 8192 scale/rot/dc; 4096 sh1/sh2/sh3 | App. A1; run script |
| λ^(s) | 32768 | App. A1 |
| λ^(r), λ^(DC), λ^(SH1..3) | 256 | App. A1 |
| λ_GS / λ_SH (6 rate points) | 0.0005/0.005, 0.002/0.02, 0.005/0.05, 0.01/0.1, 0.02/0.2, 0.05/0.5 | App. Tab. A.1–A.3, run script |
| highest rate (per-scene tables) | λ_GS 0.0005, λ_SH 0.005 | App. Tab. A.5–A.7 |
| vq_patch_size | 65536 (memory chunking only) | run script |
| k-means init | 10 iterations, random initial samples, at iteration 15001 on all Gaussians; logits = log(L2-normalise(bin counts)) | vq/ecvq.py, vq/kmeans.py |
| opacity quant | 8 bit uniform over [min, max] of **logit** opacity | utils/encode_utils.py |
| rANS precision | 16 bit | utils/encode_utils.py |

## 4. Schedules
- 1–15000: plain 3DGS (masks and VQ inactive; mask/VQ optimizers not stepped).
- iteration 15001 (start): k-means init of the six codebooks; masks and VQ active.
- 16000…20000 every 1000 (after optimizer step): dead-codeword reactivation (p ≤ 1e-5 → copy a
  live codeword sampled ∝ p, + N(0, 1e-4) noise, copy its logit).
- 20001: rate-constrained assignment on.
- VQ optimizer LR × 0.1 after 12000 VQ steps (iterations ≥ 27001).
- iteration 30000: no optimizer step (3DGS), then mask apply + encode (`apply_and_save_vq`).

## 5. Builds on
3DGS (official Inria code, same rasterizer). Unchanged: initialisation, densification, opacity
reset, LR schedules, L1+D-SSIM, SH degree schedule. Changed: from 15001 the rendered attributes
are masked and quantised.

## 6. Protocol
- Datasets: Mip-NeRF 360 (9 scenes), T&T (truck, train), Deep Blending (drjohnson, playroom),
  NeRF-Synthetic. "We follow the experimental protocols provided by 3DGS".
- Resolution: the run script passes neither `-i` nor `-r`: images = full-resolution `images/`
  folder, `--resolution -1` → widths > 1600 are PIL-resized to width 1600
  (`utils/camera_utils.py`). → our `mipnerf360-1600` dataset. T&T / DB at native size.
- Split: `--eval`, llffhold 8. Background black. Metrics: official `metrics.py` (3DGS SSIM,
  PSNR, LPIPS-VGG via lpipsPyTorch) on renders saved as PNG.
- Evaluated model: `render.py` builds `Scene`, whose `__init__` calls `gaussians.decode(model_path/bitstream)`
  (scene/__init__.py:79): the paper's numbers are on the **decoded bitstream** (fp16 positions, 8-bit
  opacity, codewords) — our default `evaluate: "decoded"`. (reviewer's estimate of the fp16-position/8-bit-opacity effect vs the undecoded model: 0.01-0.05 dB.)
- Size ("Rate (MB)", csv "Size [Bytes]", MB = 1e6 bytes): total bytes of the `bitstream/`
  directory = `index_bitstream.bin` (length-prefixed rANS strings of scale, rot, dc, sh1, sh2,
  sh3, opa) + `header.bin` (8 uint32 boundaries + 2 float32) + `codebook.npz` + `logits.npz`
  (shrunk codebooks/logits float32; opacity log-hist float64) + `position.npz` (fp16 xyz),
  matching Table 3 (header, indexes, codebooks, logits, positions).

## 7. Ambiguities / paper-vs-code
1. VQ distance: paper squared L2; code unsquared Euclidean (assignment and VQ loss). Use code.
2. Loss normalisation: code divides rate and VQ loss also by vq_dim = 55 (not in Eq. 11). Use code.
3. L_SH averaged over GS-kept Gaussians only (code). L_GS over all N.
4. Evaluation is on the decoded bitstream (official `Scene.__init__` → `decode`), as our default.
   `evaluate: "official"` is a misnamed diagnostic undecoded mode (fp32 xyz, unquantised opacity) that no official evaluation mirrors; reviewer's estimate of the fp16-position/8-bit-opacity effect vs the undecoded model: 0.01-0.05 dB. (Corrected after review-1.)
5. The paper's "3DGS" rows are the authors' re-implementation at the 1600-px protocol.
6. Final-iteration order: masks pruned → `remove_negatives` (negative scale codewords → 1e-8) →
   rate-constrained re-assignment of all attributes → sort (np.argsort, unstable) → shrink.
