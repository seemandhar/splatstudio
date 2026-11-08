# OMG review 1

**Verdict: APPROVED for the 30k benchmark.** 0 critical, 0 major, 4 minor issues. Unit tests pass (12/12).

## Red flag: smoke decoded PSNR is 19.22. Is it a decode bug? No.
I re-ran the compressed 2k smoke on garden with `smoke_schedule.py` and a monkey-patch on `_encode`. The patch measured validation PSNR before and after encode. The script is `/tmp/claude-1013/-home-sejain/6ad4135e-056e-4bd4-9844-6b8764bea041/scratchpad/diag.py` (GPU 5).

| state | PSNR |
|---|---|
| live, before SVQ (fp32 per-Gaussian geometry/T/V) | 19.85 |
| live SVQ model just before encode (bank SH degree 1) | 19.68 |
| live SVQ, but MLP_s fed fp16-rounded positions | 19.66 |
| live SVQ, but fp16-rounded MLP weights | 19.68 |
| **decoded/baked model (what validation and ss-eval measure)** | **19.65** |
| decoded, but MLPs evaluated at the fp32 positions | 19.67 |
| decoded, rendered at SH degree 3 instead of 1 | 16.99 |

- Encode→decode costs **−0.02 dB**, which is inside the official ~0.0–0.1 dB.
- SVQ costs 0.17 dB.
- So the following are all correct:
  - the G-PCC reorder against the attribute/index order;
  - the Huffman index mapping;
  - the fp16 codebook and MLP weights;
  - the fp16-position space feature;
  - the SH/opacity bake from the MLPs.
- The −1.9 dB gap to MS's compressed smoke (21.54) is already there in the live model. Its causes are on the training side of the compressed schedule:
  - the MLPs get 700 steps of a schedule built for 15k (the warm-up plus the first milestone use all of it);
  - there are 2× fewer Gaussians after the LD prune at τ 0.96.
- The degree-3 row matters only in the compressed schedule, where bands 2–3 are never trained. At 30k the bank reaches degree 3 at 18k, and the runtime evaluates at degree 3 (runtime.py:198/242). The run-length guard (density.py:98) refuses other lengths.

## Checked against the official code. All match.
- **Mini-Splatting base**: the copied ms_* files are unchanged from the matched port.
  - The official degree-0 phase before 15k is reproduced by the shN masking (ms_density.py:298).
  - The xyz LR restart is at it−15000+5000.
- **15k**: `intersection_sampling` + reinit (with T/V zeroed, density.py:134), then construct_net, which sets T = sh0 and V = 0 and gives every Gaussian optimiser fresh Adam state (density.py:197).
  - This matches train.py:161-180 and gaussian_model.py construct_net / training_setup (f_static/f_view lr 0.0025, no opacity/f_dc in the optimiser).
  - The MLP layouts are tcnn with no bias and padding to 16. The parameter count is 15360, which matches the 30720 B of MLPs in storage.
  - Frequency(16) and contract_to_unisphere (aabb [-1,1]) match render_imp:173.
- **Net schedule**: the net Adam steps for it > 15000 and the scheduler steps after it (density.py:117-121), as in train.py:187-190. The closed form is LinearLR(0.01, 100) plus MultiStepLR([1000, 3500, 6000], 0.33).
- **20k LD prune** (density.py:179, core.py:40-68):
  - importance comes from intersection_preserving with the MLP opacity (render_imp);
  - neighbours are `T[order[clamp(order∓1)]]`, the official permutation-twice construction (gaussian_model.py ld_scoring);
  - λ = 2;
  - `cdf_keep_mask` is identical to init_cdf_mask;
  - all Gaussian Adam state is fresh and the MLP Adam is kept.
- **SVQ**: K-means runs after step 28999, i.e. on the state that render 29000 sees (train.py:88).
  - Clusters are 3×64 of length 1, 2×512 of length 2 and 3×1024 of length 2 (arguments 100-105).
  - K-means runs on raw log-scale and raw quaternions and on cat[T,V].
  - Indices are frozen.
  - Codebook Adam uses lr 1e-8 and eps 1e-15, steps for it ≥ 29000 and never calls zero_grad, as in the official code.
  - The render uses only the codebooks (renderer.py:335).
- **requires_grad**: the base opacities/sh0/shN and the post-SVQ scales/quats/T/V receive grad=None, so Adam skips them; the official optimiser does not hold them either. Every `_replace_all` creates fresh leaf Parameters. The bake keeps requires_grad (density.py:282).
- **Storage**: `official_bytes` follows write_storage (compress_utils.py:27):
  - the G-PCC stream length;
  - codebook entries × 2 B;
  - the dahuffman payload length, with the htables **not** counted (as officially);
  - MLP params × 2 B.

  `container_mb` (the real file) is also reported. For comparison with the paper's 4.61/2.95 MB, use `official_total_mb`, and quote the container size next to it.

## Minor
1. **density.py:264-277 vs gaussian_model.py:899/train.py:125.** Officially the evaluated model is `decode(decompress=False)`: fp16 positions, but fp32 codebooks and tcnn master weights. Ours decodes the fp16 codebooks and MLPs, so it is stricter. Measured effect ≤0.02 dB. Add this to the fidelity-gap list in report.md.
2. **probes.py:330 (omg-codec).**
   - The reference model is the live model with fp16-rounded means, so the probe cannot detect sensitivity of the hash-like 2^15 π features to position rounding.
   - Suggestion: put a pre/post-encode PSNR on a few training views (cached `self.images`) into the `encode` report. Then every 30k run self-certifies the decode loss.
3. **evidence/test.json.** The note on the smoke drop should cite these measurements (live 19.68 → decoded 19.65) instead of only reasoning about it.
4. **density.py:113-124 (gap 6).** The per-Gaussian Adam still steps the means at step 30000 (lr ~1.6e-6·extent). This is negligible and documented.

## Launch
```
python3 tools/bench.py launch --method omg --dataset mipnerf360 --scenes bicycle counter --steps 30000 \
  --exp omg-mipnerf360 --method-per-scene mipnerf360/bicycle=omg,mipnerf360/counter=omg-indoor \
  --extra "--recipe 3dgs --num-gaussians 10000000 --max-gaussians 12000000"
```
- Use the exact flags from bench_request.json.
- Put the run directory on /bigdata with a symlink.
- Afterwards run `tools/ssenv python reproductions/omg/plugin/report_size.py runs/<exp>/<run>`.
- Compare against runs/mini-splatting-mipnerf360 (bicycle) and runs/mini-splatting-indoor-mipnerf360 (counter).
