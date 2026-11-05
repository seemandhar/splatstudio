# EAGLES — native SplatStudio reproduction report

**Outcome: matched.** All three scenes are within ±0.5 dB of the paper, and the compression cost relative to 3DGS is reproduced. The model evaluated is the **decoded** model, after encode → decode → bake.

- **PSNR (ours / paper):** bicycle 24.80 / 25.04 (−0.24), garden 26.52 / 26.91 (−0.39), counter 28.09 / 28.40 (−0.31).
- **Encoded size, paper formula:** 105.7 / 64.5 / 23.6 MB vs the paper's 104 / 74 / 25 MB.

Paper: Girish, Gupta, Shrivastava, *EAGLES: Efficient Accelerated 3D Gaussians with Lightweight EncodingS*, ECCV 2024, arXiv:2312.04564 (plan id 39).
Official code: https://github.com/Sharath-girish/efficientgaussian at commit b9b50db (read, not run).
Plugin: `reproductions/eagles/plugin` (method `eagles`).

## Protocol
- **Data:** Mip-NeRF 360 with images_4 for outdoor scenes and images_2 for indoor; llffhold 8; 30000 steps; `--recipe 3dgs`; all SfM points kept.
- **Scenes:** the paper reports per-scene results (supp. Table 2). Three were chosen: bicycle and garden (outdoor), counter (indoor).
- **Baseline:** shared `runs/baseline3dgs-mipnerf360-30000/gsplat/`, which is gsplat with the opacity-reset fix.
- **Comparison:** our change relative to that gsplat baseline is compared with the paper's change relative to its own 3DGS runs (the "3D-GS*" rows).
- **Metrics:** computed by `ss-eval` on the test split, rendered from the baked checkpoint in a fresh process with no decoder state.

## Results (test split, decoded model)

| scene | PSNR | SSIM | LPIPS | paper PSNR / SSIM / LPIPS | Δ PSNR vs paper |
|---|---|---|---|---|---|
| bicycle | 24.802 | 0.7513 | 0.2680 | 25.04 / 0.75 / 0.24 | −0.24 |
| garden | 26.523 | 0.8438 | 0.1689 | 26.91 / 0.84 / 0.15 | −0.39 |
| counter | 28.090 | 0.8972 | 0.2777 | 28.40 / 0.90 / 0.20 | −0.31 |

### Size, Gaussian count, time

| scene | Gaussians (ours / paper) | paper_size_MB (ours / paper) | real coded MB | latent bits per Gaussian (SH / rot / opa) | 3DGS-equivalent PLY of our model | train s / peak GiB |
|---|---|---|---|---|---|---|
| bicycle | 2.33M / 2.26M | 105.65 / 104 | 105.66 | 34.2 / 35.5 / 4.8 | 551 MiB | 5047 / 4.96 |
| garden | 1.45M / 1.65M | 64.54 / 74 | 64.55 | 28.1 / 35.1 / 4.6 | 343 MiB | 8221 / 3.12 |
| counter | 0.53M / 0.56M | 23.58 / 25 | 23.59 | 32.3 / 33.9 / 4.4 | 124 MiB | 2791 / 1.37 |

How to read the size columns:
- **`paper_size_MB`** is exactly the official `GaussianModelSQ.size()`. It is the per-latent-dimension empirical entropy, plus 32 bits per decoder parameter and per unquantized value (position, SH DC, scale), divided by 8·10⁶. This is the paper's "Storage Mem".
- **"real coded MB"** is the actual range-coded bitstream (constriction), plus the value/count tables, the fp32 decoder weights and the fp32 unquantized attributes. It agrees with the formula to within 0.01 MB.
- **Dominant cost:** the unquantized fp32 attributes (9 floats per Gaussian = 288 bits) are 80% or more of every model.
- **Encode / decode time:** 2.2 / 1.7 s for bicycle; 0.5 / 0.3 s for counter.

Training times come from 8 GPUs shared with other users. They are not comparable to the paper or to the baseline; garden in particular was slowed by contention.

The paper's "Average" row is not the mean of its own per-scene rows (PSNR 27.23 vs 27.19; storage 54 vs 59.9 MB). For that reason only per-scene comparisons are made here.

### Relative to the baseline (the fidelity claim)

| scene | gsplat baseline PSNR / SSIM / LPIPS / Gaussians | ours − gsplat | paper EAGLES − paper 3D-GS* |
|---|---|---|---|
| bicycle | 24.921 / 0.7573 / 0.2428 / 6.59M | −0.12 dB | −0.09 dB |
| garden | 26.893 / 0.8598 / 0.1311 / 4.76M | −0.37 dB | −0.41 dB |
| counter | 28.846 / 0.9073 / 0.2605 / 1.22M | −0.76 dB | −0.71 dB |

- **PSNR cost:** our cost relative to the baseline matches the paper's cost relative to its 3DGS runs within 0.05 dB on every scene.
- **Gaussian counts:** we keep 35% / 30% / 43% of the baseline's Gaussians. The paper keeps 43% / 29% / 48% of its own 3DGS counts.
- **SSIM / LPIPS:** our losses are slightly larger than the paper's. LPIPS rises by 0.025 / 0.038 / 0.017; the paper reports +0.00 / +0.03 / +0.02.

## Mechanism → implementation
- **C1, quantized attributes.** Rotation (8 dims), opacity (1) and SH-rest (16) are per-Gaussian latent fields `q_rot`, `q_opa`, `q_shN`. They are decoded by learned linear decoders with straight-through rounding (Eq. 1). The SH decoder uses a fixed 16×45 DCT matrix with a learned (1,45) scale and shift.
  - **Initialization:** latents start from the decoder inverse (supp. Eq. 1). Rotation uses the minimum-norm least-squares solution, opacity uses (x−b)/(S+1e-6), and SH starts at zero.
  - **Learning rates:** latent LR = 3DGS LR / ‖S‖_F for SH and rotation, and 0.05 for opacity. Decoders use Adam with LR 1e-4 and eps 1e-15.
  - **Opacity reset:** done in latent space.
  - **Where the decoders live:** they are global learned state, which the runtime cannot hold. The density owns a `DecoderBank` (an `nn.Module` with its own Adam), steps it in `after_backward`, and registers it in a process-local slot that the renderer reads.
  - **Base fields:** `quats`, `opacities` and `shN` are overwritten with the decoded values every step. This way gsplat's split, which uses the rotation, and its opacity pruning act on the decoded attributes.
- **Decoded evaluation.** The pattern follows the Reduced-3DGS plugin (copied, not imported). At step 30000 the latents are rounded and entropy-coded. The bitstream is decoded and checked to be bit-exact, then passed through the decoders. The result is baked into the base fields and the bank is marked as baked. Validation, the checkpoint and `ss-eval` therefore render exactly the decoded model. Decoder weights, the bitstream and the size report are stored in `density_state["eagles"]`.
- **C2, progressive training.** The render scale follows a cosine from 0.3 to 1.0 over steps 1–21000 (21000 = int(0.7·30001)). The renderer renders at floor(s·H) with K scaled by s. The loss downsamples the target with the official bilinear `F.interpolate` call.
- **C3, controlled densification and influence pruning.**
  - **Densification:** a `DefaultStrategy` subclass densifies every 125 steps from step 500 to 18000. Opacity resets at multiples of 2500 below 18000.
  - **Influence:** the per-Gaussian influence Σ_p α·T (Eq. 3) is computed by a CUDA kernel (`csrc/influence.cu`) adapted, with attribution, from the Reduced-3DGS transmittance kernel.
  - **Pruning:** at steps 5100, 10200, 15300, 20400 and 25500 the 15% of Gaussians with the lowest influence are pruned. Each round removes 12–21% of the model, a little more than 15% because of ties at the quantile. Clones and split children start with zero influence, as in the official code.

## Finding: the yaml's decoder and latent LRs never reach the official optimizer
`configs/efficient-3dgs.yaml` lists `*_ldecs_lr` (0.005 / 0.01 / 0.3) and `*_lr_scaling` (0.1) under `quantize_params`. However, `train_eval.py` passes only `config['opt_params']` to `OptimizationParams`, which is the class that defines those keys. `QuantizeParams` has no such attributes. The official runs therefore used the defaults: decoder LR 1e-4 and latent LR scale 1.0, which is exactly the paper's supp. Table 1. The reviewer verified this against `arguments/__init__.py`.

The first smoke run used the yaml values, and training degenerated. Every opacity saturated to 1, `q_opa` had only 2 distinct values, the SH latents stayed at about zero, and the run scored 22.03 dB on garden at 2k steps without progressive training. With the effective values the same control scores 23.01 dB, against 23.04 for gsplat. Those first runs are kept as `evidence/*-deadkeys`.

## Decode/bake equality evidence
- **Probe `codec-bake` (CUDA).** It renders with the decoders during training, then overwrites the base fields with garbage, encodes, decodes and bakes. It then saves and reloads the checkpoint with `weights_only=True` and renders with a fresh renderer that has no decoders. Maximum absolute difference from the training render: 0.0. The baked latents are exact integers.
- **Encode step.** The decoded bitstream is compared with the rounded latents using `torch.equal`; a mismatch raises an error. Every benchmark run passed this check.
- **Compressed-schedule smoke run.** It runs every event within 2000 steps on garden: 22.53 dB with 123k Gaussians and 5.40 MB. It was then evaluated through `runtime.evaluate` on the baked checkpoint after clearing the bank, which is the same path `ss-eval` takes.

## Probes (`ss-verify`: declared_checks_passed, 12 probes, 4 on CUDA)
- **decoder-ste:** checks the forward pass against independent formulas, including a closed-form DCT-II basis; checks that values within a rounding bin decode identically; checks the straight-through gradient.
- **latent-init-lr:** checks the inversion targets (identity quaternion, logit 0.1, zero SH), the latent LRs from the decoder norms, and the decoder LR of 1e-4.
- **entropy-size:** checks `size()` against analytic entropies (2 bits and 1.5 bits per value) and the decoder bit count (27936). The range coder round-trips losslessly with 0.04% overhead.
- **progressive-schedule:** checks the cosine boundary values, the floor(s·H) render size, K·s, and that the loss uses the downsampled target.
- **influence-kernel:** compares the CUDA Σ α·T against per-pixel compositing in torch (error 3e-6). The total influence equals the rendered alpha.
- **influence-prune:** checks that exactly the set at or below the 15% quantile is pruned and influence is reset, and that clones and split children start at zero.
- **opacity-reset:** checks the latent written by the reset and that its Adam moments are zeroed.
- **codec-bake:** described in the previous section.
- **schedule:** checks 139 refine steps, resets at 2500–17500, prunes at 5100–25500, and encode at 30000. It also checks that gsplat's own reset never fires.
- **Base probes:** state-transitions, photometric-gradient and renderer-gradient also pass.

## Fidelity gaps and deviations
1. **Runtime conventions shared with the baseline:** views sampled at random with replacement (official: a shuffled stack); white background (official: black); optimizer step before densification; xyz LR schedule without the delay; gsplat rasterizer with alpha clamp 0.999 (official 0.99).
2. **No screen-size pruning** (radii > 20 px). This is the gsplat default, and the baseline does the same.
3. **Influence summed with `atomicAdd`.** The official CUDA uses a non-atomic `+=`, a data race that undercounts.
4. **Downsampled camera uses K·s,** which matches `F.interpolate`'s pixel mapping exactly. The official code keeps the field of view and uses focal × W'/W. The difference is sub-pixel.
5. **Real bitstream coder differs.** We range-code each latent dimension with its own categorical model; the official pkl uses torchac with one flattened CDF per attribute. The paper's reported number, `size()`, is reproduced exactly.
6. **Decoders are not runtime parameters.** The decoded model is baked at the final step, so a run whose `--steps` differs from `total_steps` (30000) never encodes and produces no size report.
7. **Final-step update:** the runtime applies the latent and decoder optimizer steps at step 30000. The official code skips the optimizer step at the last iteration. This is a one-step difference.

## Review (approved; minor issues only)
- **Process-global decoder slot is never cleared.** A stale, un-baked bank could decode a different run's parameters in the same process. This cannot happen through `ss-train` or `ss-eval` (fresh process each time), and the probes clear the slot themselves. Not fixed.
- **Final-step optimizer update:** gap 7 above.
- **Paper's Mip-NeRF 360 "Average" row** is inconsistent with its per-scene rows, so only per-scene comparisons are made.
- **gsplat's own opacity reset of the base field** at multiples of 2500. It is harmless because the base field is overwritten with the decoded values in the same call, and the opacity optimizer receives no gradients. This is checked by the `schedule` probe and the per-step overwrite.

## Smoke tests (garden, 2000 steps)
| run | PSNR (dB) | Gaussians | size |
|---|---|---|---|
| gsplat baseline | 23.04 | 839k | — |
| EAGLES, default 30k schedule | 20.58 | 653k | — |
| EAGLES, progressive training off (control) | 23.01 | — | — |
| EAGLES, compressed schedule, decoded | 22.53 | 123k | 5.40 MB |

The default-schedule run scores low because at step 2000 it still trains at about 0.33× resolution but is evaluated at full resolution.

## Reproduce
```bash
cd /home/sejain/splatify/reproductions/eagles/plugin && ../../../tools/ssenv pip install -e .
T=/home/sejain/splatify/tools/ssenv
$T ss-verify eagles --device cuda --output ../evidence/verify.json
$T ss-train eagles --data ~/datasets/mipnerf360/native/bicycle --output runs/eagles-bicycle \
   --steps 30000 --recipe 3dgs --num-gaussians 10000000 --max-gaussians 12000000
$T ss-eval --run runs/eagles-bicycle --split test
$T python /home/sejain/splatify/reproductions/eagles/plugin/report_size.py runs/eagles-bicycle
# all events in 2k steps:  $T python plugin/smoke_schedule.py <data> <out>
# benchmark: python3 tools/bench.py with bench_request.json (mipnerf360 bicycle garden counter, 30000)
```
Benchmark runs: `runs/eagles-mipnerf360/eagles/mipnerf360-{bicycle,garden,counter}/`.
