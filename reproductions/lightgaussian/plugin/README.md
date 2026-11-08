# splatstudio-method-lightgaussian

Native SplatStudio port of **LightGaussian: Unbounded 3D Gaussian Compression with 15x Reduction
and 200+ FPS** (Fan et al., NeurIPS 2024, https://arxiv.org/abs/2311.17245), following the
official code github.com/VITA-Group/LightGaussian @ 6676b98 and its modified rasterizer
github.com/Kevin-2017/compress-diff-gaussian-rasterization @ 9893d8c (both under the Inria
Gaussian-Splatting non-commercial research license). Paper fidelity unverified.

## Methods
| name | steps | pipeline |
|---|---|---|
| `lightgaussian` | 45000 | 3DGS 30k → significance prune 66 % → 5k co-adaptation → SH 3→2 + 5k distillation (pseudo views) → significance-weighted VQ of SH → 5k VQ fine-tune → official zip, decoded-model eval |
| `lightgaussian-released` | 40000 | the same without the VQ fine-tune (the released `vectree.py` output is final) |
| `lightgaussian-smoke` | 2000 | compressed schedule 1000/1300/1600/2000, 100 VQ iterations (diagnostic only) |

`--steps` must equal the density's `total_steps` (a guard raises otherwise).

## Mechanism → code
- C1 `significance.py` + `csrc/significance.cu`: CUDA port of `renderCUDA_count` on gsplat tile
  lists (score += σ·T, alpha clamp 0.99, skip alpha < 1/255, stop when T(1−alpha) < 1e-4 without
  counting), summed over all training views; `v = score·(V/sort_desc(V)[int(0.9N)])^0.1`
  (`prune.py`, no clip); remove `v <= sort(v)[int(0.66(N−1))]` via `gsplat.strategy.ops.remove`.
- Co-adaptation (30001–35000): photometric loss, no densification, Adam moments kept, non-means
  LR ×0.95 per 400 its, means LR = 3DGS schedule with max_steps 35000 (`prune_finetune.py`).
- C2 (35001–40000, `renderer.py`, `losses.py`, `density.py`): teacher = frozen copy of the 35k
  model (SH 3); student SH degree 2 (`shN[:, 8:] = 0` after every step); fresh optimizer state,
  AdamW weight decay 0.01, LR ×0.90 per 500 its, means max_steps 40000, opacity frozen
  (restored after each step; the Parameter stays trainable); loss L1 + 0.2 D-SSIM(student,
  teacher) only; pseudo view (camera centre + N(0, 0.05²)) on steps with `k % 3 != 0`.
- C3 (`vq.py`, `codec.py`): at 40000 v-score recomputed; top 40 % keep fp16 SH; the rest get one
  of 8192 codes (kaiming-uniform init, weighted EMA 0.8 with Laplace smoothing, 1000 × 80000
  samples, 10 dead codes replaced per iteration by the most significant samples); model set to the
  official dequantised values. VQ fine-tune 40001–45000: codebook gradient = Σ member SH
  gradients (post-accumulate-grad hooks), fixed assignment, all attributes trainable, fresh
  AdamW (wd 0.01), LR ×0.95 per 400 its. At the last step the official `extreme_saving` zip is
  written in memory (codebook fp16, 13-bit indices, packbits mask, non-VQ SH fp16, opacity/
  scale/rotation fp16, xyz fp32) and decoded back into the params, so checkpoint, validation and
  `ss-eval` measure the decoded model.

## Size
`tools/ssenv python reproductions/lightgaussian/plugin/report_size.py <run_dir> [--export]`
prints `size_MiB` (= official `os.path.getsize(zip)/1024/1024`) and per-file bytes; also the
`encode` line of the training log.

## Fidelity gaps
See `reproductions/lightgaussian/plan.md` (VQ fine-tune hyper-parameters are not released;
teacher = pruned model per Alg. 1; white background; random view sampling; gsplat projection).

## Commands
```bash
cd reproductions/lightgaussian/plugin && ../../../tools/ssenv pip install -e .
tools/ssenv ss-verify lightgaussian --device cuda
tools/ssenv ss-train lightgaussian --data ~/datasets/mipnerf360/native/garden --output runs/x \
  --steps 45000 --recipe 3dgs --num-gaussians 10000000 --max-gaussians 12000000
```
