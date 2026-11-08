# LightGaussian — reproduction report

**Outcome: matched** (both scenes within ±0.5 dB of the paper's per-scene final-model PSNR). Not separately reviewed.

Paper: Fan et al., *LightGaussian*, NeurIPS 2024, arXiv 2311.17245. Port of the official code
VITA-Group/LightGaussian @6676b98 (prune.py, prune_finetune.py, distill_train.py, vectree/), plus the
significance rule from compress-diff-gaussian-rasterization @9893d8c. Both are under the Inria
non-commercial research license. The significance rule runs as a CUDA kernel on gsplat tile lists.
Plugin: `plugin/` (methods `lightgaussian`, `lightgaussian-released`, `lightgaussian-smoke`).

## Protocol
The benchmark is one 45000-step `ss-train` run per scene (`runs/lightgaussian-mipnerf360`, recipe 3dgs):
- steps 1–30000: 3DGS, the same recipe as `baseline3dgs-mipnerf360-30000`;
- step 30000: significance prune of 66%, then 5k steps of co-adaptation;
- SH 3→2 distillation for 5k steps, with pseudo views;
- significance-weighted VQ of the SH;
- a 5k-step VQ fine-tune;
- the official `extreme_saving` zip.

`ss-eval` measures the **decoded** compressed model on llffhold8 test views. Size is the official zip
bytes / 1024² (`plugin/report_size.py`). Paper targets come from appendix Tab. 6–8 ("Ours", final
compressed model; room 31.399 = Tab. 2 #8/#9 after VQ fine-tune).

## Results (Mip-NeRF 360, test split)
| scene | model | PSNR | SSIM | LPIPS | #Gaussians | size |
|---|---|---|---|---|---|---|
| garden | **ours (lightgaussian, decoded)** | **26.81** | 0.855 | 0.146 | 1.62M (from 4.76M) | 73.6 MiB |
| garden | paper LightGaussian | 26.961 | 0.855 | 0.122 | – | – (not per scene) |
| garden | our gsplat baseline 30k | 26.89 | 0.860 | 0.131 | 4.76M | – |
| garden | paper 3D-GS* | 27.29 | 0.862 | 0.108 | – | – |
| room | **ours (lightgaussian, decoded)** | **31.39** | 0.917 | 0.303 | 591k (from 1.74M) | 26.9 MiB |
| room | paper LightGaussian | 31.399 | 0.916 | 0.232 | – | 21 MB (Tab. 2) |
| room | our gsplat baseline 30k | 31.52 | 0.916 | 0.298 | 1.73M | – |
| room | paper 3D-GS* | 31.336 | 0.917 | 0.221 | – | 365 MB (Tab. 2) |

PSNR gap to the paper: garden −0.16 dB, room −0.01 dB. Change from our own 3DGS baseline: −0.09 / −0.13 dB.
The paper's change from its 3DGS: −0.33 / +0.06 dB.

Our LPIPS is higher than the paper's on both scenes, and the baseline's LPIPS is too. This comes from the
known LPIPS implementation difference (`lpips` VGG vs `lpipsPyTorch`), not from the compression.

Sizes are larger than the paper's. Room is 26.9 MiB vs 21 MB. Garden is 73.6 MiB against a paper
9-scene mean of 45 MB, and the paper gives no per-scene sizes. Likely causes:
- more Gaussians survive the prune (our 3DGS has 1.74M on room, 66% pruned → 591k);
- xyz is stored as fp32, as in the released code.

Training took 126 min (garden) and 120 min (room).

VQ statistics:
- garden: 971k of 1.62M Gaussians quantised, 4501 codes used, non-VQ significance share 0.85;
- room: 354k of 591k Gaussians quantised, 2816 codes used, non-VQ significance share 0.95.

Smoke (2k compressed schedule, garden): 22.20 dB vs gsplat 23.05 dB at 2k. This is expected, because
densification stops at the step-1000 prune.

## Deviations from paper / official code
1. **Distillation teacher = the pruned + co-adapted 35k model** (paper Alg. 1). The official script's
   placeholder names a `chkpnt30000` teacher, which would be the unpruned 3DGS model.
2. **VQ fine-tune optimiser is our choice.** The fine-tune is described in the paper (§3.4, supp.)
   but is not in the released code. We use:
   - fresh AdamW (weight decay 0.01, eps 1e-15) with the 3DGS LRs, ×0.95 every 400 steps;
   - a means LR on the 3DGS exponential schedule with max_steps 45000;
   - codebook LRs equal to the SH LRs (0.0025 / 0.000125);
   - a fixed assignment, with the codebook gradient equal to the sum of its members' SH gradients;
   - all attributes trainable, including opacity.

   `lightgaussian-released` (40k, no fine-tune) gives the released-code path. It was not benchmarked.
3. **Code over paper:**
   - distillation loss L1 + 0.2 D-SSIM (paper Eq. 5: L2);
   - pseudo-view σ = 0.05, translation only, on steps with k % 3 ≠ 0 (paper σ = 0.1);
   - kaiming-uniform codebook init with weighted EMA and 10 dead codes replaced per iteration (paper: K-means);
   - V/V₉₀ volume normalisation without the min(·, 1) clip;
   - xyz stored fp32 (paper: fp16);
   - SH features 27-d including DC;
   - opacity frozen during distillation, fresh AdamW (wd 0.01) for distillation, Adam state kept
     for co-adaptation.
4. **Runtime differences:**
   - white background (official: black);
   - random-with-replacement view sampling;
   - SSIM with valid convolution, LPIPS from the `lpips` package;
   - gsplat projection (pixel centres at +0.5) in the significance kernel, with atomics where the
     official kernel uses a racy `+=`;
   - prune runs after the optimizer step of 30000 (official: inside 30001);
   - AdamW decay applied after the Adam update (an O(lr²) difference);
   - SH degree 2 expressed as zeroed degree-3 coefficients;
   - 3DGS stage = our gsplat baseline recipe.
5. FPS was not measured.
6. The paper's per-scene 9-scene mean (27.26) does not match its Tab. 1 (27.13). We matched per scene.
