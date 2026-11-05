# splatstudio-method-coherentgs

Native SplatStudio port of **CoherentGS: Sparse Novel View Synthesis with Coherent 3D
Gaussians** (Paliwal et al., ECCV 2024, [arXiv 2403.19495](https://arxiv.org/abs/2403.19495),
official code [avinashpaliwal/CoherentGS](https://github.com/avinashpaliwal/CoherentGS)).
Paper fidelity unverified until benchmarked.

## Mechanism → code
| paper | code |
|---|---|
| one Gaussian per input pixel on its ray, x = g(D_init + ΔD, p) (Eq. 3) | `bank.Bank.xyz/gaussians`, `renderer.render_train` |
| implicit decoders f_φ(n) for ΔD and Δα, 5-channel depth masks (§4.1, supp. 1) | `decoder.ImplicitDecoder` (official architecture), `bank.masked_channel_sum` (50·tanh / 100·tanh) |
| L_multi = (1-λ)TV + λ MTV on 1/(1+depth) (Eq. 4-5) | `losses.create_smooth` (λ = it/20000, weight 5) |
| flow consistency of corresponding Gaussians (Eq. 6) | `bank.warping_loss`, `losses.create_flow` (effective weight 0.01) |
| multisampling (§4.2) | `renderer`: it > 12000 → s = randint(1,4), s× render + AvgPool |
| coarse scale/offset alignment (Eq. 8), radius r = D/(2f)·1.2 (Eq. 9) | `bank.align`, `bank.log_scales`, `density.initialize/after_backward` |
| objective: 0.8 ML1 + 0.2 D-SSIM + 0.05 VGG relu1_1 (code) | `losses.create_photometric` |

Global state (decoders, rays, flows) lives in a process-local bank owned by the density.
Every training render writes the derived means / opacity logits into `params`, so the
checkpoint is a plain Gaussian set and `ss-eval` needs no decoder.

## Data
Dataset `llff-fewview3-cgs` built by `reproductions/coherentgs/tools/prepare_cgs_llff.py` from
the author-processed LLFF archive (README Google-Drive link; Depth Anything + FlowFormer++
outputs). Needs `--num-gaussians >= 3·504·378` so every pixel point is kept, and
`--recipe splatstudio` (the 3dgs recipe's kNN scale init would allocate ~20 GB for 571k points;
scales are overwritten anyway). VGG19 conv1_1 weights: `~/models/coherentgs/vgg19_conv1_1.pt`
(sha256 01808af4…, extracted from torchvision `vgg19-dcbb9e9d.pth`, sha256 dcbb9e9d…).

## Run
```bash
ss-train coherentgs --data ~/datasets/llff/fewview3-cgs/fern --output runs/x --steps 13000 \
    --recipe splatstudio --num-gaussians 10000000 --max-gaussians 12000000
ss-eval --run runs/x --split test                       # unmasked
python -m ss_coherentgs.evaluate --run runs/x           # + paper's masked metrics (alpha < 1e-3)
```
Step s = official iteration 7000 + s; the 1000 alignment iterations run inside
`density.initialize`.

## Fidelity gaps
Masked metrics need the plugin tool (fill convention assumed: white in both images); view
sampling with replacement (official: shuffled stack); gsplat rasterizer (alpha clamp 0.999,
straight-through opacity cap for gsplat's dropped gradients) vs FSGS/Inria; runtime Adam steps
shN/scales/quats on the final step; RNG streams differ; only LLFF 3-view is set up.
