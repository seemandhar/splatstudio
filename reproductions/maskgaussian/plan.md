# Plan: MaskGaussian native port

Method name `maskgaussian` (Ours-α, the per-scene setting), module `ss_maskgaussian`.
Extra registered variants: `maskgaussian-beta`, `maskgaussian-gamma` (paper settings β/γ),
`maskgaussian-opacitymask` (paper ablation "Mask × opacity": same everything, mask multiplies
opacity instead of entering the blend) — optional controls, not part of the benchmark target.

## Components
| kind | name | factory | requires → provides | config (paper value) |
|---|---|---|---|---|
| renderer | maskgs-renderer | ss_maskgaussian.renderer:create_renderer | state.gaussian3d, camera.pinhole → render.rgb, gradient.screen.signed | background [0,0,0]; score_offset [10,1]; tau 1.0; accum "official"\|"exact"; mask_mode "raster"\|"opacity" |
| density | maskgs-density | ss_maskgaussian.density:create_density | gradient.screen.signed | 3DGS ADC (500/15000/100, 2e-4, 0.01, 0.005, reset 3000, big-world 0.1 after 3000); prune_samples 10; mask_prune_every 1000; total_steps 30000; final_sample true |
| loss | photometric | splatstudio.losses:create_photometric (L1 + 0.2 D-SSIM, unchanged) | render.rgb | – |
| loss | maskgs-mask-loss | ss_maskgaussian.losses:create_mask_loss | render.rgb (reads info["mask"]) | lambda 0.1, from 19000, until 20000 (inclusive) |

## Fields
| name | shape | lr | init | density handling |
|---|---|---|---|---|
| mask_scores | (2,) | 0.01 | 0.0 (+ fixed offset [10,1] in renderer = official init [10,1]; Adam is shift-invariant) | copied on clone/split, removed on prune |

## Mechanism
- Renderer (training, grad enabled): logits = mask_scores + offset; M = F.gumbel_softmax(
  log_softmax(logits), tau, hard=True)[:, :1] (straight-through, code gaussian_model.py:160);
  vendored mask-diff-gaussian-rasterization (540dc11) JIT-built from csrc/; autograd Function
  returns dL/dM from the kernel; info["mask"] = M (graph), info["mask_hard"]. Eval (no grad):
  masks = 1 (official render_orig_3dgs on the pruned model; masked kernel with M=1 is the stock
  rasterizer).
- `accum: exact` compiles the same sources with -DMASKGS_EXACT_ACCUM, which uses the mask of
  the Gaussian behind (last_mask) in the accum_rec recursion = supplementary Eq. b_i. Default
  `official` = unmodified kernel (produced the paper numbers).
- Density: speedy-splat-style faithful 3DGS ADC (Inria clone/split/prune), with the prune set
  of densify_and_prune extended by "never sampled in 10 Gumbel draws"; for step ≥ 15000 and
  step % 1000 == 0: mask_prune; at step == total_steps: keep one Gumbel sample (save_ply_default).
- Mask loss: λ(step) · mean(M)^2, λ(step) = λ if from ≤ step ≤ until else 0 (simple_scheduler).

## Probes
| probe | contribution | check | fails if |
|---|---|---|---|
| maskgs-raster-exact | C1 | `exact` kernel: image and dL/dM, dL/dopacity, colour grads equal float64 autograd of dense Eq. 3–4 compositor; `official` kernel equals an independent float64 back-to-front replica of backward.cu's recursion and differs from exact only when adjacent masks differ; M=1 image equals stock blending | mask ignored in forward/backward, wrong accumulation |
| maskgs-unsampled-gradient | C1 | through the full renderer + Gumbel ST, Gaussians with sampled M=0 that overlap pixels get nonzero mask_scores grad and do not change the image; the same check run on the naive `opacity` mode (opacity × M) finds exactly zero grads and is reported as failing | masked Gaussians get no gradient |
| maskgs-gumbel | C2 | empirical P(M=1) over 20000 draws matches softmax(logits)[0] within 4σ for several logits; d(M)/d(score) equals the analytic soft-sample Jacobian (straight-through); init offset gives p=σ(9) | deterministic/thresholded mask, no ST gradient |
| maskgs-mask-loss | C3 | loss = λ mean(M)^2 at steps inside window, 0 at from−1 and until+1; grad wrt M = 2λ mean/N | wrong schedule / L1 loss |
| maskgs-prune-schedule | C4 | on a fixture: densify steps remove never-sampled Gaussians (scores forcing p≈0), keep p≈1 ones; mask-prune at 15000,16000,… only; final step keeps a single sample; optimizer state aligned; mask_scores copied on clone/split | missing prune, wrong cadence |

## Files
plugin/pyproject.toml, ss_maskgaussian/{__init__,raster,renderer,density,losses,probes}.py,
ss_maskgaussian/csrc/maskrast (vendored, + LICENSE.md, #ifdef patch), csrc/third_party/glm,
README.md, tests/test_maskgaussian.py.

## Unchanged from 3DGS
Photometric loss, LRs, SH schedule, opacity reset, ADC thresholds; black background (official).

## Fidelity gaps
1. Runtime order: optimizer step happens before density (official: densify then step) — same
   as all native 3DGS-style ports; negligible.
2. Random view sampling with replacement (runtime) vs official shuffled stack — small.
3. Means LR schedule: runtime exponential 1.6e-4→1.6e-6 ×extent (official same, no delay) – none.
4. Final iteration optimizer step: official skips it at 30000; runtime steps — negligible.
5. GPU RNG streams differ → different Gumbel samples; final model is a random sample (as official).
6. Evaluation SSIM/LPIPS implementation differences (runtime-wide).
