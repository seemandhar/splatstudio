# CoherentGS native plan

Method `coherentgs`, module `ss_coherentgs`, package `splatstudio-method-coherentgs`.
Dataset: new `llff-fewview3-cgs` (~/datasets/llff/fewview3-cgs/<scene>) built by
`tools/prepare_cgs_llff.py` from the author-processed LLFF archive (Depth Anything `depth_rel`,
FlowFormer++ flows + consistency weights; images_8 verified pixel-identical to our
nerf_llff_data/images_8).

## Why a new dataset
`llff-fewview3-r8` has the same split names but (a) images are full-res PIL-bicubic /8, not the
official images_8 files, (b) the COLMAP world frame is not the official one — CoherentGS's
hyper-parameters are in absolute units of the official frame (poses_bounds.npy translations
× 1/(0.75·min bound): alignment lrs 1/5, ΔD = 50·tanh, depth init 1/(disp/max+0.2)), and
(c) it needs per-pixel init + depth masks + flows. Files: `images/` (images_8 PNG, all views),
`transforms.json` (official c2w, OpenGL columns; fl = poses_bounds f/8; centred principal point;
train = 3 linspace views, test = every 8th, val = rest), `points.npz` (one point per training
pixel, camera-major/row-major, unaligned mono depth, pixel colours), `coherentgs.npz` sidecar
(train viewmats, z_mono [N,H,W], dep_mask [N,5,H,W], flows [N,N,H,W,2], flow weights [N,N,H,W],
mono depth source hashes).

## Components
| contribution | kind / factory | requires → provides | config (paper/code value) |
|---|---|---|---|
| C1 pixel Gaussians, C2 decoders, C5 multisampling | renderer `ss_coherentgs.renderer:create_renderer` | state.gaussian3d, camera.pinhole → render.rgb, render.depth.expected_z, render.coherent.xyz, render.coherent.disparity | multisample_start 5000 (it 12000), multisample_max 4, alpha_cap 0.999, sh schedule [8000, 9000, 10000] |
| C1/C2/C6 state, decoders, alignment, radius reset | density `ss_coherentgs.density:create_density` | () | n_views 3, mask_channels 5, ngf_depth 15, ngf_opac 10, depth_scale 50, opac_scale 100, init_opacity 0.5, radius_mult 1.2, align_steps 1000, align_lr_scale 1, align_lr_offset 5, decoder lrs (1e-6..1e-4, 1000) / (1e-6..1e-5, 2000), AdamW, lr shN 5e-5 / scales 1e-4 / quats 1e-5, reset_every 100, reset_stop 8000, total_steps 13000, seed 0 |
| photometric (code) | loss `ss_coherentgs.losses:create_photometric` (name `photometric`) | render.rgb | ml1 0.8 (weights 1,1/4,1/8,1/16,1/32), ssim 0.2 (same-padding), vgg 0.05 (relu1_1, /32) |
| C3 L_multi | loss `create_smooth` | render.coherent.disparity | weight 5, gamma = (offset+step)/20000, offset 7000 |
| C4 L_flow | loss `create_flow` | render.coherent.xyz | weight 0.01 (code: 0.1·0.1), threshold 0.5 |

## Global state (runtime has no global learned modules)
Process-local `Bank` owned by the density (registered in `ss_coherentgs.bank.ACTIVE`): xy rays,
c2w, z_init, dep_mask, flows/masks, the two `ImplicitDecoder`s with AdamW + CyclicLR, the posenc
input, step counter and a private RNG. The renderer uses it while grad is enabled (training);
in a no-grad render (validation/test, fresh ss-eval process) it renders `params` only.
Persistence without a blob: on every training render the derived means and opacity logits are
copied (detached) into `params["means"]`/`params["opacities"]`; the final step does not step the
decoders (official `if iteration < opt.iterations`), so the checkpoint holds exactly the final
Gaussians and ss-eval needs no decoder. `means`/`opacities`/`sh0` receive no gradient (DC is fixed
in the official code) so the runtime's Adam leaves them untouched.

## Fields
No custom fields. Base-field LR overrides in density.initialize: shN 5e-5, scales 1e-4,
quats 1e-5 (official training_setup); means/opacities/sh0 are derived/fixed.

## Step mapping
ss-train `--steps 13000`; native step s ↔ official iteration 7000+s. Alignment (1000 Adam
iterations, no rendering) runs inside density.initialize. Radius reset before steps with
(7000+s) % 100 == 0 and 7000+s < 15000 (done in after_backward of s−1). SH degree 0 until
s 8000, then 1/2/3 at s 8000/9000/10000 (renderer ignores the runtime's own SH schedule while
training). Multisampling for s > 5000 (it > 12000).

## Probes
| probe | checks (fails if mechanism removed) |
|---|---|
| ray-structure | means = c2w·(xy·z, z) for z = z_init + 50·chsum(tanh·S/ΣS); moving one view's decoder output changes only that view's Gaussians, along their rays (cross product ≈ 0) |
| decoder-architecture | ImplicitDecoder output shape = (N, C, H, W) for odd H/W; parameter count equals an independent count of the official layer spec; posenc matches an independent sin/cos formula |
| masked-residual | residual is piecewise-coherent: pixels with all-zero mask get 0; one-hot mask selects channel; opacity = sigmoid(logit α0 + 100·…) |
| alignment | coarse alignment on a synthetic 2-view scene with known scale/offset reduces the flow loss ≥10× and recovers the relative scale |
| radius | log-scale = log(tan(fovy/2)·D/H·1.2) against an independent 0.6·D/f formula; reset only on the schedule |
| tv-loss | L_multi equals an independent finite-difference formula; γ boundary at step 1 (0.35005) and 13000 (1.0) |
| flow-loss | equals an independent grid_sample-free bilinear warp formula on a fixture; zero for consistent geometry; weight 0.01 |
| photometric | ML1 + SSIM(same) + VGG relu1_1 against independent formulas |
| multisample-render (CUDA) | s-times supersampled render pooled to H×W differs from s=1 and equals manual pooling; eval (no-grad) render uses params only |
| persistence (CUDA) | after a training render params["means"]/["opacities"] equal the derived values; a no-grad render with bank cleared reproduces the training render at s=1 |

## Fidelity gaps
1. **Masked evaluation**: paper masks pixels with CoherentGS opacity < 1e-3; ss-eval is
   unmasked. Plugin tool `python -m ss_coherentgs.evaluate --run R` adds masked metrics
   (white fill in both images, as loss_utils.ssim(mask=)); fill convention assumed. Expected
   effect: unmasked PSNR noticeably lower (holes are white vs GT content).
2. **View sampling**: runtime samples with replacement; official uses a shuffled stack. Small.
3. **Rasteriser**: gsplat vs Inria/FSGS (alpha clamp 0.999 vs 0.99, gradient drop fixed by a
   straight-through cap). Small.
4. **Final step**: runtime's Adam steps shN/scales/quats on the last step (official skips all
   steps on the last iteration). Negligible.
5. **SSIM / LPIPS in eval**: runtime SSIM uses valid convolution; LPIPS-VGG on [-1,1]. FSGS's eval
   uses the same lpips package (vgg). Small.
6. **Seeds/RNG**: decoder init / shuffles use torch/python RNGs, not bit-identical to the official.
7. Only 3-view LLFF is benchmarked (2/4-view and NVS-RGBD not).
8. Preprocessing reuses the author-provided depth/flow files, so it is exact for LLFF but the
   Depth Anything / FlowFormer++ inference itself is not re-run.

feasible_natively: true (inside components; global state via the process-local bank).
