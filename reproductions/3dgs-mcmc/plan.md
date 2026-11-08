# 3DGS-MCMC (plan id 25) — faithful config vs built-in gsplat-mcmc

Official code: github.com/ubc-vision/3dgs-mcmc @7b4fc9f (read-only clone in scratch).
gsplat 1.5.3 MCMCStrategy relocate/sample_add/inject_noise match official gaussian_model.py /
train.py (same relocation kernel, multinomial by opacity, Adam moments zeroed at sampled idx,
+5 % births, grid (500,25000) %100, noise op_sigmoid(1-o; k=100, x0=0.995) * noise_lr * lr * Σ).

Built-in `gsplat-mcmc` deviations (all fixed in plugin `mcmc-3dgs`):
| item | official | built-in | mcmc-3dgs |
|---|---|---|---|
| cap_max | per-scene configs (bicycle 5.9M, garden 5.2M, stump 4.75M, room 1.5M, counter 1.2M, kitchen 1.8M, bonsai 1.3M, truck 2.6M, train 1.1M, drjohnson 3.4M, playroom 2.5M) | 1M everywhere | per scene from --data basename |
| opacity_reg / scale_reg | 0.01 / 0.01 (DB opacity 0.001) | none | same as official |
| init opacity | 0.5 | 0.1 (runtime) | 0.5 |
| init scale | log(0.1·sqrt(mean 3-NN d²)) | 1× 3-NN (recipe 3dgs) | 0.1× |
| background | black | white | black |
| images | Inria -r 4 / -r 2 PIL resize of full-res | (dataset choice) | bench on mipnerf360-pilr |

Remaining (runtime) gaps: views sampled with replacement (official: shuffled stack);
relocation runs after the Adam step (official: before); noise uses the post-decay lr; SH degree
increments at steps 1001/2001/3001 (official 1000/2000/3000); gsplat rasterizer vs official
diff-gaussian-rasterization; single seed (paper: mean of 3 runs); LPIPS offset of our runtime.
