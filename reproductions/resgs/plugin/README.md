# splatstudio-method-resgs

Native SplatStudio port of **ResGS: Residual Densification of 3D Gaussian for Efficient Detail
Recovery** (Lyu et al., ICCV 2025, arXiv 2412.07494), released "Ours (AbsGS)" variant of
https://github.com/yanzhelyu/ResGS (commit a0db69d). Paper fidelity unverified.

## Mechanism -> code
- Residual split (Eq. 4-5): `density.residual_split` — every Gaussian with mean abs gradient
  >= its threshold keeps its mean/scale/SH, gets opacity x beta (0.3, opacity Adam moments
  zeroed), and a child is appended with scale / lambda_s (1.6), mean ~ N(mu, Sigma), parent's
  pre-reduction opacity, same rotation/SH, zero Adam moments, level + 1. Then prune opacity <
  0.005. No clone/split, no size pruning. Implemented with `gsplat.strategy.ops`
  (`_update_param_with_optimizer`, `remove`).
- Varying threshold (Eq. 8): `density.thresholds` — tau * alpha^(l-k) for l < k, alpha = 2^(1/3),
  tau = 0.00067 (abs gradient, NDC-normalised as in 3DGS/gsplat).
- Image pyramid: `pyramid.py` (exact torch port of cv2.pyrDown), `renderer.py` renders training
  views at pyrDown^{2,1,0} size with FoV kept, `losses.py` supervises against the pyramid image;
  evaluation always full resolution.
- Schedule: `schedule.py` is a literal port of the official train.py counters: stages change at
  2500 / 6000, 500-iteration warm-ups without stats/densification, substage counter
  (levels 1..9 first used at 800, 1400, 2000, 3000, 4000, 5000, 6000, 8400, 10200),
  densification every 100 in (700, 12000).
- AbsGS host: absolute gradients; opacity x0.8 (moments zeroed) every 500 from 2500 to 24500,
  preceded by pruning < 0.005 from 12000 on; no opacity reset.
- Official learning rates: means 1.28e-4 (x extent, x0.01 decay over the run), sh0 0.002,
  shN 0.0001, opacity 0.04, scaling 0.006, rotation 0.001. Same-padded D-SSIM (3DGS).

All values are in `density.DEFAULTS` / component configs.

## Fidelity gaps
White background (official black); with-replacement view sampling; optimizer step before
density ops (runtime order); K scaled by w'/W for pyramid cameras (sub-pixel principal point
difference vs 3DGS FoV camera); gsplat rasterizer instead of the AbsGS 3DGS fork; evaluation
SSIM valid-padded (runtime).

## Commands
```
tools/ssenv pip install -e reproductions/resgs/plugin
tools/ssenv ss-verify resgs --device cuda
tools/ssenv ss-train resgs --data ~/datasets/mipnerf360/native/garden --output runs/x --recipe 3dgs --num-gaussians 10000000
```
