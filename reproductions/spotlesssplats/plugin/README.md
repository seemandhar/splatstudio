# splatstudio-method-spotlesssplats

Native SplatStudio port of **SpotLessSplats: Ignoring Distractors in 3D Gaussian Splatting**
(Sabour, Goli et al., ACM TOG; [arXiv 2406.20055](https://arxiv.org/abs/2406.20055)), following
the official gsplat trainer [lilygoli/SpotLessSplats](https://github.com/lilygoli/SpotLessSplats)
(`examples/spotless_trainer.py`, commit 0caae3c, Apache-2.0). Paper fidelity unverified.

| method | paper row |
|---|---|
| `spotlesssplats` | SLS-mlp w/ UBP (headline, Fig. 3–5, Fig. 8 last row) |
| `spotlesssplats-noubp` | SLS-mlp w/o UBP (Fig. 8) |
| `spotlesssplats-agg` | SLS-agg (Fig. 8; official `--cluster`, no UBP) |

## Mechanism → code
- `losses.py: sls-robust` — masked L1 `mean(M ⊙ |clamp(Î)−I|)` (Eq. 1; replaces L1+D-SSIM, official
  `ssim_lambda=0`); RobustNeRF mask `inlier OR box3x3(inlier)>0.5` at the τ=0.7 residual quantile
  (Eq. 2); 10000-bin histogram discounted by 0.95 per step, frozen at t=15000 (§4.2.2); SLS-mlp:
  `Linear(1280+80,16)-ReLU-Linear(16,1)-Sigmoid` on bilinear-upsampled SD features + 20-frequency
  PE (first layer applied at 50×50 before the upsampling — mathematically identical, probe
  `mlp-equivalence`); SLS-agg: 100-cluster inlier-fraction vote (Eq. 3–4); Bernoulli warm-up
  `M ~ B(α + (1−α)M*)`, `α = exp(−3e-3·⌊(t+1)/1.5⌋)` (Eq. 9, code value of β1).
- `losses.py: sls-mlp` — hinge `mean(relu(p − M_τ=.9) + relu(M_τ=.5 − p)) + 0.5·reg` (Eq. 6–8);
  the reg (max|W1|·max|W2|) has no gradient, as in the official code. MLP Adam lr 1e-3.
- `density.py: sls-density` — official refine loop (dup/split grad2d ≥ 2e-4 every 100 steps for
  500 < t < 15000, prune opacity < 0.005, no opacity reset, no scale/screen prune), UBP (Eq. 10:
  per-pixel squared screen-mean gradients from the JIT CUDA kernel `csrc/ubp.cu`, summed over 100
  steps, prune < 1e-14, new Gaussians utilisation 1), SH-rest clamp to ≤ 0.001 at t = 8002.
- `renderer.py: sls-renderer` — gsplat `rasterization` RGB on black (official), SH colours evaluated
  explicitly (same formula as gsplat), hook storing dL/dÎ for UBP.
- `csrc/ubp.cu` — derived from gsplat 1.5.3 `RasterizeToPixels3DGSBwd.cu` (Apache-2.0); matches
  brute-force per-pixel autograd to 1e-7 relative (probe `ubp-kernel`).

Steps: the runtime's 1-based step k is the official 0-based t = k−1 in every schedule.

## Data
Per-training-image SD 2.1 features `<scene>/SD/<stem>.npy` [1280,50,50] (authors' release,
Borealis doi:10.5683/SP3/WOFXFT) and, for agg, `<scene>/SLS_clusters/<stem>.npy`. Scenes are built
by `reproductions/spotlesssplats/tools/prepare_spotless.py` (official Parser: images_8, cv2
undistortion, world normalisation; train=`clutter`, test=`extra`, val = copies of test) and
`make_clusters.py`; `extract_sd_features.py` recomputes features for new scenes (SD 2.1 weights at
`~/models/sd21`). The data root is taken from `ss-train --data` (or `SLS_DATA`).

```bash
tools/ssenv ss-train spotlesssplats --data ~/datasets/spotless/native/android --output runs/x \
  --steps 30000 --recipe 3dgs --num-gaussians 10000000
```

## Fidelity gaps
GLO appearance (paper Eq. 11) not implemented (absent from the official code too); view sampling
with replacement (official: shuffled epochs); optimizer step precedes density (official refines
before the step); scene scale from training cameras only; identity initial quaternions; single
seed (paper: mean of 5); LPIPS-VGG (official code AlexNet); paper β1 = 3e-4 vs code 3e-3 (code
used); paper UBP threshold 1e-8 vs code 1e-14 (code used).
